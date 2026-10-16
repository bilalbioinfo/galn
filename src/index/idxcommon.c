/* idxcommon.c -- shared helpers for the G-ALN index builders. See idxcommon.h. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <zlib.h>
#include "idxcommon.h"
#include "bwa_upstream/bntseq.h"
#include "bwa_upstream/utils.h"

double galn_realtime(void)
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return tp.tv_sec + tp.tv_usec * 1e-6;
}

static void die(const char *msg, const char *fn)
{
	fprintf(stderr, "[galn-index] ERROR: %s: %s\n", msg, fn);
	exit(1);
}

int64_t galn_pac_len(const char *fn_pac) /* == bwa_seq_len() */
{
	FILE *fp = fopen(fn_pac, "rb");
	int64_t pac_len;
	unsigned char c;
	if (!fp) die("cannot open", fn_pac);
	if (fseeko(fp, -1, SEEK_END) != 0) die("cannot seek", fn_pac);
	pac_len = ftello(fp);
	if (fread(&c, 1, 1, fp) != 1) die("cannot read", fn_pac);
	fclose(fp);
	return (pac_len - 1) * 4 + (int)c;
}

uint8_t *galn_read_pac(const char *fn_pac, int64_t *l_pac)
{
	int64_t l = galn_pac_len(fn_pac), sz = (l >> 2) + ((l & 3) ? 1 : 0), got = 0;
	uint8_t *pac = (uint8_t*)malloc(sz > 0 ? sz : 1);
	FILE *fp = fopen(fn_pac, "rb");
	if (!fp || !pac) die("cannot open/alloc", fn_pac);
	while (got < sz) {
		size_t r = fread(pac + got, 1, (size_t)(sz - got < (1 << 26) ? sz - got : (1 << 26)), fp);
		if (r == 0) die("short read", fn_pac);
		got += r;
	}
	fclose(fp);
	*l_pac = l;
	return pac;
}

void galn_pac_to_text(const uint8_t *pac, int64_t l_pac, uint8_t *T)
{
	int64_t i, n = 2 * l_pac;
#pragma omp parallel for schedule(static)
	for (i = 0; i < l_pac; ++i) {
		uint8_t c = pac[i >> 2] >> ((~i & 3) << 1) & 3; /* _get_pac() */
		T[i] = c;
		T[n - 1 - i] = 3 - c; /* bns_fasta2bntseq(): reverse complement appended */
	}
}

void galn_pack_bwt(const uint8_t *L, uint64_t n, uint32_t *buf, uint64_t L2[5])
{
	int64_t nb = (int64_t)((n + 127) / 128), b;
	uint64_t nw = (n + 15) / 16, acc[4] = {0, 0, 0, 0};
	uint64_t *cnt = (uint64_t*)malloc(sizeof(uint64_t) * 4 * (nb + 1));
	/* pass 1: pack 2-bit symbols (MSB first within each u32, as bwt_pac2bwt) and count per 128-block */
#pragma omp parallel for schedule(static)
	for (b = 0; b < nb; ++b) {
		uint64_t s = (uint64_t)b * 128, e = s + 128 < n ? s + 128 : n, i;
		uint64_t c[4] = {0, 0, 0, 0};
		uint32_t *w = buf + (uint64_t)b * 16 + 8;
		int j, nwb = (int)((e - s + 15) / 16);
		for (j = 0; j < nwb; ++j) w[j] = 0;
		for (i = s; i < e; ++i) {
			w[(i - s) >> 4] |= (uint32_t)L[i] << ((15 - (i & 15)) << 1);
			++c[L[i]];
		}
		memcpy(cnt + 4 * b, c, sizeof(c));
	}
	/* pass 2: exclusive prefix sums = Occ before each block (bwt_bwtupdate_core) */
	for (b = 0; b < nb; ++b) {
		int k;
		for (k = 0; k < 4; ++k) {
			uint64_t t = cnt[4 * b + k];
			cnt[4 * b + k] = acc[k];
			acc[k] += t;
		}
	}
	memcpy(cnt + 4 * nb, acc, sizeof(acc));
#pragma omp parallel for schedule(static)
	for (b = 0; b < nb; ++b)
		memcpy(buf + (uint64_t)b * 16, cnt + 4 * b, 4 * sizeof(uint64_t));
	memcpy(buf + 8 * (uint64_t)nb + nw, acc, 4 * sizeof(uint64_t)); /* the last element */
	L2[0] = 0;
	L2[1] = acc[0];
	L2[2] = L2[1] + acc[1];
	L2[3] = L2[2] + acc[2];
	L2[4] = L2[3] + acc[3];
	free(cnt);
}

static void fwrite_all(const void *p, size_t size, size_t nmemb, FILE *fp, const char *fn)
{
	const char *q = (const char*)p;
	size_t total = size * nmemb, done = 0;
	while (done < total) {
		size_t chunk = total - done < ((size_t)1 << 28) ? total - done : ((size_t)1 << 28);
		size_t r = fwrite(q + done, 1, chunk, fp);
		if (r != chunk) die("write error", fn);
		done += r;
	}
}

int galn_write_bwt(const char *fn, uint64_t primary, const uint64_t L2[5], const uint32_t *buf, uint64_t n_words)
{
	FILE *fp = fopen(fn, "wb");
	if (!fp) die("cannot open for writing", fn);
	fwrite_all(&primary, 8, 1, fp, fn);
	fwrite_all(L2 + 1, 8, 4, fp, fn);
	fwrite_all(buf, 4, n_words, fp, fn);
	if (fclose(fp) != 0) die("close error", fn);
	return 0;
}

int galn_write_sa(const char *fn, uint64_t primary, const uint64_t L2[5], uint64_t n, const uint64_t *samples, uint64_t n_samples)
{
	/* bwt_dump_sa() writes &bwt->sa_intv (an int) with sizeof(bwtint_t)=8 bytes: the int value 32 followed
	 * by the 4 struct-padding bytes before n_sa, which are zero because bwt_t is calloc'ed. */
	uint64_t intv = GALN_SA_INTV;
	FILE *fp;
	if (n_samples != (n + GALN_SA_INTV) / GALN_SA_INTV - 1) die("inconsistent number of SA samples", fn);
	fp = fopen(fn, "wb");
	if (!fp) die("cannot open for writing", fn);
	fwrite_all(&primary, 8, 1, fp, fn);
	fwrite_all(L2 + 1, 8, 4, fp, fn);
	fwrite_all(&intv, 8, 1, fp, fn);
	fwrite_all(&n, 8, 1, fp, fn);
	fwrite_all(samples, 8, n_samples, fp, fn);
	if (fclose(fp) != 0) die("close error", fn);
	return 0;
}

int64_t galn_fasta_to_pac(const char *fn_fa, const char *prefix)
{
	gzFile fp = xzopen(fn_fa, "r");
	int64_t l_pac = bns_fasta2bntseq(fp, prefix, 1);
	err_gzclose(fp);
	return l_pac;
}
