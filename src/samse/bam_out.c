/* bam_out.c — see bam_out.h. Record encoding follows the SAM/BAM specification (v1) and htslib's
 * choices where the spec leaves one (smallest integer type for 'i' tags, bin of unmapped reads). */
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <zlib.h>
#include <omp.h>
#include "bam_out.h"
#include "khash.h"
#include "utils.h"

KHASH_MAP_INIT_STR(s2i, int)

#define BGZF_IN  0xff00         /* uncompressed bytes per BGZF block (as htslib) */
#define BGZF_OUT 0x10000        /* maximum size of a compressed BGZF block */

struct bam_out_s {
	FILE *fp;
	int level;
	khash_t(s2i) *ref;          /* reference name -> id */
	kstring_t buf;              /* uncompressed bytes not yet written */
};

static void put32(kstring_t *s, int32_t x) { kputsn((const char*)&x, 4, s); }   /* BAM is little-endian, as the hosts */

/* one BGZF block (gzip member with the BC extra field) of n <= BGZF_IN bytes; returns its size */
static int bgzf_block(int level, const uint8_t *in, int n, uint8_t *out)
{
	static const uint8_t hdr[16] = { 0x1f, 0x8b, 8, 4, 0, 0, 0, 0, 0, 0xff, 6, 0, 'B', 'C', 2, 0 };
	z_stream z;
	int ret, clen, bsize;
	uint32_t crc, isize = n;
	for (;;) {
		memset(&z, 0, sizeof(z));
		deflateInit2(&z, level, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY);
		z.next_in = (Bytef*)in; z.avail_in = n;
		z.next_out = out + 18; z.avail_out = BGZF_OUT - 18 - 8;
		ret = deflate(&z, Z_FINISH);
		clen = (int)z.total_out;
		deflateEnd(&z);
		if (ret == Z_STREAM_END) break;
		if (level == 0) _err_fatal_simple(__func__, "cannot compress a BGZF block");
		level = 0;              /* incompressible data: store it (always fits) */
	}
	bsize = 18 + clen + 8;
	memcpy(out, hdr, 16);
	out[16] = (bsize - 1) & 0xff; out[17] = (bsize - 1) >> 8;
	crc = crc32(crc32(0L, Z_NULL, 0), in, n);
	memcpy(out + 18 + clen, &crc, 4);
	memcpy(out + 18 + clen + 4, &isize, 4);
	return bsize;
}

/* compress and write the buffered bytes: all full blocks, plus the partial last one if `all` */
static void flush_blocks(bam_out_t *b, int all)
{
	int nb = (int)(b->buf.l / BGZF_IN), i;
	size_t done;
	if (all && b->buf.l % BGZF_IN) ++nb;
	if (nb == 0) return;
	uint8_t *out = (uint8_t*)malloc((size_t)nb * BGZF_OUT);
	int *len = (int*)malloc(nb * sizeof(int));
	#pragma omp parallel for schedule(dynamic, 1)
	for (i = 0; i < nb; ++i) {
		size_t off = (size_t)i * BGZF_IN;
		int n = b->buf.l - off < BGZF_IN? (int)(b->buf.l - off) : BGZF_IN;
		len[i] = bgzf_block(b->level, (const uint8_t*)b->buf.s + off, n, out + (size_t)i * BGZF_OUT);
	}
	for (i = 0; i < nb; ++i) err_fwrite(out + (size_t)i * BGZF_OUT, 1, len[i], b->fp);
	done = all? b->buf.l : (size_t)nb * BGZF_IN;
	memmove(b->buf.s, b->buf.s + done, b->buf.l - done);
	b->buf.l -= done;
	free(out); free(len);
}

bam_out_t *bam_out_open(FILE *fp, int level, const char *hdr, const bntseq_t *bns)
{
	bam_out_t *b = (bam_out_t*)calloc(1, sizeof(bam_out_t));
	int i, absent, l_text = (int)strlen(hdr);
	b->fp = fp;
	b->level = level < 0? 0 : level > 9? 9 : level;
	b->ref = kh_init(s2i);
	for (i = 0; i < bns->n_seqs; ++i) {
		khint_t k = kh_put(s2i, b->ref, bns->anns[i].name, &absent);
		kh_val(b->ref, k) = i;
	}
	kputsn("BAM\1", 4, &b->buf);
	put32(&b->buf, l_text);
	kputsn(hdr, l_text, &b->buf);
	put32(&b->buf, bns->n_seqs);
	for (i = 0; i < bns->n_seqs; ++i) {
		int l = (int)strlen(bns->anns[i].name) + 1;
		put32(&b->buf, l);
		kputsn(bns->anns[i].name, l, &b->buf);   /* includes the NUL */
		put32(&b->buf, bns->anns[i].len);
	}
	flush_blocks(b, 1);                          /* the header in its own blocks, as htslib */
	return b;
}

/* reference id of a name field (not NUL-terminated); -1 for "*" or an unknown name */
static int ref_id(const bam_out_t *b, const char *p, int l)
{
	char tmp[1024], *s = l < 1024? tmp : (char*)malloc(l + 1);
	khint_t k;
	int id;
	if (l == 1 && p[0] == '*') return -1;
	memcpy(s, p, l); s[l] = 0;
	k = kh_get(s2i, b->ref, s);
	id = k == kh_end(b->ref)? -1 : kh_val(b->ref, k);
	if (s != tmp) free(s);
	return id;
}

static int reg2bin(int64_t beg, int64_t end)
{
	--end;
	if (beg >> 14 == end >> 14) return (int)(((1 << 15) - 1) / 7 + (beg >> 14));
	if (beg >> 17 == end >> 17) return (int)(((1 << 12) - 1) / 7 + (beg >> 17));
	if (beg >> 20 == end >> 20) return (int)(((1 << 9) - 1) / 7 + (beg >> 20));
	if (beg >> 23 == end >> 23) return (int)(((1 << 6) - 1) / 7 + (beg >> 23));
	if (beg >> 26 == end >> 26) return (int)(((1 << 3) - 1) / 7 + (beg >> 26));
	return 0;
}

int bam_out_encode(const bam_out_t *b, const char *line, int len, kstring_t *rec)
{
	static const char *ops = "MIDNSHP=X";
	const char *f[11], *p = line, *end, *q;
	int fl[11], i, refid, mrefid, n_cigar = 0, l_seq, flag, mapq;
	int64_t pos, mpos, tlen, rlen = 0;
	size_t start = rec->l, at;
	if (len > 0 && line[len - 1] == '\n') --len;
	end = line + len;
	for (i = 0; i < 11; ++i) {
		q = (const char*)memchr(p, '\t', end - p);
		if (!q) { if (i < 10) return -1; q = end; }
		f[i] = p; fl[i] = (int)(q - p);
		p = q < end? q + 1 : end;
	}
	if (fl[0] > 254) return -1;
	flag = atoi(f[1]);
	refid = ref_id(b, f[2], fl[2]);
	pos = strtoll(f[3], 0, 10) - 1;
	mapq = atoi(f[4]);
	if (fl[6] == 1 && f[6][0] == '=') mrefid = refid;
	else mrefid = ref_id(b, f[6], fl[6]);
	mpos = strtoll(f[7], 0, 10) - 1;
	tlen = strtoll(f[8], 0, 10);
	l_seq = (fl[9] == 1 && f[9][0] == '*')? 0 : fl[9];
	if (!(fl[5] == 1 && f[5][0] == '*'))
		for (i = 0; i < fl[5]; ++i) if (f[5][i] < '0' || f[5][i] > '9') ++n_cigar;

	put32(rec, 0);                               /* block_size, set at the end */
	put32(rec, refid);
	put32(rec, (int32_t)pos);
	kputc(fl[0] + 1, rec);                       /* l_read_name */
	kputc(mapq, rec);
	at = rec->l; kputsn("\0\0", 2, rec);         /* bin, set once the reference length is known */
	{ uint16_t x = n_cigar; kputsn((const char*)&x, 2, rec); }
	{ uint16_t x = flag; kputsn((const char*)&x, 2, rec); }
	put32(rec, l_seq);
	put32(rec, mrefid);
	put32(rec, (int32_t)mpos);
	put32(rec, (int32_t)tlen);
	kputsn(f[0], fl[0], rec); kputc(0, rec);
	if (n_cigar) {                               /* CIGAR: len<<4 | op */
		const char *c = f[5], *ce = f[5] + fl[5];
		while (c < ce) {
			uint32_t l = 0, op;
			const char *o;
			while (c < ce && *c >= '0' && *c <= '9') l = l * 10 + (*c++ - '0');
			if (c >= ce || !(o = strchr(ops, *c))) return -1;
			op = (uint32_t)(o - ops); ++c;
			if (op == 0 || op == 2 || op == 3 || op == 7 || op == 8) rlen += l;
			l = l << 4 | op;
			kputsn((const char*)&l, 4, rec);
		}
	}
	{
		uint16_t bin = (uint16_t)reg2bin(pos, pos + (rlen? rlen : 1));
		memcpy(rec->s + at, &bin, 2);
	}
	for (i = 0; i < l_seq; i += 2) {             /* SEQ: 4-bit codes of "=ACMGRSVTWYHKDBN", two per byte */
		static const char *nt16 = "=ACMGRSVTWYHKDBN";
		const char *a, *bb;
		int hi, lo;
		a = strchr(nt16, f[9][i] >= 'a'? f[9][i] - 32 : f[9][i]);
		hi = a? (int)(a - nt16) : 15;
		lo = 0;
		if (i + 1 < l_seq) {
			bb = strchr(nt16, f[9][i + 1] >= 'a'? f[9][i + 1] - 32 : f[9][i + 1]);
			lo = bb? (int)(bb - nt16) : 15;
		}
		kputc(hi << 4 | lo, rec);
	}
	if (fl[10] == 1 && f[10][0] == '*') for (i = 0; i < l_seq; ++i) kputc(0xff, rec);
	else {
		if (fl[10] != l_seq) return -1;
		for (i = 0; i < l_seq; ++i) kputc(f[10][i] - 33, rec);
	}
	while (p < end) {                            /* optional fields: XX:A:c, XX:i:n, XX:Z:text */
		const char *t = p, *te;
		q = (const char*)memchr(p, '\t', end - p);
		te = q? q : end;
		p = q? q + 1 : end;
		if (te - t < 5 || t[2] != ':' || t[4] != ':') return -1;
		kputsn(t, 2, rec);
		if (t[3] == 'A') { kputc('A', rec); kputc(t[5], rec); }
		else if (t[3] == 'Z') { kputc('Z', rec); kputsn(t + 5, (int)(te - t - 5), rec); kputc(0, rec); }
		else if (t[3] == 'i') {
			long long x = strtoll(t + 5, 0, 10);
			if (x < 0) {
				if (x >= INT8_MIN) { int8_t v = (int8_t)x; kputc('c', rec); kputsn((const char*)&v, 1, rec); }
				else if (x >= INT16_MIN) { int16_t v = (int16_t)x; kputc('s', rec); kputsn((const char*)&v, 2, rec); }
				else { int32_t v = (int32_t)x; kputc('i', rec); kputsn((const char*)&v, 4, rec); }
			} else {
				if (x <= UINT8_MAX) { uint8_t v = (uint8_t)x; kputc('C', rec); kputsn((const char*)&v, 1, rec); }
				else if (x <= UINT16_MAX) { uint16_t v = (uint16_t)x; kputc('S', rec); kputsn((const char*)&v, 2, rec); }
				else { uint32_t v = (uint32_t)x; kputc('I', rec); kputsn((const char*)&v, 4, rec); }
			}
		} else return -1;
	}
	{
		int32_t bs = (int32_t)(rec->l - start - 4);
		memcpy(rec->s + start, &bs, 4);
	}
	return 0;
}

void bam_out_write(bam_out_t *b, int n, const kstring_t *recs)
{
	int i;
	for (i = 0; i < n; ++i) kputsn(recs[i].s, (int)recs[i].l, &b->buf);
	flush_blocks(b, 0);
}

void bam_out_close(bam_out_t *b)
{
	static const uint8_t eof[28] = { 0x1f, 0x8b, 8, 4, 0, 0, 0, 0, 0, 0xff, 6, 0, 'B', 'C', 2, 0, 0x1b, 0,
	                                 3, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	flush_blocks(b, 1);
	err_fwrite(eof, 1, 28, b->fp);
	kh_destroy(s2i, b->ref);
	free(b->buf.s);
	free(b);
}
