/* fasta_pac.c: FASTA -> <prefix>.pac/.ann/.amb on all host threads, byte-identical to BWA 0.7.19's
 * bns_fasta2bntseq(for_only=1), i.e. to what `bwa index` writes.
 *
 * The input is read whole into memory (a plain file with parallel reads; gzip files and pipes through zlib,
 * as upstream reads them), then:
 *   1. lines starting with '>' or '@' split it into records, as in kseq_read() ('+' would start a FASTQ quality);
 *   2. each header is cut into name and comment as kseq_read() does;
 *   3. the sequence bytes are cut into pieces of equal size: a first pass counts the bases and the ambiguous
 *      bases of each piece, a second packs the bases at their final offsets. An ambiguous base gets lrand48()&3
 *      from the generator advanced to the piece's first draw (seed 11, as upstream), so it gets the same value
 *      as in the serial code; runs of the same ambiguous character make the .amb holes.
 * Inputs this does not reproduce (FASTQ records, NUL or non-ASCII bytes in a sequence, a sequence of 2^31 bases
 * or more) are read by the upstream code instead.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/stat.h>
#include <zlib.h>
#include <omp.h>
#include "idxcommon.h"
#include "bwa_upstream/bntseq.h"
#include "bwa_upstream/utils.h"

#define LCG_MASK ((1ULL << 48) - 1)
#define LCG_A 0x5DEECE66DULL
#define LCG_C 0xBULL

typedef struct { int64_t rs, re, len; char *name, *anno; int32_t n_ambs; } rec_t;   /* sequence bytes [rs, re) */
typedef struct { int64_t offset; int32_t len; char amb; int32_t rid; } hole_t;
typedef struct { int64_t start, end, nb, na, cont, nh, mh; int weird; hole_t *h; } piece_t;   /* [start, end) of the sequence bytes */

static void fatal(const char *fn, const char *msg)
{
	fprintf(stderr, "[galn index] %s: %s\n", fn, msg);
	exit(1);
}

/* The state of lrand48() after k more draws (srand48(11): x = 11<<16 | 0x330e; each draw x = a*x + c mod 2^48) */
static uint64_t lcg_skip(uint64_t x, uint64_t k)
{
	uint64_t a = LCG_A, c = LCG_C, A = 1, C = 0;
	for (; k; k >>= 1) {
		if (k & 1) { A = A * a & LCG_MASK; C = (C * a + c) & LCG_MASK; }
		c = (a + 1) * c & LCG_MASK;
		a = a * a & LCG_MASK;
	}
	return (A * x + C) & LCG_MASK;
}

/* ---- input ---- */

/* The whole input as gzread() gives it to upstream. *rereadable: the file can be opened again by its name. */
static uint8_t *read_all(const char *fn, int64_t *len, int *rereadable)
{
	int fd = strcmp(fn, "-") ? open(fn, O_RDONLY) : 0;
	struct stat st;
	uint8_t *b, m[2];
	int64_t n = 0, cap = 1 << 26;
	gzFile g;
	if (fd < 0) fatal(fn, strerror(errno));
	if (fstat(fd, &st) != 0) fatal(fn, strerror(errno));
	*rereadable = fd != 0 && S_ISREG(st.st_mode);
	if (*rereadable && st.st_size >= 2 && pread(fd, m, 2, 0) == 2 && !(m[0] == 0x1f && m[1] == 0x8b)) {
		/* not gzip: gzread() would copy it unchanged; read it on all threads */
		const int64_t size = st.st_size, chunk = 64 << 20;
		int fail = 0;
		if (!(b = (uint8_t*)malloc(size + 1))) fatal(fn, "out of memory");
#pragma omp parallel for schedule(dynamic, 1) reduction(|:fail)
		for (int64_t o = 0; o < size; o += chunk) {
			int64_t p = o, e = o + chunk < size ? o + chunk : size;
			while (p < e) {
				ssize_t r = pread(fd, b + p, e - p, p);
				if (r <= 0) { fail = 1; break; }
				p += r;
			}
		}
		close(fd);
		if (fail) fatal(fn, "read error");
		*len = size;
		return b;
	}
	if (S_ISFIFO(st.st_mode)) fcntl(fd, F_SETPIPE_SZ, 1 << 20);   /* fewer, larger reads from `cat` and the like */
	if (!(g = gzdopen(fd, "r"))) fatal(fn, "out of memory");
	gzbuffer(g, 1 << 22);
	if (!(b = (uint8_t*)malloc(cap))) fatal(fn, "out of memory");
	for (;;) {
		int r;
		if (cap - n < (1 << 24)) {
			cap *= 2;
			if (!(b = (uint8_t*)realloc(b, cap))) fatal(fn, "out of memory");
		}
		r = gzread(g, b + n, (unsigned)(cap - n - 1 < (1 << 30) ? cap - n - 1 : (1 << 30)));
		if (r < 0) {
			int e;
			const char *msg = gzerror(g, &e);
			fatal(fn, e == Z_ERRNO ? strerror(errno) : msg);
		}
		if (r == 0) break;
		n += r;
	}
	gzclose(g);
	*len = n;
	return b;
}

/* ---- kseq_read() rules ---- */

static inline int is_space(int c) { return c == ' ' || (c >= '\t' && c <= '\r'); }   /* isspace() in the C locale */

static char *copy_str(const uint8_t *s, int64_t l)
{
	char *p = (char*)malloc(l + 1);
	memcpy(p, s, l);
	p[l] = 0;
	return p;
}

/* The header after the '>' (or '@') at h-1: the name up to the first white space; unless that was the line end,
 * the comment up to the line end, without a final '\r' if it is longer than one character (ks_getuntil2()).
 * Returns where the sequence lines start, or -1 if the input ends at the '>' (kseq_read() gives no record). */
static int64_t parse_header(const uint8_t *B, int64_t N, int64_t h, rec_t *r)
{
	int64_t p = h, e;
	int c = 0;
	if (p >= N) return -1;
	for (e = p; e < N && !is_space(B[e]); ++e);
	r->name = copy_str(B + p, e - p);
	p = e;
	if (p < N) c = B[p++];
	r->anno = NULL;
	if (c != '\n' && p < N) {
		const uint8_t *nl = (const uint8_t*)memchr(B + p, '\n', N - p);
		int64_t l;
		e = nl ? nl - B : N;
		l = e - p;
		if (l > 1 && B[e - 1] == '\r') --l;
		if (l > 0) r->anno = copy_str(B + p, l);
		p = nl ? e + 1 : N;
	}
	if (!r->anno) r->anno = strdup("(null)");
	return p;
}

/* Whether kseq drops the '\r' at x of a sequence starting at rs: ks_getuntil2() cuts one '\r' off the end of a line
 * when the sequence is longer than one character by then; a line of one character at the very end of the input is
 * not cut (ks_getuntil2() returns before). */
static int cr_dropped(const uint8_t *B, int64_t N, int64_t x, int64_t rs)
{
	int64_t y;
	if (x + 1 < N && B[x + 1] != '\n') return 0;    /* not at a line end */
	if (B[x - 1] != '\n') return 1;                 /* the line has more characters */
	if (x + 1 == N) return 0;
	for (y = x - 1; y >= rs && B[y] == '\n'; --y);  /* a "\r" line keeps its '\r' if it is the first line */
	return y >= rs;
}

/* The base before x in the sequence starting at rs (0 at the start, as add1()'s `lasts`) */
static int prev_base(const uint8_t *B, int64_t N, int64_t x, int64_t rs)
{
	for (int64_t y = x - 1; y >= rs; --y)
		if (B[y] != '\n' && !(B[y] == '\r' && cr_dropped(B, N, y, rs))) return B[y];
	return 0;
}

/* ---- the sequence bytes, in pieces ---- */

typedef struct {
	const uint8_t *B;
	int64_t N, nrec, *cum;   /* cum[i]: sequence bytes before record i */
	rec_t *rec;
	uint8_t *pac;
} seqs_t;

/* The record whose sequence holds byte s of all sequence bytes */
static int64_t rec_of(const seqs_t *S, int64_t s)
{
	int64_t lo = 0, hi = S->nrec;   /* the last i with cum[i] <= s */
	while (hi - lo > 1) {
		int64_t mid = (lo + hi) / 2;
		if (S->cum[mid] <= s) lo = mid; else hi = mid;
	}
	return lo;
}

static void count_piece(const seqs_t *S, piece_t *P, int64_t *rec_len)
{
	const uint8_t *B = S->B;
	int64_t s = P->start, i = rec_of(S, s);
	P->nb = P->na = 0; P->weird = 0;
	for (; s < P->end; ++i) {
		const rec_t *r = &S->rec[i];
		int64_t x = r->rs + (s - S->cum[i]), b = r->re < x + (P->end - s) ? r->re : x + (P->end - s), nb = 0, na = 0;
		if (x >= b) continue;
		s += b - x;
		while (x < b) {
			const uint8_t *nl = (const uint8_t*)memchr(B + x, '\n', b - x);
			int64_t e = nl ? nl - B : b, f = e;
			if (e > x && B[e - 1] == '\r' && cr_dropped(B, S->N, e - 1, r->rs)) f = e - 1;
			nb += f - x;
			for (; x < f; ++x)
				if (nst_nt4_table[B[x]] >= 4) {
					++na;
					if (B[x] == 0 || B[x] >= 128) P->weird = 1;
				}
			x = e + 1;
		}
		P->nb += nb; P->na += na;
		__atomic_fetch_add(&rec_len[i], nb, __ATOMIC_RELAXED);
	}
}

static void push_hole(piece_t *P, int64_t offset, char amb, int64_t rid)
{
	if (P->nh == P->mh) {
		P->mh = P->mh ? P->mh * 2 : 64;
		P->h = (hole_t*)realloc(P->h, P->mh * sizeof(hole_t));
	}
	P->h[P->nh].offset = offset; P->h[P->nh].len = 1; P->h[P->nh].amb = amb; P->h[P->nh].rid = (int32_t)rid;
	++P->nh;
}

/* add1() on the bases of the piece: g0 the offset of its first base, k0 the number of lrand48() draws before it */
static void pack_piece(const seqs_t *S, piece_t *P, int64_t g0, uint64_t k0)
{
	const uint8_t *B = S->B;
	uint8_t *pac = S->pac;
	const int64_t first = g0 >> 2, last = (g0 + P->nb - 1) >> 2;
	uint64_t X = lcg_skip(0xB330EULL, k0);
	int64_t s = P->start, i = rec_of(S, s), g = g0, bi = first;
	uint8_t acc = 0;
	if (P->nb == 0) return;
	for (; s < P->end; ++i) {
		const rec_t *r = &S->rec[i];
		int64_t x = r->rs + (s - S->cum[i]), b = r->re < x + (P->end - s) ? r->re : x + (P->end - s);
		int lasts;
		if (x >= b) continue;
		s += b - x;
		lasts = x == r->rs ? 0 : prev_base(B, S->N, x, r->rs);
		while (x < b) {
			const uint8_t *nl = (const uint8_t*)memchr(B + x, '\n', b - x);
			int64_t e = nl ? nl - B : b, f = e;
			if (e > x && B[e - 1] == '\r' && cr_dropped(B, S->N, e - 1, r->rs)) f = e - 1;
			for (; x < f; ++x, ++g) {
				int ch = B[x], c = nst_nt4_table[ch];
				if (c >= 4) {
					if (lasts == ch) {   /* the same hole */
						if (P->nh > 0 && P->h[P->nh - 1].rid == i) ++P->h[P->nh - 1].len;
						else ++P->cont;    /* the hole open where the piece starts */
					} else push_hole(P, g, (char)ch, i);
					X = (LCG_A * X + LCG_C) & LCG_MASK;
					c = (int)(X >> 17) & 3;
				}
				lasts = ch;
				if (g >> 2 != bi) {
					if (bi == first || bi == last) __atomic_fetch_or(&pac[bi], acc, __ATOMIC_RELAXED); else pac[bi] = acc;
					bi = g >> 2; acc = 0;
				}
				acc |= (uint8_t)(c << ((~g & 3) << 1));
			}
			x = e + 1;
		}
	}
	__atomic_fetch_or(&pac[bi], acc, __ATOMIC_RELAXED);
}

/* ---- the upstream code, for what the above does not cover ---- */

typedef struct { int fd; const uint8_t *b; int64_t n; } feed_t;

static void *feed(void *arg)
{
	feed_t *f = (feed_t*)arg;
	for (int64_t o = 0; o < f->n; ) {
		ssize_t r = write(f->fd, f->b + o, f->n - o < (1 << 26) ? f->n - o : (1 << 26));
		if (r <= 0) break;
		o += r;
	}
	close(f->fd);
	return NULL;
}

static int64_t upstream(const char *fn, const char *prefix, const uint8_t *B, int64_t N, int rereadable)
{
	int pfd[2];
	pthread_t tid;
	feed_t f;
	gzFile g;
	int64_t l;
	char tmp[1 << 16];
	if (rereadable) return galn_fasta_to_pac_serial(fn, prefix);
	/* a pipe or stdin: give upstream the bytes already read, through a pipe */
	if (pipe(pfd) != 0) fatal(fn, strerror(errno));
	f.fd = pfd[1]; f.b = B; f.n = N;
	pthread_create(&tid, NULL, feed, &f);
	g = gzdopen(pfd[0], "r");
	l = bns_fasta2bntseq(g, prefix, 1);
	while (read(pfd[0], tmp, sizeof(tmp)) > 0);   /* let the feeder finish */
	gzclose(g);
	pthread_join(tid, NULL);
	return l;
}

/* ---- FASTA -> .pac/.ann/.amb ---- */

static void write_pac(const char *prefix, const uint8_t *pac, int64_t l_pac)   /* as bns_fasta2bntseq() */
{
	char *fn = (char*)malloc(strlen(prefix) + 5);
	FILE *fp;
	uint8_t ct;
	sprintf(fn, "%s.pac", prefix);
	fp = xopen(fn, "wb");
	err_fwrite(pac, 1, (l_pac >> 2) + ((l_pac & 3) == 0 ? 0 : 1), fp);
	if (l_pac % 4 == 0) {
		ct = 0;
		err_fwrite(&ct, 1, 1, fp);
	}
	ct = l_pac % 4;
	err_fwrite(&ct, 1, 1, fp);
	err_fflush(fp);
	err_fclose(fp);
	free(fn);
}

int64_t galn_fasta_to_pac(const char *fn, const char *prefix, uint8_t **pac_out)
{
	const int T = omp_get_max_threads();
	double t0 = galn_realtime(), t1, t2;
	int64_t N, q, nrec, nmk = 0, R, np = 0, l_pac = 0, n_holes = 0, i, k;
	int rereadable, fastq = 0, weird = 0;
	const char *why = NULL;
	uint8_t *B;
	int64_t **mk, *nm, *rec_len, *goff;
	uint64_t *koff;
	rec_t *rec;
	piece_t *P = NULL;
	seqs_t S = {0};
	bntseq_t *bns;

	if (pac_out) *pac_out = NULL;
	if (getenv("GALN_FASTA_SERIAL")) return galn_fasta_to_pac_serial(fn, prefix);   /* testing */
	B = read_all(fn, &N, &rereadable);
	t1 = galn_realtime();

	/* 1. records: the first '>' or '@' anywhere (kseq skips what is before it), then line starts with '>' or '@' */
	for (q = 0; q < N && B[q] != '>' && B[q] != '@'; ++q);
	mk = (int64_t**)calloc(T, sizeof(int64_t*));
	nm = (int64_t*)calloc(T, sizeof(int64_t));
	if (q < N - 1) {
#pragma omp parallel num_threads(T) reduction(|:fastq)
		{
			const int t = omp_get_thread_num(), nt = omp_get_num_threads();
			const uint8_t *p = B + q + (N - 1 - q) * t / nt, *e = B + q + (N - 1 - q) * (t + 1) / nt;   /* '\n' before N-1 */
			int64_t m = 0;
			while (p < e && (p = (const uint8_t*)memchr(p, '\n', e - p))) {
				if (p[1] == '>' || p[1] == '@') {
					if (nm[t] == m) {
						m = m ? m * 2 : 1024;
						mk[t] = (int64_t*)realloc(mk[t], m * sizeof(int64_t));
					}
					mk[t][nm[t]++] = p + 1 - B;
				} else if (p[1] == '+') fastq = 1;
				++p;
			}
		}
	}
	for (int t = 0; t < T; ++t) nmk += nm[t];
	nrec = q < N ? nmk + 1 : 0;
	rec = (rec_t*)calloc(nrec + 1, sizeof(rec_t));
	{
		int64_t j = 1;
		if (nrec > 0) rec[0].rs = q + 1;   /* header starts, for now */
		for (int t = 0; t < T; ++t) {
			for (int64_t u = 0; u < nm[t]; ++u, ++j) {
				rec[j].rs = mk[t][u] + 1;
				rec[j - 1].re = mk[t][u];
			}
			free(mk[t]);
		}
		if (nrec > 0) rec[nrec - 1].re = N;
	}
	free(mk); free(nm);

	/* 2. headers */
	{
		int last_empty = 0;
#pragma omp parallel for schedule(dynamic, 1024) num_threads(T) reduction(|:last_empty)
		for (i = 0; i < nrec; ++i) {
			int64_t p = parse_header(B, N, rec[i].rs, &rec[i]);
			if (p < 0) last_empty = 1;   /* only the last record: the input ends with '>' */
			else rec[i].rs = p;
		}
		if (last_empty) --nrec;
	}

	if (fastq) why = "FASTQ records";
	if (!why) {
		/* 3. the sequence bytes, in pieces of equal size */
		int64_t *cum = (int64_t*)malloc((nrec + 1) * sizeof(int64_t)), size;
		cum[0] = 0;
		for (i = 0; i < nrec; ++i) cum[i + 1] = cum[i] + (rec[i].re - rec[i].rs);
		R = cum[nrec];
		size = (R + 8 * T - 1) / (8 * T);
		if (size < (1 << 20)) size = 1 << 20;
		if (getenv("GALN_FASTA_PIECE") && atoll(getenv("GALN_FASTA_PIECE")) > 0) size = atoll(getenv("GALN_FASTA_PIECE"));   /* testing */
		np = R > 0 ? (R + size - 1) / size : 0;
		P = (piece_t*)calloc(np + 1, sizeof(piece_t));
		for (k = 0; k < np; ++k) {
			P[k].start = k * size;
			P[k].end = (k + 1) * size < R ? (k + 1) * size : R;
		}
		S.B = B; S.N = N; S.nrec = nrec; S.cum = cum; S.rec = rec; S.pac = NULL;
		rec_len = (int64_t*)calloc(nrec + 1, sizeof(int64_t));
#pragma omp parallel for schedule(dynamic, 1) num_threads(T)
		for (k = 0; k < np; ++k) count_piece(&S, &P[k], rec_len);
		goff = (int64_t*)malloc((np + 1) * sizeof(int64_t));
		koff = (uint64_t*)malloc((np + 1) * sizeof(uint64_t));
		goff[0] = 0; koff[0] = 0;
		for (k = 0; k < np; ++k) {
			goff[k + 1] = goff[k] + P[k].nb;
			koff[k + 1] = koff[k] + P[k].na;
			weird |= P[k].weird;
		}
		l_pac = goff[np];
		for (i = 0; i < nrec; ++i) {
			rec[i].len = rec_len[i];
			if (rec_len[i] > INT32_MAX) why = "a sequence of 2^31 bases or more";
		}
		if (weird) why = "NUL or non-ASCII bytes in a sequence";
		if (!why) {
			S.pac = (uint8_t*)calloc((l_pac >> 2) + 2, 1);
			if (!S.pac) fatal(fn, "out of memory");
#pragma omp parallel for schedule(dynamic, 1) num_threads(T)
			for (k = 0; k < np; ++k) pack_piece(&S, &P[k], goff[k], koff[k]);
		}
		free(cum); free(rec_len); free(goff); free(koff);
	}
	t2 = galn_realtime();

	if (why) {
		GALN_VLOG("[galn-index] %s: %s; reading it with the serial code\n", fn, why);
		for (i = 0; i < nrec; ++i) { free(rec[i].name); free(rec[i].anno); }
		free(rec);
		for (k = 0; k < np; ++k) free(P[k].h);
		free(P); free(S.pac);
		l_pac = upstream(fn, prefix, B, N, rereadable);
		free(B);
		return l_pac;
	}
	free(B);

	/* holes in order: a piece's first bases may lengthen the last hole of the pieces before it */
	bns = (bntseq_t*)calloc(1, sizeof(bntseq_t));
	for (k = 0; k < np; ++k) n_holes += P[k].nh;
	bns->ambs = (bntamb1_t*)calloc(n_holes > 0 ? n_holes : 1, sizeof(bntamb1_t));
	n_holes = 0;
	for (k = 0; k < np; ++k) {
		if (P[k].cont) bns->ambs[n_holes - 1].len += (int32_t)P[k].cont;
		for (int64_t j = 0; j < P[k].nh; ++j, ++n_holes) {
			bns->ambs[n_holes].offset = P[k].h[j].offset;
			bns->ambs[n_holes].len = P[k].h[j].len;
			bns->ambs[n_holes].amb = P[k].h[j].amb;
			++rec[P[k].h[j].rid].n_ambs;
		}
		free(P[k].h);
	}
	free(P);

	bns->l_pac = l_pac;
	bns->n_seqs = (int32_t)nrec;
	bns->seed = 11;
	bns->n_holes = (int32_t)n_holes;
	bns->anns = (bntann1_t*)calloc(nrec > 0 ? nrec : 1, sizeof(bntann1_t));
	for (i = 0; i < nrec; ++i) {
		bntann1_t *p = &bns->anns[i];
		p->name = rec[i].name; p->anno = rec[i].anno;
		p->gi = 0; p->len = (int32_t)rec[i].len;
		p->offset = i == 0 ? 0 : (p - 1)->offset + (p - 1)->len;
		p->n_ambs = rec[i].n_ambs;
	}
	free(rec);
	write_pac(prefix, S.pac, l_pac);
	bns_dump(bns, prefix);
	bns_destroy(bns);
	if (pac_out) *pac_out = S.pac; else free(S.pac);
	GALN_VLOG("[galn-index] FASTA: read %.2f GB in %.2f s (%.2f GB/s), %lld sequences parsed in %.2f s, files written in %.2f s\n",
			N / 1e9, t1 - t0, N / 1e9 / (t1 - t0 > 1e-9 ? t1 - t0 : 1e-9), (long long)nrec, t2 - t1, galn_realtime() - t2);
	return l_pac;
}
