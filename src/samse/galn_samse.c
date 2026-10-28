/* galn_samse.c — `galn samse`: `bwa samse` (BWA 0.7.19) with the index loaded once and the
 * per-read work spread over all CPU cores.
 *
 *   galn samse [options] <prefix> <in.sai> <in.fq>
 *
 * bwa samse reloads the .bwt, .sa and .pac files for every batch of 262,144 reads and does all
 * work on one thread. Here the index is loaded once, and each batch is processed in three steps:
 *   1. reader thread: read the batch (bwa_read_seq) and its .sai records, and pick each read's
 *      hit among equally good ones (bwa_aln2seq_core, drand48 seeded from the index as in bwa).
 *      This runs in input order on one thread, so every read gets the same hit as with bwa samse.
 *      The reader prepares the next batch while the current one is processed.
 *   2. all cores, one read at a time: genome position (bwa_cal_pac_pos_core, bwa_sa2pos), CIGAR
 *      of gapped hits (bwa_refine_gapped_core), MD/NM (bwa_cal_md1), trimmed reads
 *      (bwa_correct_trimmed) — BWA's own functions from bwase.c, in bwa's order — and the SAM line
 *      (sam_line, a copy of bwa_print_sam1 that writes to a string).
 *   3. output in input order: BAM (default; records encoded on all cores, BGZF blocks compressed on all
 *      cores, see bam_out.c) or SAM (--sam). Each finished batch goes back to the reader thread, which
 *      frees it: the reads were allocated there, and freeing them on another thread is slow.
 * A gzip-compressed FASTQ is decompressed on one more thread, which feeds the reader through a pipe
 * (BWA's reader then sees plain FASTQ), so decompression and parsing overlap.
 * The alignments are the same as bwa samse's; only the @PG header line differs.
 *
 * -k INT (like bowtie2 -k): up to INT records per read, the alternative hits as secondary alignments of their own
 * instead of the XA tag. The reader thread lists the first INT hits of the .sai record (best first, as bwa_aln2seq_core
 * lists XA hits; it draws no random numbers, so the main hit is bwa's); their positions and CIGARs are made as for XA,
 * and sam_secondary() writes them after the main record. --no-unal leaves out reads without an aligned record.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <pthread.h>
#include <omp.h>
#include "sam_common.h"
#include "bwase.h"
#include "utils.h"
#include "version.h"
#include "bam_out.h"

#define SAMSE_BATCH 0x40000   /* reads per batch, as bwa samse */
#define SAMSE_QUEUE 2         /* batches the reader may prepare ahead */

/* Everything bwa_sai2sam_se_core does for one read after bwa_aln2seq_core, in bwa's order:
 * bwa_cal_pac_pos, bwa_refine_gapped (CIGAR, MD, trimmed reads), bwa_print_sam1. With -k (k > 0) the hits in
 * s->multi become records of their own (at most k-1) instead of the XA tag. Returns the number of those. */
static int process_read(const bntseq_t *bns, const bwt_t *bwt, ubyte_t *pac, const gap_opt_t *opt, int k, int no_unal,
                        bwa_seq_t *s, kstring_t *md, kstring_t *sec, kstring_t *out)
{
	int j, n_multi, strand, len = s->len, n_sec = 0;
	// bwa_cal_pac_pos
	bwa_cal_pac_pos_core(bns, bwt, s, opt->max_diff, opt->fnr);
	for (j = n_multi = 0; j < s->n_multi; ++j) {
		bwt_multi1_t *q = s->multi + j;
		q->pos = bwa_sa2pos(bns, bwt, q->pos, s->len + q->ref_shift, &strand);
		q->strand = strand;
		if (q->pos != s->pos && q->pos != (bwtint_t)-1)
			s->multi[n_multi++] = *q;
	}
	s->n_multi = n_multi;
	refine_read(bns, pac, s, md); // bwa_refine_gapped
	if (k > 0) {
		sec->l = 0;
		if (s->type != BWA_TYPE_NO_MATCH) n_sec = sam_secondary(sec, bns, pac, s, len, k - 1, opt->mode, md);
		for (j = 0; j < s->n_multi; ++j) free(s->multi[j].cigar);
		s->n_multi = 0; // no XA tag
	}
	// bwa_print_sam1
	out->l = 0;
	sam_line(out, bns, s, 0, opt->mode, opt->max_top2);
	if (no_unal && n_sec == 0 && (atoi(strchr(out->s, '\t') + 1) & SAM_FSU)) out->l = 0;
	else if (n_sec) kputsn(sec->s, sec->l, out);
	return n_sec;
}

/* -k: the first k hits of the .sai record, best first, as bwa_aln2seq_core lists hits for XA (no random numbers) */
static void list_hits(int n_aln, const bwt_aln1_t *aln, bwa_seq_t *s, int k)
{
	int i, z = 0;
	bwtint_t n = 0;
	for (i = 0; i < n_aln; ++i) n += aln[i].l - aln[i].k + 1;
	if (n <= 1 || k <= 1) return;   // no hit, or only the main one
	if (n > (bwtint_t)k) n = k;
	s->multi = (bwt_multi1_t*)calloc(n, sizeof(bwt_multi1_t));
	for (i = 0; i < n_aln && z < k; ++i) {
		const bwt_aln1_t *q = aln + i;
		bwtint_t l;
		for (l = q->k; l <= q->l && z < k; ++l) {
			s->multi[z].pos = l;
			s->multi[z].gap = q->n_gapo + q->n_gape;
			s->multi[z].ref_shift = (int)q->n_del - (int)q->n_ins;
			s->multi[z++].mm = q->n_mm;
		}
	}
	s->n_multi = z;
}

/* ---- reader thread: batches of reads with their chosen hits, in input order ---- */
typedef struct batch_s {
	int n;                      /* reads; 0 marks the end of the input */
	bwa_seq_t *seqs;
	struct batch_s *next;
} batch_t;

typedef struct {
	bwa_seqio_t *ks;
	FILE *fp_sa;
	const gap_opt_t *opt;
	int n_occ, k;
	pthread_mutex_t m;
	pthread_cond_t c;
	batch_t *head, *tail;
	int n_queued;
	batch_t *done;              /* finished batches, freed by the reader (which allocated them) */
	double t_read, t_sai, t_free, t_full;   /* reader time: parsing, .sai + hit choice, freeing, queue full */
} reader_t;

/* free the batches the main thread has finished (reader thread only) */
static void free_done(reader_t *R)
{
	batch_t *b, *next;
	double t = omp_get_wtime();
	pthread_mutex_lock(&R->m);
	b = R->done; R->done = 0;
	pthread_mutex_unlock(&R->m);
	for (; b; b = next) {
		next = b->next;
		bwa_free_read_seq(b->n, b->seqs);
		free(b);
	}
	R->t_free += omp_get_wtime() - t;
}

/* main thread: hand a finished batch back to the reader */
static void batch_done(reader_t *R, batch_t *b)
{
	pthread_mutex_lock(&R->m);
	b->next = R->done; R->done = b;
	pthread_cond_broadcast(&R->c);
	pthread_mutex_unlock(&R->m);
}

static void *reader_main(void *arg)
{
	reader_t *R = (reader_t*)arg;
	bwt_aln1_t *aln = 0;
	int m_aln = 0;
	for (;;) {
		int i, n_seqs = 0;
		double t;
		free_done(R);
		t = omp_get_wtime();
		bwa_seq_t *seqs = bwa_read_seq(R->ks, SAMSE_BATCH, &n_seqs, R->opt->mode, R->opt->trim_qual);
		batch_t *b = (batch_t*)calloc(1, sizeof(batch_t));
		R->t_read += omp_get_wtime() - t; t = omp_get_wtime();
		if (seqs) {
			for (i = 0; i < n_seqs; ++i) { // as in bwa_sai2sam_se_core: the only use of drand48
				bwa_seq_t *p = seqs + i;
				int n_aln;
				err_fread_noeof(&n_aln, 4, 1, R->fp_sa);
				if (n_aln > m_aln) {
					m_aln = n_aln;
					aln = (bwt_aln1_t*)realloc(aln, sizeof(bwt_aln1_t) * m_aln);
				}
				err_fread_noeof(aln, sizeof(bwt_aln1_t), n_aln, R->fp_sa);
				bwa_aln2seq_core(n_aln, aln, p, 1, R->k? 0 : R->n_occ);
				if (R->k) list_hits(n_aln, aln, p, R->k);
			}
			b->n = n_seqs;
			b->seqs = seqs;
		}
		R->t_sai += omp_get_wtime() - t; t = omp_get_wtime();
		pthread_mutex_lock(&R->m);
		while (R->n_queued >= SAMSE_QUEUE) {
			if (R->done) {      /* free finished batches while waiting */
				pthread_mutex_unlock(&R->m);
				free_done(R);
				pthread_mutex_lock(&R->m);
				continue;
			}
			pthread_cond_wait(&R->c, &R->m);
		}
		R->t_full += omp_get_wtime() - t;
		if (R->tail) R->tail->next = b; else R->head = b;
		R->tail = b;
		++R->n_queued;
		pthread_cond_broadcast(&R->c);
		pthread_mutex_unlock(&R->m);
		if (!seqs) break;
	}
	free(aln);
	return 0;
}

static batch_t *next_batch(reader_t *R)
{
	batch_t *b;
	pthread_mutex_lock(&R->m);
	while (!R->head) pthread_cond_wait(&R->c, &R->m);
	b = R->head;
	R->head = b->next;
	if (!R->head) R->tail = 0;
	--R->n_queued;
	pthread_cond_broadcast(&R->c);
	pthread_mutex_unlock(&R->m);
	return b;
}

static int samse_usage(void)
{
	fprintf(stderr, "\nUsage:   galn samse [options] <prefix> <in.sai> <in.fq> > out.bam\n\n"
	        "Options: -n INT    maximum hits to output in the XA tag; reads with more hits get none [3]\n"
	        "         -k INT    up to INT records per read, best first: the main alignment and INT-1 more as\n"
	        "                   secondary alignments (flag 256, MAPQ 0, NM and MD tags), instead of the XA tag\n"
	        "         --no-unal leave out reads that did not align\n"
	        "         -f FILE   file to write output to instead of stdout\n"
	        "         -r STR    read group header line such as '@RG\\tID:foo\\tSM:bar'\n"
	        "         -l INT    BAM compression level, 0-9 [6]\n"
	        "         --sam     write SAM instead of BAM\n"
	        "         -v        print timing details to stderr\n\n"
	        "Writes unsorted BAM with the same alignments as `bwa samse` (BWA 0.7.19); with --sam the output\n"
	        "is identical to bwa samse's apart from the @PG header line.\n"
	        "Uses all CPU cores of the job (or OMP_NUM_THREADS).\n\n");
	return 1;
}

/* argv: the arguments after `galn samse` */
int galn_samse(int argc, char *argv[])
{
	int i, n_occ = 3, k = 0, no_unal = 0, verbose = 0, n_pos = 0, sam_out = 0, level = 6, bad = 0;
	char *pos[3], *rg_line = 0, *prefix;
	const char *fn_out = 0;
	double t0 = omp_get_wtime(), t_idx, t_wait = 0, t_proc = 0, t_write = 0, t;
	long long n_reads = 0, n_sec = 0;
	kstring_t pg = {0, 0, 0};
	for (i = 0; i < argc; ++i) {
		const char *a = argv[i];
		if (!strcmp(a, "--sam")) { sam_out = 1; continue; }
		if (!strcmp(a, "--no-unal")) { no_unal = 1; continue; }
		if (a[0] == '-' && a[1] && !a[2]) {
			if (a[1] == 'v') { verbose = 1; continue; }
			if (i + 1 >= argc) return samse_usage();
			if (a[1] == 'n') n_occ = atoi(argv[++i]);
			else if (a[1] == 'k') { if ((k = atoi(argv[++i])) < 1) return samse_usage(); }
			else if (a[1] == 'f') fn_out = argv[++i];
			else if (a[1] == 'l') level = atoi(argv[++i]);
			else if (a[1] == 'r') { if ((rg_line = bwa_set_rg(argv[++i])) == 0) return 1; }
			else return samse_usage();
		} else if (n_pos < 3) pos[n_pos++] = argv[i];
		else return samse_usage();
	}
	if (n_pos != 3) return samse_usage();
	if ((prefix = bwa_idx_infer_prefix(pos[0])) == 0) {
		fprintf(stderr, "[galn samse] fail to locate the index %s\n", pos[0]);
		return 1;
	}
	if (fn_out) xreopen(fn_out, "w", stdout);
	else if (!sam_out && isatty(fileno(stdout))) {
		fprintf(stderr, "[galn samse] BAM output: use -f out.bam or redirect stdout (or --sam for SAM)\n");
		return 1;
	}
	ksprintf(&pg, "@PG\tID:galn\tPN:galn\tVN:%s\tCL:galn samse", GALN_VERSION);
	for (i = 0; i < argc; ++i) ksprintf(&pg, " %s", argv[i]);
	bwa_pg = pg.s;

	// index annotations, the random seed and the .sai header (as bwa_sai2sam_se_core)
	bwase_initialize();
	bntseq_t *bns = bns_restore(prefix);
	srand48(bns->seed);
	FILE *fp_sa = xopen(pos[1], "r");
	char magic[4];
	gap_opt_t opt;
	err_fread_noeof(magic, 1, 4, fp_sa);
	if (strncmp(magic, SAI_MAGIC, 4) != 0) {
		fprintf(stderr, "[galn samse] %s is not a .sai file of this BWA version\n", pos[1]);
		return 1;
	}
	err_fread_noeof(&opt, sizeof(gap_opt_t), 1, fp_sa);
	kstring_t hdr = {0, 0, 0};
	sam_header(&hdr, bns, rg_line);

	// the reader starts on the first batches while the index loads
	reader_t R;
	memset(&R, 0, sizeof(R));
	inflate_t I;
	R.ks = open_reads_threaded(opt.mode, pos[2], &I);
	R.fp_sa = fp_sa; R.opt = &opt; R.n_occ = n_occ; R.k = k;
	pthread_mutex_init(&R.m, 0);
	pthread_cond_init(&R.c, 0);
	pthread_t tid;
	pthread_create(&tid, 0, reader_main, &R);

	// the index, once: .bwt + .sa for positions, .pac for CIGAR/MD
	bwt_t *bwt = load_bwt_sa(prefix);
	ubyte_t *pac = load_pac(bns);
	t_idx = omp_get_wtime() - t0;

	bam_out_t *B = 0;
	if (sam_out) err_fwrite(hdr.s, 1, hdr.l, stdout);
	else B = bam_out_open(stdout, level, hdr.s, bns);
	kstring_t *out = (kstring_t*)calloc(SAMSE_BATCH, sizeof(kstring_t));
	kstring_t *rec = sam_out? 0 : (kstring_t*)calloc(SAMSE_BATCH, sizeof(kstring_t));
	for (;;) {
		t = omp_get_wtime();
		batch_t *b = next_batch(&R);
		int n = b->n;
		t_wait += omp_get_wtime() - t;
		if (n == 0) { free(b); break; }
		t = omp_get_wtime();
		#pragma omp parallel
		{
			kstring_t md = {0, 0, 0}, sec = {0, 0, 0};
			int r;
			#pragma omp for schedule(dynamic, 256) reduction(+:bad,n_sec)
			for (r = 0; r < n; ++r) {
				n_sec += process_read(bns, bwt, pac, &opt, k, no_unal, b->seqs + r, &md, &sec, &out[r]);
				if (rec) { // one record per line
					const char *p = out[r].s, *e = out[r].s + out[r].l;
					rec[r].l = 0;
					while (p < e) {
						const char *nl = (const char*)memchr(p, '\n', e - p);
						int l = nl? (int)(nl - p + 1) : (int)(e - p);
						if (bam_out_encode(B, p, l, &rec[r]) < 0) ++bad;
						p += l;
					}
				}
			}
			free(md.s); free(sec.s);
		}
		if (bad) {
			fprintf(stderr, "[galn samse] internal error: could not encode %d SAM lines as BAM\n", bad);
			return 1;
		}
		t_proc += omp_get_wtime() - t; t = omp_get_wtime();
		if (rec) bam_out_write(B, n, rec);
		else for (i = 0; i < n; ++i) err_fwrite(out[i].s, 1, out[i].l, stdout);
		t_write += omp_get_wtime() - t;
		batch_done(&R, b);
		n_reads += n;
	}
	pthread_join(tid, 0);
	free_done(&R);
	if (B) bam_out_close(B);
	bwa_seq_close(R.ks); // closes the pipe's read end, so the decompression thread always finishes
	inflate_join(&I);
	err_fflush(stdout);
	if (k) fprintf(stderr, "[galn samse] %lld reads in %.1f s; %lld secondary records\n", n_reads, omp_get_wtime() - t0, n_sec);
	else fprintf(stderr, "[galn samse] %lld reads in %.1f s\n", n_reads, omp_get_wtime() - t0);
	if (verbose)
		fprintf(stderr, "[galn samse] index %.2f s, waiting for reads %.2f s, per-read work %.2f s, writing %s %.2f s; "
		        "reader: parsing %.2f s, .sai %.2f s, freeing %.2f s, queue full %.2f s; %d threads\n", t_idx, t_wait, t_proc,
		        sam_out? "SAM" : "BAM", t_write, R.t_read, R.t_sai, R.t_free, R.t_full, omp_get_max_threads());

	for (i = 0; i < SAMSE_BATCH; ++i) { free(out[i].s); if (rec) free(rec[i].s); }
	free(out); free(rec); free(hdr.s);
	free(pac);
	bwt_destroy(bwt);
	bns_destroy(bns);
	err_fclose(fp_sa);
	free(prefix); free(rg_line); free(pg.s);
	return 0;
}
