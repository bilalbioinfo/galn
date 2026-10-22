/* galn_sampe.c — `galn sampe`: `bwa sampe` (BWA 0.7.19) with the index loaded once and the per-pair work
 * spread over all CPU cores, writing BAM (default) or SAM.
 *
 *   galn sampe [options] <prefix> <in1.sai> <in2.sai> <in1.fq> <in2.fq>
 *
 * Each batch of 262,144 pairs goes through the steps of bwa_sai2sam_pe_core / bwa_cal_pac_pos_pe in bwa's
 * order:
 *   reader thread  both ends and their .sai records; each read's hit among equally good ones
 *                  (bwa_aln2seq: the only use of drand48, called in bwa's order, so every read gets the same
 *                  hit as with bwa sampe). The reader prepares the next batch meanwhile and frees old ones.
 *   all cores      single-end positions and mapping qualities
 *   one thread     insert-size estimate from the whole batch (infer_isize)
 *   all cores      per pair: pairing (pairing), alternative hits, mate rescue by Smith-Waterman (the body
 *                  of bwa_paired_sw), gapped CIGAR/MD (refine_read) and both SAM lines (sam_line)
 *   main thread    BAM (or SAM) in input order
 * bwape.c is included unchanged, so infer_isize, pairing and bwa_sw_core are BWA's own functions (static
 * there). The alignments are the same as bwa sampe's; only the @PG header line differs.
 */
#include "bwape.c"

#include <unistd.h>
#include <pthread.h>
#include <omp.h>
#include "sam_common.h"
#include "version.h"
#include "bam_out.h"

#define SAMPE_BATCH 0x40000   /* pairs per batch, as bwa sampe */
#define SAMPE_QUEUE 2         /* batches the reader may prepare ahead */

/* the single-end part of bwa_cal_pac_pos_pe's first loop that follows bwa_aln2seq (no random numbers) */
static void se_pos(const bntseq_t *bns, const bwt_t *bwt, const gap_opt_t *gopt, bwa_seq_t *p)
{
	if (p->type == BWA_TYPE_UNIQUE || p->type == BWA_TYPE_REPEAT) {
		int strand;
		int max_diff = gopt->fnr > 0.0? bwa_cal_maxdiff(p->len, BWA_AVG_ERR, gopt->fnr) : gopt->max_diff;
		p->seQ = p->mapQ = bwa_approx_mapQ(p, max_diff);
		p->pos = bwa_sa2pos(bns, bwt, p->sa, p->len + p->ref_shift, &strand);
		p->strand = strand;
		if (p->pos == (bwtint_t)-1) p->type = BWA_TYPE_NO_MATCH;
	}
}

/* one pair of bwa_cal_pac_pos_pe's second loop (pairing and alternative hits); d is per thread */
static int pe_pair(const bntseq_t *bns, const bwt_t *bwt, bwa_seq_t *p[2], aln_buf_t *buf[2], int i, pe_data_t *d,
                   const pe_opt_t *opt, const gap_opt_t *gopt, const isize_info_t *ii)
{
	int j, cnt_chg = 0;
	for (j = 0; j < 2; ++j) kv_copy(bwt_aln1_t, d->aln[j], buf[j][i].aln);
	if ((p[0]->type == BWA_TYPE_UNIQUE || p[0]->type == BWA_TYPE_REPEAT)
		&& (p[1]->type == BWA_TYPE_UNIQUE || p[1]->type == BWA_TYPE_REPEAT))
	{ // only when both ends mapped
		pair64_t x;
		int k;
		long long n_occ[2];
		for (j = 0; j < 2; ++j) {
			n_occ[j] = 0;
			for (k = 0; k < d->aln[j].n; ++k)
				n_occ[j] += d->aln[j].a[k].l - d->aln[j].a[k].k + 1;
		}
		if (n_occ[0] > opt->max_occ || n_occ[1] > opt->max_occ) return 0; // bwa: `continue`, skipping the hits below
		d->arr.n = 0;
		for (j = 0; j < 2; ++j) {
			for (k = 0; k < d->aln[j].n; ++k) {
				bwt_aln1_t *r = d->aln[j].a + k;
				bwtint_t l;
				for (l = r->k; l <= r->l; ++l) { // bwa's "calculate on the fly" branch (its hash branch is disabled)
					int strand;
					x.x = bwa_sa2pos(bns, bwt, l, p[j]->len + p[j]->ref_shift, &strand);
					x.y = k<<2 | strand<<1 | j;
					kv_push(pair64_t, d->arr, x);
				}
			}
		}
		cnt_chg += pairing(p, d, opt, gopt->s_mm, ii);
	}
	if (opt->N_multi || opt->n_multi) {
		for (j = 0; j < 2; ++j) {
			if (p[j]->type != BWA_TYPE_NO_MATCH) {
				int k, n_multi;
				// set_main = 0: bwa_aln2seq_core draws no random numbers here (all hits fit in n_multi)
				if (!(p[j]->extra_flag&SAM_FPP) && p[1-j]->type != BWA_TYPE_NO_MATCH) {
					bwa_aln2seq_core(d->aln[j].n, d->aln[j].a, p[j], 0, p[j]->c1+p[j]->c2-1 > opt->N_multi? opt->n_multi : opt->N_multi);
				} else bwa_aln2seq_core(d->aln[j].n, d->aln[j].a, p[j], 0, opt->n_multi);
				for (k = 0, n_multi = 0; k < p[j]->n_multi; ++k) {
					int strand;
					bwt_multi1_t *q = p[j]->multi + k;
					q->pos = bwa_sa2pos(bns, bwt, q->pos, p[j]->len + q->ref_shift, &strand);
					q->strand = strand;
					if (q->pos != p[j]->pos && q->pos != (bwtint_t)-1)
						p[j]->multi[n_multi++] = *q;
				}
				p[j]->n_multi = n_multi;
			}
		}
	}
	return cnt_chg;
}

/* one pair of bwa_paired_sw's loop: Smith-Waterman rescue of a mate; n_tot/n_mapped count as in bwa */
static void mate_sw(const bntseq_t *bns, const ubyte_t *pacseq, bwa_seq_t *p[2], const pe_opt_t *popt,
                    const isize_info_t *ii, uint64_t n_tot[2], uint64_t n_mapped[2])
{
	if ((p[0]->mapQ >= SW_MIN_MAPQ || p[1]->mapQ >= SW_MIN_MAPQ) && (p[0]->extra_flag&SAM_FPP) == 0) { // unpaired and one read has high mapQ
		int k, n_cigar[2], is_singleton, mapQ = 0, mq_adjust[2];
		int64_t beg[2], end[2];
		bwa_cigar_t *cigar[2];
		uint32_t cnt[2];

		mq_adjust[0] = mq_adjust[1] = 255; // not effective
		is_singleton = (p[0]->type == BWA_TYPE_NO_MATCH || p[1]->type == BWA_TYPE_NO_MATCH)? 1 : 0;

		++n_tot[is_singleton];
		cigar[0] = cigar[1] = 0;
		n_cigar[0] = n_cigar[1] = 0;
		if (popt->type != BWA_PET_STD) return; // other types of pairing is not considered
		for (k = 0; k < 2; ++k) { // p[1-k] is the reference read and p[k] is the read considered to be modified
			ubyte_t *seq;
			if (p[1-k]->type == BWA_TYPE_NO_MATCH) continue; // if p[1-k] is unmapped, skip
			{
				if (p[1-k]->strand == 0) { // then the mate is on the reverse strand and has larger coordinate
					__set_rght_coor(beg[k], end[k], p[1-k], p[k]);
					seq = p[k]->rseq;
				} else { // then the mate is on forward stand and has smaller coordinate
					__set_left_coor(beg[k], end[k], p[1-k], p[k]);
					seq = p[k]->seq;
					seq_reverse(p[k]->len, seq, 0); // because ->seq is reversed; this will reversed back shortly
				}
			}
			// perform SW alignment
			cigar[k] = bwa_sw_core(bns->l_pac, pacseq, p[k]->len, seq, &beg[k], end[k] - beg[k], &n_cigar[k], &cnt[k]);
			if (cigar[k] && p[k]->type != BWA_TYPE_NO_MATCH) { // re-evaluate cigar[k]
				int s_old, clip = 0, s_new;
				if (__cigar_op(cigar[k][0]) == 3) clip += __cigar_len(cigar[k][0]);
				if (__cigar_op(cigar[k][n_cigar[k]-1]) == 3) clip += __cigar_len(cigar[k][n_cigar[k]-1]);
				s_old = (int)((p[k]->n_mm * 9 + p[k]->n_gapo * 13 + p[k]->n_gape * 2) / 3. * 8. + .499);
				s_new = (int)(((cnt[k]>>16) * 9 + (cnt[k]>>8&0xff) * 13 + (cnt[k]&0xff) * 2 + clip * 3) / 3. * 8. + .499);
				s_old += -4.343 * log(ii->ap_prior / bns->l_pac);
				s_new += (int)(-4.343 * log(.5 * erfc(M_SQRT1_2 * 1.5) + .499)); // assume the mapped isize is 1.5\sigma
				if (s_old < s_new) { // reject SW alignment
					mq_adjust[k] = s_new - s_old;
					free(cigar[k]); cigar[k] = 0; n_cigar[k] = 0;
				} else mq_adjust[k] = s_old - s_new;
			}
			// now revserse sequence back such that p[*]->seq looks untouched
			if (popt->type == BWA_PET_STD) {
				if (p[1-k]->strand == 1) seq_reverse(p[k]->len, seq, 0);
			} else {
				if (p[1-k]->strand == 0) seq_reverse(p[k]->len, seq, 0);
			}
		}
		k = -1; // no read to be changed
		if (cigar[0] && cigar[1]) {
			k = p[0]->mapQ < p[1]->mapQ? 0 : 1; // p[k] to be fixed
			mapQ = abs(p[1]->mapQ - p[0]->mapQ);
		} else if (cigar[0]) k = 0, mapQ = p[1]->mapQ;
		else if (cigar[1]) k = 1, mapQ = p[0]->mapQ;
		if (k >= 0 && p[k]->pos != beg[k]) {
			++n_mapped[is_singleton];
			{ // recalculate mapping quality
				int tmp = (int)p[1-k]->mapQ - p[k]->mapQ/2 - 8;
				if (tmp <= 0) tmp = 1;
				if (mapQ > tmp) mapQ = tmp;
				p[k]->mapQ = p[1-k]->mapQ = mapQ;
				p[k]->seQ = p[1-k]->seQ = p[1-k]->seQ < mapQ? p[1-k]->seQ : mapQ;
				if (p[k]->mapQ > mq_adjust[k]) p[k]->mapQ = mq_adjust[k];
				if (p[k]->seQ > mq_adjust[k]) p[k]->seQ = mq_adjust[k];
			}
			// update CIGAR
			free(p[k]->cigar); p[k]->cigar = cigar[k]; cigar[k] = 0;
			p[k]->n_cigar = n_cigar[k];
			// update the rest of information
			__set_fixed(p[1-k], p[k], beg[k], cnt[k]);
		}
		free(cigar[0]); free(cigar[1]);
	}
}

/* ---- reader thread: batches of pairs with their .sai records and chosen hits, in input order ---- */
typedef struct pbatch_s {
	int n;                      /* pairs; 0 marks the end of the input */
	bwa_seq_t *seqs[2];
	aln_buf_t *buf[2];          /* each read's .sai records (bwa keeps them for pairing) */
	struct pbatch_s *next;
} pbatch_t;

typedef struct {
	bwa_seqio_t *ks[2];
	FILE *fp_sa[2];
	const gap_opt_t *opt0, *opt;   /* the .sai headers of end 1 and end 2, used as bwa does */
	pthread_mutex_t m;
	pthread_cond_t c;
	pbatch_t *head, *tail, *done;
	int n_queued;
} preader_t;

static void free_pbatch(pbatch_t *b)
{
	int i, j;
	for (j = 0; j < 2; ++j) {
		if (b->buf[j]) for (i = 0; i < b->n; ++i) kv_destroy(b->buf[j][i].aln);
		free(b->buf[j]);
		if (b->seqs[j]) bwa_free_read_seq(b->n, b->seqs[j]);
	}
	free(b);
}

static void pfree_done(preader_t *R)
{
	pbatch_t *b, *next;
	pthread_mutex_lock(&R->m);
	b = R->done; R->done = 0;
	pthread_mutex_unlock(&R->m);
	for (; b; b = next) { next = b->next; free_pbatch(b); }
}

static void *preader_main(void *arg)
{
	preader_t *R = (preader_t*)arg;
	kvec_t(bwt_aln1_t) aln = {0, 0, 0};
	for (;;) {
		int i, j, n0 = 0, n1 = 0;
		pbatch_t *b = (pbatch_t*)calloc(1, sizeof(pbatch_t));
		pfree_done(R);
		b->seqs[0] = bwa_read_seq(R->ks[0], SAMPE_BATCH, &n0, R->opt0->mode, R->opt0->trim_qual);
		if (b->seqs[0]) {
			b->seqs[1] = bwa_read_seq(R->ks[1], SAMPE_BATCH, &n1, R->opt->mode, R->opt->trim_qual);
			if (n1 != n0) {
				fprintf(stderr, "[galn sampe] the two read files have different numbers of reads\n");
				exit(1);
			}
			b->n = n0;
			b->buf[0] = (aln_buf_t*)calloc(n0, sizeof(aln_buf_t));
			b->buf[1] = (aln_buf_t*)calloc(n0, sizeof(aln_buf_t));
			for (i = 0; i != n0; ++i) { // as the first loop of bwa_cal_pac_pos_pe, up to bwa_aln2seq
				for (j = 0; j < 2; ++j) {
					int n_aln;
					bwa_seq_t *p = b->seqs[j] + i;
					p->n_multi = 0;
					p->extra_flag |= SAM_FPD | (j == 0? SAM_FR1 : SAM_FR2);
					err_fread_noeof(&n_aln, 4, 1, R->fp_sa[j]);
					if (n_aln > kv_max(aln)) kv_resize(bwt_aln1_t, aln, n_aln);
					aln.n = n_aln;
					err_fread_noeof(aln.a, sizeof(bwt_aln1_t), n_aln, R->fp_sa[j]);
					kv_copy(bwt_aln1_t, b->buf[j][i].aln, aln); // backup
					bwa_aln2seq(n_aln, aln.a, p);
				}
			}
		}
		pthread_mutex_lock(&R->m);
		while (R->n_queued >= SAMPE_QUEUE) {
			if (R->done) {
				pthread_mutex_unlock(&R->m);
				pfree_done(R);
				pthread_mutex_lock(&R->m);
				continue;
			}
			pthread_cond_wait(&R->c, &R->m);
		}
		if (R->tail) R->tail->next = b; else R->head = b;
		R->tail = b;
		++R->n_queued;
		pthread_cond_broadcast(&R->c);
		pthread_mutex_unlock(&R->m);
		if (b->n == 0) break;
	}
	kv_destroy(aln);
	return 0;
}

static pbatch_t *pnext_batch(preader_t *R)
{
	pbatch_t *b;
	pthread_mutex_lock(&R->m);
	while (!R->head) pthread_cond_wait(&R->c, &R->m);
	b = R->head;
	R->head = b->next;
	if (!R->head) R->tail = 0;
	--R->n_queued;
	pthread_cond_broadcast(&R->c);
	pthread_mutex_unlock(&R->m);
	b->next = 0;
	return b;
}

static void pbatch_done(preader_t *R, pbatch_t *b)
{
	pthread_mutex_lock(&R->m);
	b->next = R->done; R->done = b;
	pthread_cond_broadcast(&R->c);
	pthread_mutex_unlock(&R->m);
}

static int sampe_usage(const pe_opt_t *popt)
{
	fprintf(stderr, "\nUsage:   galn sampe [options] <prefix> <in1.sai> <in2.sai> <in1.fq> <in2.fq> > out.bam\n\n");
	fprintf(stderr, "Options: -a INT   maximum insert size [%d]\n", popt->max_isize);
	fprintf(stderr, "         -o INT   maximum occurrences for one end [%d]\n", popt->max_occ);
	fprintf(stderr, "         -n INT   maximum hits to output for paired reads [%d]\n", popt->n_multi);
	fprintf(stderr, "         -N INT   maximum hits to output for discordant pairs [%d]\n", popt->N_multi);
	fprintf(stderr, "         -c FLOAT prior of chimeric rate (lower bound) [%.1le]\n", popt->ap_prior);
	fprintf(stderr, "         -f FILE  file to write output to instead of stdout\n");
	fprintf(stderr, "         -r STR   read group header line such as '@RG\\tID:foo\\tSM:bar' [null]\n");
	fprintf(stderr, "         -s       disable Smith-Waterman for the unmapped mate\n");
	fprintf(stderr, "         -A       disable insert size estimate (force -s)\n");
	fprintf(stderr, "         -P       accepted for compatibility (the index is always loaded once)\n");
	fprintf(stderr, "         -l INT   BAM compression level, 0-9 [6]\n");
	fprintf(stderr, "         --sam    write SAM instead of BAM\n");
	fprintf(stderr, "         -v       print timing to stderr\n\n");
	fprintf(stderr, "Writes unsorted BAM with the same alignments as `bwa sampe` (BWA 0.7.19); with --sam the output\n"
	        "is identical to bwa sampe's apart from the @PG header line.\n"
	        "Uses all CPU cores of the job (or OMP_NUM_THREADS).\n\n");
	return 1;
}

/* argv: the arguments after `galn sampe` */
int galn_sampe(int argc, char *argv[])
{
	int i, j, n_pos = 0, sam_out = 0, level = 6, verbose = 0;
	char *pos[5], *rg_line = 0, *prefix;
	const char *fn_out = 0;
	double t0 = omp_get_wtime(), t_idx;
	long long n_pairs = 0, cnt_chg_tot = 0;
	uint64_t n_tot[2] = {0, 0}, n_mapped[2] = {0, 0};
	pe_opt_t *popt = bwa_init_pe_opt();
	kstring_t pg = {0, 0, 0};
	for (i = 0; i < argc; ++i) {
		const char *a = argv[i];
		if (!strcmp(a, "--sam")) { sam_out = 1; continue; }
		if (a[0] == '-' && a[1] && !a[2]) {
			if (a[1] == 's') { popt->is_sw = 0; continue; }
			if (a[1] == 'P') { popt->is_preload = 1; continue; }
			if (a[1] == 'A') { popt->force_isize = 1; continue; }
			if (a[1] == 'v') { verbose = 1; continue; }
			if (i + 1 >= argc) return sampe_usage(popt);
			if (a[1] == 'a') popt->max_isize = atoi(argv[++i]);
			else if (a[1] == 'o') popt->max_occ = atoi(argv[++i]);
			else if (a[1] == 'n') popt->n_multi = atoi(argv[++i]);
			else if (a[1] == 'N') popt->N_multi = atoi(argv[++i]);
			else if (a[1] == 'c') popt->ap_prior = atof(argv[++i]);
			else if (a[1] == 'f') fn_out = argv[++i];
			else if (a[1] == 'l') level = atoi(argv[++i]);
			else if (a[1] == 'r') { if ((rg_line = bwa_set_rg(argv[++i])) == 0) return 1; }
			else return sampe_usage(popt);
		} else if (n_pos < 5) pos[n_pos++] = argv[i];
		else return sampe_usage(popt);
	}
	if (n_pos != 5) return sampe_usage(popt);
	if ((prefix = bwa_idx_infer_prefix(pos[0])) == 0) {
		fprintf(stderr, "[galn sampe] fail to locate the index %s\n", pos[0]);
		return 1;
	}
	if (fn_out) xreopen(fn_out, "w", stdout);
	else if (!sam_out && isatty(fileno(stdout))) {
		fprintf(stderr, "[galn sampe] BAM output: use -f out.bam or redirect stdout (or --sam for SAM)\n");
		return 1;
	}
	ksprintf(&pg, "@PG\tID:galn\tPN:galn\tVN:%s\tCL:galn sampe", GALN_VERSION);
	for (i = 0; i < argc; ++i) ksprintf(&pg, " %s", argv[i]);
	bwa_pg = pg.s;

	// as bwa_sai2sam_pe_core: annotations, random seed, .sai headers (end 2's header is the one used later)
	bwase_initialize();
	bntseq_t *bns = bns_restore(prefix);
	srand48(bns->seed);
	FILE *fp_sa[2];
	char magic[2][4];
	gap_opt_t opt, opt0;
	fp_sa[0] = xopen(pos[1], "r");
	fp_sa[1] = xopen(pos[2], "r");
	err_fread_noeof(magic[0], 1, 4, fp_sa[0]);
	err_fread_noeof(magic[1], 1, 4, fp_sa[1]);
	if (strncmp(magic[0], SAI_MAGIC, 4) != 0 || strncmp(magic[1], SAI_MAGIC, 4) != 0) {
		fprintf(stderr, "[galn sampe] %s or %s is not a .sai file of this BWA version\n", pos[1], pos[2]);
		return 1;
	}
	err_fread_noeof(&opt, sizeof(gap_opt_t), 1, fp_sa[0]);
	opt0 = opt;
	err_fread_noeof(&opt, sizeof(gap_opt_t), 1, fp_sa[1]); // overwritten, as in bwa
	kstring_t hdr = {0, 0, 0};
	sam_header(&hdr, bns, rg_line);

	// the reader starts on the first batches while the index loads
	preader_t R;
	inflate_t I[2];
	memset(&R, 0, sizeof(R));
	R.ks[0] = open_reads_threaded(opt0.mode, pos[3], &I[0]);
	R.ks[1] = open_reads_threaded(opt.mode, pos[4], &I[1]);
	R.fp_sa[0] = fp_sa[0]; R.fp_sa[1] = fp_sa[1];
	R.opt0 = &opt0; R.opt = &opt;
	pthread_mutex_init(&R.m, 0);
	pthread_cond_init(&R.c, 0);
	pthread_t tid;
	pthread_create(&tid, 0, preader_main, &R);

	bwt_t *bwt = load_bwt_sa(prefix);
	ubyte_t *pac = load_pac(bns);
	t_idx = omp_get_wtime() - t0;

	bam_out_t *B = 0;
	if (sam_out) err_fwrite(hdr.s, 1, hdr.l, stdout);
	else B = bam_out_open(stdout, level, hdr.s, bns);
	kstring_t *out = (kstring_t*)calloc(2 * SAMPE_BATCH, sizeof(kstring_t));
	kstring_t *rec = sam_out? 0 : (kstring_t*)calloc(2 * SAMPE_BATCH, sizeof(kstring_t));
	isize_info_t last_ii;
	last_ii.avg = -1.0;
	for (;;) {
		pbatch_t *b = pnext_batch(&R);
		int n = b->n, bad = 0, bad_name = -1, cnt_chg = 0;
		uint64_t nt0 = 0, nt1 = 0, nm0 = 0, nm1 = 0;
		isize_info_t ii;
		if (n == 0) { free(b); break; }
		// single-end positions (rest of bwa_cal_pac_pos_pe's first loop)
		#pragma omp parallel for schedule(dynamic, 256)
		for (i = 0; i < n; ++i) {
			se_pos(bns, bwt, &opt, b->seqs[0] + i);
			se_pos(bns, bwt, &opt, b->seqs[1] + i);
		}
		// insert size from the batch
		infer_isize(n, b->seqs, &ii, popt->ap_prior, bwt->seq_len/2);
		if (ii.avg < 0.0 && last_ii.avg > 0.0) ii = last_ii;
		if (popt->force_isize) {
			fprintf(stderr, "[galn sampe] discard insert size estimate as user's request.\n");
			ii.low = ii.high = 0; ii.avg = ii.std = -1.0;
		}
		// per pair: pairing, alternative hits, mate rescue, CIGAR/MD, SAM lines (and BAM records)
		#pragma omp parallel reduction(+:cnt_chg, nt0, nt1, nm0, nm1, bad)
		{
			pe_data_t *d = (pe_data_t*)calloc(1, sizeof(pe_data_t));
			kstring_t md = {0, 0, 0};
			uint64_t tot[2] = {0, 0}, mapped[2] = {0, 0};
			int r;
			#pragma omp for schedule(dynamic, 64)
			for (r = 0; r < n; ++r) {
				bwa_seq_t *p[2];
				p[0] = b->seqs[0] + r; p[1] = b->seqs[1] + r;
				cnt_chg += pe_pair(bns, bwt, p, b->buf, r, d, popt, &opt, &ii);
				if (popt->is_sw && ii.avg >= 0.0) mate_sw(bns, pac, p, popt, &ii, tot, mapped);
				refine_read(bns, pac, p[0], &md);
				refine_read(bns, pac, p[1], &md);
				if (p[0]->bc[0] || p[1]->bc[0]) {
					strcat(p[0]->bc, p[1]->bc);
					strcpy(p[1]->bc, p[0]->bc);
				}
				out[2*r].l = out[2*r+1].l = 0;
				sam_line(&out[2*r], bns, p[0], p[1], opt.mode, opt.max_top2);
				sam_line(&out[2*r+1], bns, p[1], p[0], opt.mode, opt.max_top2);
				if (strcmp(p[0]->name, p[1]->name) != 0) {
					#pragma omp critical
					if (bad_name < 0 || r < bad_name) bad_name = r;
				}
				if (rec) {
					rec[2*r].l = rec[2*r+1].l = 0;
					if (bam_out_encode(B, out[2*r].s, (int)out[2*r].l, &rec[2*r]) < 0) ++bad;
					if (bam_out_encode(B, out[2*r+1].s, (int)out[2*r+1].l, &rec[2*r+1]) < 0) ++bad;
				}
			}
			nt0 += tot[0]; nt1 += tot[1]; nm0 += mapped[0]; nm1 += mapped[1];
			kv_destroy(d->arr);
			kv_destroy(d->pos[0]); kv_destroy(d->pos[1]);
			kv_destroy(d->aln[0]); kv_destroy(d->aln[1]);
			free(d); free(md.s);
		}
		if (bad_name >= 0)
			err_fatal(__func__, "paired reads have different names: \"%s\", \"%s\"\n",
			          b->seqs[0][bad_name].name, b->seqs[1][bad_name].name);
		if (bad) {
			fprintf(stderr, "[galn sampe] internal error: could not encode %d SAM lines as BAM\n", bad);
			return 1;
		}
		if (rec) bam_out_write(B, 2 * n, rec);
		else for (i = 0; i < 2 * n; ++i) err_fwrite(out[i].s, 1, out[i].l, stdout);
		last_ii = ii;
		n_pairs += n; cnt_chg_tot += cnt_chg;
		n_tot[0] += nt0; n_tot[1] += nt1; n_mapped[0] += nm0; n_mapped[1] += nm1;
		pbatch_done(&R, b);
	}
	pthread_join(tid, 0);
	pfree_done(&R);
	if (B) bam_out_close(B);
	for (j = 0; j < 2; ++j) { bwa_seq_close(R.ks[j]); inflate_join(&I[j]); err_fclose(fp_sa[j]); }
	err_fflush(stdout);
	if (verbose) {
		fprintf(stderr, "[galn sampe] %lld pairs in %.2f s (index %.2f s), %d threads; %lld alignments moved by "
		        "pairing; %llu of %llu Q%d singletons mated, %llu of %llu Q%d discordant pairs fixed\n",
		        n_pairs, omp_get_wtime() - t0, t_idx, omp_get_max_threads(), cnt_chg_tot,
		        (unsigned long long)n_mapped[1], (unsigned long long)n_tot[1], SW_MIN_MAPQ,
		        (unsigned long long)n_mapped[0], (unsigned long long)n_tot[0], SW_MIN_MAPQ);
	}
	for (i = 0; i < 2 * SAMPE_BATCH; ++i) { free(out[i].s); if (rec) free(rec[i].s); }
	free(out); free(rec); free(hdr.s);
	free(pac);
	bwt_destroy(bwt);
	bns_destroy(bns);
	free(prefix); free(rg_line); free(pg.s); free(popt);
	return 0;
}
