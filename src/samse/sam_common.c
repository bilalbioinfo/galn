/* sam_common.c — see sam_common.h. sam_line, sam_header and refine_read are copies of BWA 0.7.19 code
 * (bwase.c, bwa.c) that work on one read and write to a string, so that reads can be processed in parallel. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include "sam_common.h"
#include "bwase.h"
#include "utils.h"

/* pos_5 (bwase.c, static there) */
static int64_t pos_5(const bwa_seq_t *p)
{
	if (p->type != BWA_TYPE_NO_MATCH)
		return p->strand? pos_end(p) : p->pos;
	return -1;
}

/* bwa_print_seq (bwase.c), into a string */
static void put_seq(kstring_t *o, const bwa_seq_t *seq)
{
	int i;
	if (seq->strand == 0) {
		for (i = 0; i < seq->full_len; ++i) kputc("ACGTN"[seq->seq[i]], o);
	} else {
		for (i = seq->full_len - 1; i >= 0; --i) kputc("TGCAN"[seq->seq[i]], o);
	}
}

/* bwa_print_sam1 (bwase.c), writing the SAM line into `o` instead of stdout. `mate` is NULL for
 * single-end reads. Like the original, it reverses p->qual of reverse-strand reads. */
void sam_line(kstring_t *o, const bntseq_t *bns, bwa_seq_t *p, const bwa_seq_t *mate, int mode, int max_top2)
{
	int j;
	if (p->type != BWA_TYPE_NO_MATCH || (mate && mate->type != BWA_TYPE_NO_MATCH)) {
		int seqid, nn, am = 0, flag = p->extra_flag;
		char XT;

		if (p->type == BWA_TYPE_NO_MATCH) {
			p->pos = mate->pos;
			p->strand = mate->strand;
			flag |= SAM_FSU;
			j = 1;
		} else j = pos_end(p) - p->pos; // j is the length of the reference in the alignment

		// get seqid
		nn = bns_cnt_ambi(bns, p->pos, j, &seqid);
		if (p->type != BWA_TYPE_NO_MATCH && p->pos + j - bns->anns[seqid].offset > bns->anns[seqid].len)
			flag |= SAM_FSU; // flag UNMAP as this alignment bridges two adjacent reference sequences

		// update flag and print it
		if (p->strand) flag |= SAM_FSR;
		if (mate) {
			if (mate->type != BWA_TYPE_NO_MATCH) {
				if (mate->strand) flag |= SAM_FMR;
			} else flag |= SAM_FMU;
		}
		ksprintf(o, "%s\t%d\t%s\t", p->name, flag, bns->anns[seqid].name);
		ksprintf(o, "%d\t%d\t", (int)(p->pos - bns->anns[seqid].offset + 1), p->mapQ);

		// print CIGAR
		if (p->cigar) {
			for (j = 0; j != p->n_cigar; ++j)
				ksprintf(o, "%d%c", __cigar_len(p->cigar[j]), "MIDS"[__cigar_op(p->cigar[j])]);
		} else if (p->type == BWA_TYPE_NO_MATCH) kputc('*', o);
		else ksprintf(o, "%dM", p->len);

		// print mate coordinate
		if (mate && mate->type != BWA_TYPE_NO_MATCH) {
			int m_seqid;
			long long isize;
			am = mate->seQ < p->seQ? mate->seQ : p->seQ; // smaller single-end mapping quality
			bns_cnt_ambi(bns, mate->pos, mate->len, &m_seqid);
			ksprintf(o, "\t%s\t", (seqid == m_seqid)? "=" : bns->anns[m_seqid].name);
			isize = (seqid == m_seqid)? pos_5(mate) - pos_5(p) : 0;
			if (p->type == BWA_TYPE_NO_MATCH) isize = 0;
			ksprintf(o, "%d\t%lld\t", (int)(mate->pos - bns->anns[m_seqid].offset + 1), isize);
		} else if (mate) ksprintf(o, "\t=\t%d\t0\t", (int)(p->pos - bns->anns[seqid].offset + 1));
		else kputs("\t*\t0\t0\t", o);

		// print sequence and quality
		put_seq(o, p);
		kputc('\t', o);
		if (p->qual) {
			if (p->strand) seq_reverse(p->len, p->qual, 0); // reverse quality
			kputs((char*)p->qual, o);
		} else kputc('*', o);

		if (bwa_rg_id[0]) ksprintf(o, "\tRG:Z:%s", bwa_rg_id);
		if (p->bc[0]) ksprintf(o, "\tBC:Z:%s", p->bc);
		if (p->clip_len < p->full_len) ksprintf(o, "\tXC:i:%d", p->clip_len);
		if (p->type != BWA_TYPE_NO_MATCH) {
			int i;
			// calculate XT tag
			XT = "NURM"[p->type];
			if (nn > 10) XT = 'N';
			// print tags
			ksprintf(o, "\tXT:A:%c\t%s:i:%d", XT, (mode & BWA_MODE_COMPREAD)? "NM" : "CM", p->nm);
			if (nn) ksprintf(o, "\tXN:i:%d", nn);
			if (mate) ksprintf(o, "\tSM:i:%d\tAM:i:%d", p->seQ, am);
			if (p->type != BWA_TYPE_MATESW) { // X0 and X1 are not available for this type of alignment
				ksprintf(o, "\tX0:i:%d", p->c1);
				if (p->c1 <= max_top2) ksprintf(o, "\tX1:i:%d", p->c2);
			}
			ksprintf(o, "\tXM:i:%d\tXO:i:%d\tXG:i:%d", p->n_mm, p->n_gapo, p->n_gapo+p->n_gape);
			if (p->md) ksprintf(o, "\tMD:Z:%s", p->md);
			// print multiple hits
			if (p->n_multi) {
				kputs("\tXA:Z:", o);
				for (i = 0; i < p->n_multi; ++i) {
					bwt_multi1_t *q = p->multi + i;
					int k;
					j = pos_end_multi(q, p->len) - q->pos;
					nn = bns_cnt_ambi(bns, q->pos, j, &seqid);
					ksprintf(o, "%s,%c%d,", bns->anns[seqid].name, q->strand? '-' : '+',
					         (int)(q->pos - bns->anns[seqid].offset + 1));
					if (q->cigar) {
						for (k = 0; k < q->n_cigar; ++k)
							ksprintf(o, "%d%c", __cigar_len(q->cigar[k]), "MIDS"[__cigar_op(q->cigar[k])]);
					} else ksprintf(o, "%dM", p->len);
					ksprintf(o, ",%d;", q->gap + q->mm);
				}
			}
		}
		kputc('\n', o);
	} else { // this read has no match
		int flag = p->extra_flag | SAM_FSU;
		if (mate && mate->type == BWA_TYPE_NO_MATCH) flag |= SAM_FMU;
		ksprintf(o, "%s\t%d\t*\t0\t0\t*\t*\t0\t0\t", p->name, flag);
		put_seq(o, p);
		kputc('\t', o);
		if (p->qual) {
			if (p->strand) seq_reverse(p->len, p->qual, 0); // reverse quality
			kputs((char*)p->qual, o);
		} else kputc('*', o);
		if (bwa_rg_id[0]) ksprintf(o, "\tRG:Z:%s", bwa_rg_id);
		if (p->bc[0]) ksprintf(o, "\tBC:Z:%s", p->bc);
		if (p->clip_len < p->full_len) ksprintf(o, "\tXC:i:%d", p->clip_len);
		kputc('\n', o);
	}
}

/* CIGAR of an alternative hit, with the part trimmed off the read (bwa aln -q) soft-clipped as
 * bwa_correct_trimmed does for the main hit */
static void put_cigar(kstring_t *o, const bwa_cigar_t *c, int n, int len, int clip, int strand)
{
	int j;
	if (!c) {
		if (clip && strand) ksprintf(o, "%dS", clip);
		ksprintf(o, "%dM", len);
		if (clip && !strand) ksprintf(o, "%dS", clip);
		return;
	}
	for (j = 0; j < n; ++j) {
		int l = __cigar_len(c[j]), op = __cigar_op(c[j]);
		if (clip && strand && j == 0) {
			if (op == FROM_S) l += clip; else ksprintf(o, "%dS", clip);
		}
		if (clip && !strand && j == n - 1 && op == FROM_S) l += clip;
		ksprintf(o, "%d%c", l, "MIDS"[op]);
		if (clip && !strand && j == n - 1 && op != FROM_S) ksprintf(o, "%dS", clip);
	}
}

int sam_secondary(kstring_t *o, const bntseq_t *bns, ubyte_t *pac, const bwa_seq_t *p, int len, int max, int mode,
                  kstring_t *md)
{
	int i, j, n = 0;
	for (i = 0; i < p->n_multi && n < max; ++i) {
		const bwt_multi1_t *q = p->multi + i;
		int seqid, nm, span = (int)(pos_end_multi(q, len) - q->pos);
		char *mdz;
		bns_cnt_ambi(bns, q->pos, span, &seqid);
		if (q->pos + span - bns->anns[seqid].offset > bns->anns[seqid].len) continue; // bridges two reference sequences
		mdz = bwa_cal_md1(q->n_cigar, q->cigar, len, q->pos, q->strand? p->rseq : p->seq, bns->l_pac, pac, md, &nm);
		ksprintf(o, "%s\t%d\t%s\t%d\t0\t", p->name, 0x100 | (q->strand? SAM_FSR : 0), bns->anns[seqid].name,
		         (int)(q->pos - bns->anns[seqid].offset + 1));
		put_cigar(o, q->cigar, q->n_cigar, len, p->full_len - len, q->strand);
		kputs("\t*\t0\t0\t", o);
		if (q->strand == 0) for (j = 0; j < p->full_len; ++j) kputc("ACGTN"[p->seq[j]], o);
		else for (j = p->full_len - 1; j >= 0; --j) kputc("TGCAN"[p->seq[j]], o);
		kputc('\t', o);
		if (p->qual) {
			if (q->strand == 0) kputsn((const char*)p->qual, p->full_len, o);
			else for (j = p->full_len - 1; j >= 0; --j) kputc(p->qual[j], o);
		} else kputc('*', o);
		if (bwa_rg_id[0]) ksprintf(o, "\tRG:Z:%s", bwa_rg_id);
		if (p->bc[0]) ksprintf(o, "\tBC:Z:%s", p->bc);
		if (p->clip_len < p->full_len) ksprintf(o, "\tXC:i:%d", p->clip_len);
		ksprintf(o, "\t%s:i:%d\tMD:Z:%s\n", (mode & BWA_MODE_COMPREAD)? "NM" : "CM", nm, mdz);
		free(mdz);
		++n;
	}
	return n;
}

/* bwa_print_sam_hdr (bwa.c), into a string */
void sam_header(kstring_t *o, const bntseq_t *bns, const char *hdr_line)
{
	int i, n_HD = 0, n_SQ = 0;
	if (hdr_line) {
		const char *p = hdr_line;
		while ((p = strstr(p, "@HD\t")) != 0) {
			if (p == hdr_line || *(p-1) == '\n') ++n_HD;
			p += 4;
		}
		p = hdr_line;
		while ((p = strstr(p, "@SQ\t")) != 0) {
			if (p == hdr_line || *(p-1) == '\n') ++n_SQ;
			p += 4;
		}
	}
	if (n_HD == 0) kputs("@HD\tVN:1.5\tSO:unsorted\tGO:query\n", o);
	if (n_SQ == 0) {
		for (i = 0; i < bns->n_seqs; ++i) {
			ksprintf(o, "@SQ\tSN:%s\tLN:%d", bns->anns[i].name, bns->anns[i].len);
			if (bns->anns[i].is_alt) kputs("\tAH:*\n", o);
			else kputc('\n', o);
		}
	}
	if (hdr_line) ksprintf(o, "%s\n", hdr_line);
	if (bwa_pg) ksprintf(o, "%s\n", bwa_pg);
}

/* bwa_refine_gapped (bwase.c), for one read and in bwa's order: reverse ->seq back, CIGAR of gapped hits
 * (alternative hits, then the main one), MD/NM, correction for trimmed reads */
void refine_read(const bntseq_t *bns, ubyte_t *pac, bwa_seq_t *s, kstring_t *md)
{
	int j, k;
	seq_reverse(s->len, s->seq, 0); // IMPORTANT: s->seq is reversed here!!!
	for (j = k = 0; j < s->n_multi; ++j) {
		bwt_multi1_t *q = s->multi + j;
		int n_cigar;
		if (q->gap) { // gapped alignment
			q->cigar = bwa_refine_gapped_core(bns->l_pac, pac, s->len, q->strand? s->rseq : s->seq, q->ref_shift, &q->pos, &n_cigar);
			q->n_cigar = n_cigar;
			if (q->cigar) s->multi[k++] = *q;
		} else s->multi[k++] = *q;
	}
	s->n_multi = k; // this squeezes out gapped alignments which failed the CIGAR generation
	if (!(s->type == BWA_TYPE_NO_MATCH || s->type == BWA_TYPE_MATESW || s->n_gapo == 0)) {
		s->cigar = bwa_refine_gapped_core(bns->l_pac, pac, s->len, s->strand? s->rseq : s->seq, s->ref_shift, &s->pos, &s->n_cigar);
		if (s->cigar == 0) s->type = BWA_TYPE_NO_MATCH;
	}
	if (s->type != BWA_TYPE_NO_MATCH) { // MD tag
		int nm;
		s->md = bwa_cal_md1(s->n_cigar, s->cigar, s->len, s->pos, s->strand? s->rseq : s->seq, bns->l_pac, pac, md, &nm);
		s->nm = nm;
	}
	bwa_correct_trimmed(s);
}

bwt_t *load_bwt_sa(const char *prefix)
{
	kstring_t fn = {0, 0, 0};
	bwt_t *bwt;
	ksprintf(&fn, "%s.bwt", prefix);
	bwt = bwt_restore_bwt(fn.s);
	fn.l = 0; ksprintf(&fn, "%s.sa", prefix);
	bwt_restore_sa(fn.s, bwt);
	free(fn.s);
	return bwt;
}

ubyte_t *load_pac(const bntseq_t *bns)
{
	ubyte_t *pac = (ubyte_t*)calloc(bns->l_pac/4+1, 1);
	err_rewind(bns->fp_pac);
	err_fread_noeof(pac, 1, bns->l_pac/4+1, bns->fp_pac);
	return pac;
}

/* ---- decompression thread: gzip FASTQ -> pipe (plain input passes through unchanged) ---- */
static void *inflate_main(void *arg)
{
	inflate_t *a = (inflate_t*)arg;
	const int bs = 1 << 20;
	char *buf = (char*)malloc(bs);
	int n;
	sigset_t set; // a closed pipe (the reader stopped early) ends this thread instead of the process
	sigemptyset(&set);
	sigaddset(&set, SIGPIPE);
	pthread_sigmask(SIG_BLOCK, &set, 0);
	while ((n = gzread(a->in, buf, bs)) > 0) {
		char *p = buf;
		while (n > 0) {
			ssize_t w = write(a->fd, p, n);
			if (w < 0 && errno == EINTR) continue;
			if (w < 0 && errno == EPIPE) { n = 0; goto done; }
			if (w < 0) { fprintf(stderr, "[galn] cannot pass %s to the reader\n", a->fn); exit(1); }
			p += w; n -= (int)w;
		}
	}
done:
	if (n < 0) {
		int e;
		fprintf(stderr, "[galn] error reading %s: %s\n", a->fn, gzerror(a->in, &e));
		exit(1);
	}
	close(a->fd);
	gzclose(a->in);
	free(buf);
	return 0;
}

bwa_seqio_t *open_reads_threaded(int mode, const char *fn, inflate_t *I)
{
	int fds[2];
	char path[64];
	memset(I, 0, sizeof(*I));
	if (mode & BWA_MODE_BAM) return bwa_open_reads(mode, fn); // BAM input, as bwa
	I->fn = fn;
	if ((I->in = gzopen(fn, "r")) == 0) {
		fprintf(stderr, "[galn] cannot open %s\n", fn);
		exit(1);
	}
	gzbuffer(I->in, 1 << 20);
	if (pipe(fds) != 0) { perror("[galn] pipe"); exit(1); }
#ifdef F_SETPIPE_SZ
	fcntl(fds[1], F_SETPIPE_SZ, 1 << 20);
#endif
	I->fd = fds[1];
	I->on = 1;
	pthread_create(&I->tid, 0, inflate_main, I);
	snprintf(path, sizeof(path), "/dev/fd/%d", fds[0]);
	return bwa_open_reads(mode, path);
}

void inflate_join(inflate_t *I)
{
	if (I->on) pthread_join(I->tid, 0);
	I->on = 0;
}
