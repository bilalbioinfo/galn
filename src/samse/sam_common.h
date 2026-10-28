/* sam_common.h — shared by galn samse and galn sampe: BWA's per-read output steps (copies of
 * bwase.c code working on one read and writing to a string) and input/index helpers. */
#ifndef GALN_SAM_COMMON_H
#define GALN_SAM_COMMON_H

#include <pthread.h>
#include <zlib.h>
#include "bntseq.h"
#include "bwt.h"
#include "bwtaln.h"
#include "kstring.h"

/* bwase.c functions that bwase.h does not declare */
void bwa_aln2seq_core(int n_aln, const bwt_aln1_t *aln, bwa_seq_t *s, int set_main, int n_multi);
bwa_cigar_t *bwa_refine_gapped_core(bwtint_t l_pac, const ubyte_t *pacseq, int len, ubyte_t *seq, int ref_shift,
                                    bwtint_t *_rb, int *n_cigar);
char *bwa_cal_md1(int n_cigar, bwa_cigar_t *cigar, int len, bwtint_t pos, ubyte_t *seq, bwtint_t l_pac,
                  ubyte_t *pacseq, kstring_t *str, int *_nm);
void bwa_correct_trimmed(bwa_seq_t *s);
int64_t pos_end_multi(const bwt_multi1_t *p, int len);

/* bwa_shim.c */
extern char *bwa_pg;
extern char bwa_rg_id[256];
bwa_seqio_t *bwa_open_reads(int mode, const char *fn_fa);
char *bwa_set_rg(const char *s);
char *bwa_idx_infer_prefix(const char *hint);

/* bwa_print_sam_hdr (bwa.c), into a string */
void sam_header(kstring_t *o, const bntseq_t *bns, const char *hdr_line);

/* bwa_print_sam1 (bwase.c), into a string; mate is NULL for single-end reads */
void sam_line(kstring_t *o, const bntseq_t *bns, bwa_seq_t *p, const bwa_seq_t *mate, int mode, int max_top2);

/* galn samse -k: the alternative hits p->multi[] (positions and CIGARs as for the XA tag, after refine_read) as SAM
 * records of their own: flag 256, MAPQ 0, the read on the hit's strand, NM and MD as for the main hit (bwa_cal_md1).
 * len: the read length before bwa_correct_trimmed (the aligned part). Hits that bridge two reference sequences are
 * left out. Call before sam_line (which reverses p->qual). Writes at most max records; returns their number. */
int sam_secondary(kstring_t *o, const bntseq_t *bns, ubyte_t *pac, const bwa_seq_t *p, int len, int max, int mode,
                  kstring_t *md);

/* bwa_refine_gapped (bwase.c) for one read: CIGAR of gapped hits, MD/NM, trimmed reads */
void refine_read(const bntseq_t *bns, ubyte_t *pac, bwa_seq_t *s, kstring_t *md);

/* the index, once: .bwt + .sa (positions) and .pac (CIGAR, MD, mate rescue) */
bwt_t *load_bwt_sa(const char *prefix);
ubyte_t *load_pac(const bntseq_t *bns);

/* Open reads for bwa_read_seq. A gzip FASTQ is decompressed on its own thread into a pipe that BWA's
 * reader opens (it then sees plain FASTQ), so decompression and parsing overlap; BAM input is opened
 * as by bwa. After bwa_seq_close() of the reader, call inflate_join(). */
typedef struct { gzFile in; int fd; const char *fn; pthread_t tid; int on; } inflate_t;
bwa_seqio_t *open_reads_threaded(int mode, const char *fn, inflate_t *I);
void inflate_join(inflate_t *I);

#endif
