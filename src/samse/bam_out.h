/* bam_out.h — minimal BAM writer for galn samse/sampe: SAM text lines -> BAM records, BGZF blocks
 * compressed in parallel (OpenMP), written in order. No htslib needed; output is standard BAM
 * (SAM spec v1), readable by samtools/htslib. */
#ifndef GALN_BAM_OUT_H
#define GALN_BAM_OUT_H

#include <stdio.h>
#include "kstring.h"
#include "bntseq.h"

typedef struct bam_out_s bam_out_t;

/* Open a BAM stream on fp: writes the BAM header (SAM header text `hdr` + the references of bns).
 * level: zlib compression level 0-9. */
bam_out_t *bam_out_open(FILE *fp, int level, const char *hdr, const bntseq_t *bns);

/* Encode one SAM line (with or without its trailing newline) as a BAM record, appended to `rec`.
 * Thread-safe; returns 0, or -1 if the line cannot be parsed. */
int bam_out_encode(const bam_out_t *b, const char *line, int len, kstring_t *rec);

/* Append n encoded records (in order) to the stream; full BGZF blocks are compressed in parallel
 * and written. */
void bam_out_write(bam_out_t *b, int n, const kstring_t *recs);

/* Flush the last block, write the BGZF end-of-file marker and free b (fp is not closed). */
void bam_out_close(bam_out_t *b);

#endif
