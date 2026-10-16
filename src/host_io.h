/* host_io.h — host-side plumbing: load the .bwt index and read/encode FASTQ.
 * This stays on the host in both the CPU and GPU builds. */
#ifndef HOST_IO_H
#define HOST_IO_H

#include <stdint.h>
#include "bwtsearch.h"

typedef struct {
    char    *name;      /* read name (without leading '@', up to first whitespace) */
    int      len;       /* read length */
    int      batch;     /* upstream batch index (bwa_read_seq: 0x40000 reads, or fewer before a malformed record) */
    uint8_t *query;     /* len bytes: nt4 -> reverse -> complement, exactly as BWA feeds the search */
} read_t;

/* bwa_aln_core reads 0x40000 reads per batch */
#define HOST_ALN_BATCH 0x40000

/* Load a .bwt file into a freshly-malloc'd uint32 array + fill the descriptor.
 * Returns the bwt array (caller frees) or NULL on error. */
uint32_t *load_bwt(const char *fn, bwtgpu_meta_t *m);

/* Read FASTQ/FASTA (plain or gzip) with bwa's own kseq parser and bwa_read_seq's rules:
 * empty reads are skipped, a malformed record (kseq_read == -2) is dropped and ends the
 * current batch, and reading stops if a batch would be empty. Each read gets its batch
 * index. Returns NULL on open or read errors. Encodes each read's search query =
 * complement(nt4(read[len-1-j])). */
read_t *read_fastq(const char *fn, int *n);

/* Contiguous batch ranges: *start gets n_batch+1 entries (start[b]..start[b+1]-1). Returns n_batch. */
int     read_batch_ranges(const read_t *r, int n, int **start);
void    free_reads(read_t *r, int n);

#endif
