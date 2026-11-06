/* idxcommon.h: shared helpers for the G-ALN index builders (CPU reference and GPU).
 *
 * Everything here reproduces the on-disk conventions of BWA 0.7.19 `bwa index`:
 *   - text T = F . revcomp(F), F = forward 2-bit sequence stored in <prefix>.pac
 *   - BWT of T$ ($ smallest), $ removed, `primary` = row of $ in the (n+1)-row matrix
 *   - .bwt = primary, L2[1..4], then 128-symbol blocks [4 x u64 Occ][8 x u32 BWT words], final 4 x u64
 *   - .sa  = primary, L2[1..4], (u64)32, n, SA'[32], SA'[64], ..., SA'[32*floor(n/32)]
 */
#ifndef GALN_IDXCOMMON_H
#define GALN_IDXCOMMON_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GALN_SA_INTV 32

/* galn index -v: also print the time of each step */
extern int galn_index_verbose;
#define GALN_VLOG(...) do { if (galn_index_verbose) fprintf(stderr, __VA_ARGS__); } while (0)

double galn_realtime(void);

/* Length (in bases) of a .pac file, exactly as bwa_seq_len() computes it. */
int64_t galn_pac_len(const char *fn_pac);

/* Read the 2-bit packed .pac file (forward strand only, as left on disk by `bwa index`).
 * Returns the packed bytes (malloc'd) and sets *l_pac. */
uint8_t *galn_read_pac(const char *fn_pac, int64_t *l_pac);

/* Expand forward pac into T[0..2*l_pac-1] = F . revcomp(F), one symbol (0..3) per byte.
 * T must have room for 2*l_pac bytes. OpenMP-parallel if compiled with -fopenmp. */
void galn_pac_to_text(const uint8_t *pac, int64_t l_pac, uint8_t *T);

/* Number of uint32 words of the bwa in-memory/on-disk BWT body (after bwt_bwtupdate_core). */
static inline uint64_t galn_bwt_words(uint64_t n)
{
	uint64_t nb = (n + 127) / 128;
	return 8 * (nb + 1) + (n + 15) / 16;
}

/* Build the bwa interleaved BWT+Occ body from L[0..n-1] ($ removed, symbols 0..3).
 * buf must have galn_bwt_words(n) words. Fills L2[0..4] (L2[0]=0, L2[4]=n). */
void galn_pack_bwt(const uint8_t *L, uint64_t n, uint32_t *buf, uint64_t L2[5]);

/* The same from the BWT with the $ row kept: L[0..n], $ at index skip (its symbol is left out). */
void galn_pack_bwt_skip(const uint8_t *L, uint64_t n, uint64_t skip, uint32_t *buf, uint64_t L2[5]);

/* Write <fn> in bwt_dump_bwt() format. */
int galn_write_bwt(const char *fn, uint64_t primary, const uint64_t L2[5], const uint32_t *buf, uint64_t n_words);

/* Write <fn> in bwt_dump_sa() format. samples[0..n_samples-1] are SA'[32], SA'[64], ...
 * (i.e. bwt->sa[1..n_sa-1]); n_samples must equal (n+32)/32 - 1. */
int galn_write_sa(const char *fn, uint64_t primary, const uint64_t L2[5], uint64_t n, const uint64_t *samples, uint64_t n_samples);

/* FASTA -> <prefix>.pac/.ann/.amb, exactly the files `bwa index` leaves on disk (upstream bns_fasta2bntseq(...,
 * for_only=1)), read and parsed on all threads (fasta_pac.c). Returns l_pac (forward length); with pac_out, also the
 * forward .pac bytes (malloc'd) as galn_read_pac() would read them back. */
int64_t galn_fasta_to_pac(const char *fn_fa, const char *prefix, uint8_t **pac_out);

/* The same with the unmodified upstream bns_fasta2bntseq() on one thread */
int64_t galn_fasta_to_pac_serial(const char *fn_fa, const char *prefix);

#ifdef __cplusplus
}
#endif

#endif
