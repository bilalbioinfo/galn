/* sai_format.h — exact on-disk layout of BWA's .sai file and option block.
 *
 * Verified byte-for-byte against lh3/bwa 0.7.19-r1273 (bwtaln.h, bwtaln.c).
 * The .sai stream is:
 *     "SAI\1"                      (4 bytes, no NUL beyond the \1)
 *     gap_opt_t                    (sizeof == 64 bytes, one struct)
 *     per read: int32 n_aln, then n_aln * sizeof(bwt_aln1_t) (==24) bytes
 */
#ifndef SAI_FORMAT_H
#define SAI_FORMAT_H

#include <stdint.h>

#define SAI_MAGIC_STR "SAI\x01"
#define SAI_MAGIC_LEN 4

/* gap_opt_t — field order MUST match bwtaln.h exactly (64 bytes on LP64). */
typedef struct {
    int32_t s_mm, s_gapo, s_gape;
    int32_t mode;                 /* bits 24-31 hold barcode length */
    int32_t indel_end_skip, max_del_occ, max_entries;
    float   fnr;
    int32_t max_diff, max_gapo, max_gape;
    int32_t max_seed_diff, seed_len;
    int32_t n_threads;
    int32_t max_top2;
    int32_t trim_qual;
} sai_gap_opt_t;

/* bwt_aln1_t — 24 bytes: one packed 64-bit word then two 64-bit SA bounds. */
typedef struct {
    uint64_t packed;   /* n_mm:8, n_gapo:8, n_gape:8, score:20, n_ins:10, n_del:10 */
    uint64_t k, l;
} sai_aln1_t;

/* helpers to build/read the packed word, LSB-first to match the C bitfield ABI */
static inline uint64_t sai_pack(uint32_t n_mm, uint32_t n_gapo, uint32_t n_gape,
                                uint32_t score, uint32_t n_ins, uint32_t n_del)
{
    return ((uint64_t)(n_mm   & 0xff))
         | ((uint64_t)(n_gapo & 0xff)   << 8)
         | ((uint64_t)(n_gape & 0xff)   << 16)
         | ((uint64_t)(score  & 0xfffff)<< 24)
         | ((uint64_t)(n_ins  & 0x3ff)  << 44)
         | ((uint64_t)(n_del  & 0x3ff)  << 54);
}

#endif /* SAI_FORMAT_H */
