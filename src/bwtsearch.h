/* bwtsearch.h — DUAL-TARGET occurrence layer + exact-match search.
 *
 * Every function here is a fresh reimplementation of BWA's occurrence + backward-search primitives,
 * written to compile UNCHANGED as:
 *   - plain host C   (gcc/g++)  -> used by the CPU workers
 *   - CUDA device code (nvcc)   -> used by the GPU kernels
 *
 * The BWT is passed as raw, read-only pointers plus a tiny descriptor, so
 * this header has NO dependency on BWA's headers or on any host struct. That
 * is deliberate: the index is shared read-only by all CPU workers and GPU threads.
 *
 * Layout assumption: OCC_INTERVAL == 0x80 (128), matching upstream bwt.h.
 * Verified against lh3/bwa 0.7.19-r1273.
 */
#ifndef BWTSEARCH_H
#define BWTSEARCH_H

#include <stdint.h>

/* Host/device decoration: expands to __host__ __device__ under nvcc, empty otherwise. */
#if defined(__CUDACC__)
  #define BWTGPU_HD __host__ __device__
#else
  #define BWTGPU_HD
#endif

#define BWTGPU_OCC_INTERVAL   (0x80ULL)
#define BWTGPU_OCC_INTV_MASK  (BWTGPU_OCC_INTERVAL - 1)

/* Read-only index descriptor. Small enough to live in constant memory on GPU. */
typedef struct {
    uint64_t primary;    /* S^{-1}(0) */
    uint64_t L2[5];      /* cumulative base counts C(); L2[0]==0 */
    uint64_t seq_len;    /* length of the BWT string (== L2[4]) */
    uint64_t bwt_size;   /* number of uint32 words in the bwt array */
} bwtgpu_meta_t;

/* pointer to the OCC_INTERVAL block containing position k (valid only for OCC_INTERVAL==0x80) */
#define BWTGPU_OCC_INTV(bwt, k) ((bwt) + (((k) >> 7) << 4))

/* popcount-style base counter — verbatim port of BWA's __occ_aux */
BWTGPU_HD static inline int bwtgpu_occ_aux(uint64_t y, int c)
{
    /* reduce nucleotide counting to bit counting */
    y = ((c & 2) ? y : ~y) >> 1 & ((c & 1) ? y : ~y) & 0x5555555555555555ULL;
    y = (y & 0x3333333333333333ULL) + (y >> 2 & 0x3333333333333333ULL);
    return (int)(((y + (y >> 4)) & 0x0f0f0f0f0f0f0f0fULL) * 0x0101010101010101ULL >> 56);
}

/* Occ(c, k): number of occurrences of base c in bwt[0..k]. Verbatim port of bwt_occ. */
BWTGPU_HD static inline uint64_t bwtgpu_occ(const uint32_t *bwt, const bwtgpu_meta_t *m,
                                            uint64_t k, uint8_t c)
{
    uint64_t n;
    const uint32_t *p, *end;

    if (k == m->seq_len) return m->L2[c + 1] - m->L2[c];
    if (k == (uint64_t)(-1)) return 0;
    k -= (k >= m->primary); /* because $ is not in bwt */

    p = BWTGPU_OCC_INTV(bwt, k);
    n = ((const uint64_t*)p)[c];   /* cnt[c] stored as uint64 at block start */
    p += sizeof(uint64_t);         /* skip the 4x uint64 counts (== 8 uint32 words) */

    /* sum whole 32-base cells up to the one containing k */
    end = p + ((((k >> 5) - ((k & ~BWTGPU_OCC_INTV_MASK) >> 5)) << 1));
    for (; p < end; p += 2)
        n += bwtgpu_occ_aux((uint64_t)p[0] << 32 | p[1], c);

    /* the final partial cell */
    n += bwtgpu_occ_aux(((uint64_t)p[0] << 32 | p[1]) & ~((1ULL << ((~k & 31) << 1)) - 1), c);
    if (c == 0) n -= ~k & 31; /* correct for the masked-in bits */

    return n;
}

/* Occ for two positions at once. Verbatim port of bwt_2occ. */
BWTGPU_HD static inline void bwtgpu_2occ(const uint32_t *bwt, const bwtgpu_meta_t *m,
                                         uint64_t k, uint64_t l, uint8_t c,
                                         uint64_t *ok, uint64_t *ol)
{
    uint64_t _k, _l;
    _k = (k >= m->primary) ? k - 1 : k;
    _l = (l >= m->primary) ? l - 1 : l;
    if (_l / BWTGPU_OCC_INTERVAL != _k / BWTGPU_OCC_INTERVAL
        || k == (uint64_t)(-1) || l == (uint64_t)(-1)) {
        *ok = bwtgpu_occ(bwt, m, k, c);
        *ol = bwtgpu_occ(bwt, m, l, c);
    } else {
        uint64_t mm, n, i, j;
        const uint32_t *p;
        if (k >= m->primary) --k;
        if (l >= m->primary) --l;
        p = BWTGPU_OCC_INTV(bwt, k);
        n = ((const uint64_t*)p)[c];
        p += sizeof(uint64_t);
        /* *ok */
        j = k >> 5 << 5;
        for (i = k / BWTGPU_OCC_INTERVAL * BWTGPU_OCC_INTERVAL; i < j; i += 32, p += 2)
            n += bwtgpu_occ_aux((uint64_t)p[0] << 32 | p[1], c);
        mm = n;
        n += bwtgpu_occ_aux(((uint64_t)p[0] << 32 | p[1]) & ~((1ULL << ((~k & 31) << 1)) - 1), c);
        if (c == 0) n -= ~k & 31;
        *ok = n;
        /* *ol */
        j = l >> 5 << 5;
        for (; i < j; i += 32, p += 2)
            mm += bwtgpu_occ_aux((uint64_t)p[0] << 32 | p[1], c);
        mm += bwtgpu_occ_aux(((uint64_t)p[0] << 32 | p[1]) & ~((1ULL << ((~l & 31) << 1)) - 1), c);
        if (c == 0) mm -= ~l & 31;
        *ol = mm;
    }
}

/* Exact backward search. Verbatim port of bwt_match_exact (interval form).
 * str must be the query already transformed as BWA feeds the search:
 *   nt4-encode -> reverse -> complement (3-base), N stays >3.
 * Returns number of hits (l-k+1) and sets [*sa_begin,*sa_end]; 0 if no exact match. */
BWTGPU_HD static inline uint64_t bwtgpu_match_exact(const uint32_t *bwt, const bwtgpu_meta_t *m,
                                                    int len, const uint8_t *str,
                                                    uint64_t *sa_begin, uint64_t *sa_end)
{
    uint64_t k, l, ok, ol;
    int i;
    k = 0; l = m->seq_len;
    for (i = len - 1; i >= 0; --i) {
        uint8_t c = str[i];
        if (c > 3) return 0;                 /* N -> no match */
        bwtgpu_2occ(bwt, m, k - 1, l, c, &ok, &ol);
        k = m->L2[c] + ok + 1;
        l = m->L2[c] + ol;
        if (k > l) break;
    }
    if (k > l) return 0;
    if (sa_begin) *sa_begin = k;
    if (sa_end)   *sa_end = l;
    return l - k + 1;
}

#endif /* BWTSEARCH_H */
