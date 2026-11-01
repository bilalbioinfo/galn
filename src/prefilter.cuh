/* prefilter.cuh — search-scheme pre-filter: proves on the GPU that a read has NO alignment within
 * bwa aln's bounds, so that none of the exact engines (light tier, warp engine, CPU) searches it.
 *
 * Ported from gpualn (https://github.com/teepean/gpualn: cuda/scheme_engine.cuh, cuda/schemes.cuh,
 * cuda/fm_device.cuh), Copyright (C) 2026 teepean, GPLv3 -- the licence of G-ALN.
 *
 * WHY. A read without a hit is the most expensive read for bwt_match_gap: nothing is ever found, so
 * nothing ends the search early and every path within max_diff is enumerated. Low-endogenous
 * libraries consist mostly of such reads, and even in a library that is 94% endogenous the few %
 * unmapped reads take most of the search time.
 *
 * WHAT. A search scheme (Kucherov et al. 2016; Kianfar et al. 2018) splits the read into P parts
 * and runs a few searches on a bidirectional FM-index, each with lower/upper error bounds at the part
 * boundaries, such that every distribution of <= K errors over the parts is admitted by some search
 * ("covering"). Most of the max_diff-deep backtracking tree is never entered. bwa's BWT is built
 * over S = T . revcomp(T), so its own .bwt already supports extension in both directions (the trick
 * of bwa mem's bwt_extend): no second index and no extra GPU memory.
 *
 * CONTRACT. The answer is PF_NONE ("no hit") or PF_MAYBE. PF_NONE for a read on which bwa finds an
 * alignment would be a wrong .sai, and two things rule it out:
 *   1. every table is checked to be covering, exhaustively, before it is used (pf_validate);
 *   2. the executor is strictly MORE permissive than bwt_match_gap: it enforces max_diff, max_gapo and
 *      max_gape, which bwa's alignments obey, and drops indel_end_skip, max_del_occ, the seed
 *      constraint (-l/-k), the width lower bound and the max_entries cut-off, which only ever remove
 *      candidates.
 * It counts gap extensions against max_diff, as bwa does in BWA_MODE_GAPE (the default; not with
 * -e), so the caller runs it only in that mode. Reads it cannot answer -- max_diff 0 or above 4,
 * longer than 127 bp, or a stack/budget that runs out -- are PF_MAYBE and are searched as before.
 *
 * WHERE IT PAYS. Only reads no longer than the seed length (-l) are filtered. Longer reads are seeded
 * by bwa: at most -k differences in the seed, which makes the exact search cheap, while the filter
 * ignores the seed and would search far more than the exact engines do. With the aDNA settings
 * (-l 1024 or -l 16500) no read is seeded, so every read is filtered.
 */
#ifndef PREFILTER_CUH
#define PREFILTER_CUH

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define PF_NONE  0     /* proven: no alignment within the bounds */
#define PF_MAYBE 1     /* has a hit, or not answered: search the read exactly */
#define PF_TODO  2     /* internal: left for the scheme search */
#define PF_EXACT 3     /* internal: occurs exactly, so it has a hit (reported as PF_MAYBE) */

#define PF_MAXK   4    /* largest K with a table */
#define PF_MAXP   8
#define PF_MAXS   8
#define PF_MAXLEN 127  /* the node packs window ends in 7 bits */
#define PF_WPB    2    /* warps per block */
#define PF_CAP    1280 /* stack entries per warp (shared memory) */
#define PF_BUDGET 2000000ULL   /* node pops per read before giving up (PF_MAYBE) */

/* -------------------------------------------------------------------------------------------- */
/* occurrence counts on bwa's .bwt layout (64-byte blocks: 4 x uint64 counts + 128 bases),        */
/* returned in named registers -- an array argument would be address-taken and live in local     */
/* memory, which costs DRAM traffic on every node expansion                                      */

__constant__ uint64_t pf_L2[5];

__device__ __forceinline__ uint32_t pf_occ_aux4(uint32_t b)
{
    /* bit-sliced count of A/C/G/T in 16 bases, packed like bwa's cnt_table (A | C<<8 | G<<16 | T<<24) */
    uint32_t lo = b & 0x55555555u, hi = (b >> 1) & 0x55555555u;
    uint32_t nlo = ~lo & 0x55555555u, nhi = ~hi & 0x55555555u;
    return (uint32_t)__popc(nlo & nhi)        | ((uint32_t)__popc(lo & nhi) << 8)
         | ((uint32_t)__popc(nlo & hi) << 16) | ((uint32_t)__popc(lo & hi)  << 24);
}

/* bwt_occ4 */
__device__ __forceinline__ void pf_occ4(const uint32_t *__restrict__ bwt, uint64_t primary, uint64_t k,
                                        uint64_t &c0, uint64_t &c1, uint64_t &c2, uint64_t &c3)
{
    if (k == (uint64_t)-1) { c0 = c1 = c2 = c3 = 0; return; }
    k -= (k >= primary);
    const uint32_t *p = bwt + ((k >> 7) << 4);
    const uint64_t *pc = (const uint64_t *)p;
    c0 = __ldg(pc + 0); c1 = __ldg(pc + 1); c2 = __ldg(pc + 2); c3 = __ldg(pc + 3);
    p += 8;
    uint64_t x = 0;
    const uint32_t *end = p + ((k >> 4) - ((k & ~0x7fULL) >> 4));
    for (; p < end; ++p) x += pf_occ_aux4(__ldg(p));
    uint32_t tmp = __ldg(p) & ~((1U << ((~k & 15) << 1)) - 1);
    x += (uint64_t)pf_occ_aux4(tmp) - (~k & 15);
    c0 += x & 0xff; c1 += (x >> 8) & 0xff; c2 += (x >> 16) & 0xff; c3 += x >> 24;
}

/* bwt_2occ4, with one block load when k and l share a 128-base interval */
__device__ __forceinline__ void pf_2occ4(const uint32_t *__restrict__ bwt, uint64_t primary, uint64_t k, uint64_t l,
                                         uint64_t &ck0, uint64_t &ck1, uint64_t &ck2, uint64_t &ck3,
                                         uint64_t &cl0, uint64_t &cl1, uint64_t &cl2, uint64_t &cl3)
{
    uint64_t _k = k - (k >= primary), _l = l - (l >= primary);
    if ((_k >> 7) != (_l >> 7) || k == (uint64_t)-1 || l == (uint64_t)-1) {
        pf_occ4(bwt, primary, k, ck0, ck1, ck2, ck3);
        pf_occ4(bwt, primary, l, cl0, cl1, cl2, cl3);
        return;
    }
    k -= (k >= primary); l -= (l >= primary);
    const uint32_t *p = bwt + ((k >> 7) << 4);
    const uint64_t *pc = (const uint64_t *)p;
    ck0 = __ldg(pc + 0); ck1 = __ldg(pc + 1); ck2 = __ldg(pc + 2); ck3 = __ldg(pc + 3);
    p += 8;
    const uint32_t *endk = p + ((k >> 4) - ((k & ~0x7fULL) >> 4));
    const uint32_t *endl = p + ((l >> 4) - ((l & ~0x7fULL) >> 4));
    uint64_t x = 0, y;
    for (; p < endk; ++p) x += pf_occ_aux4(__ldg(p));
    y = x;
    { uint32_t t = __ldg(p) & ~((1U << ((~k & 15) << 1)) - 1); x += (uint64_t)pf_occ_aux4(t) - (~k & 15); }
    for (; p < endl; ++p) y += pf_occ_aux4(__ldg(p));
    { uint32_t t = __ldg(p) & ~((1U << ((~l & 15) << 1)) - 1); y += (uint64_t)pf_occ_aux4(t) - (~l & 15); }
    cl0 = ck0; cl1 = ck1; cl2 = ck2; cl3 = ck3;
    ck0 += x & 0xff; ck1 += (x >> 8) & 0xff; ck2 += (x >> 16) & 0xff; ck3 += x >> 24;
    cl0 += y & 0xff; cl1 += (y >> 8) & 0xff; cl2 += (y >> 16) & 0xff; cl3 += y >> 24;
}

/* -------------------------------------------------------------------------------------------- */
/* search-scheme tables                                                                          */
/* A scheme partitions the read into P equal parts; each search visits the parts in order pi      */
/* (contiguous: every step extends the matched window to an adjacent part) and bounds the         */
/* cumulative error count by L (lower) and U (upper) after each part.                             */

struct PfTab {
    int P, ns;                       /* parts, searches */
    uint8_t pi[PF_MAXS][PF_MAXP];
    uint8_t L [PF_MAXS][PF_MAXP];
    uint8_t U [PF_MAXS][PF_MAXP];
};
__constant__ PfTab pf_sch[PF_MAXK + 1];   /* indexed by K = max_diff */

/* strings: 0-based part indices / bounds, one character per part */
struct PfRaw { int K, P, ns; const char *pi[PF_MAXS]; const char *L[PF_MAXS]; const char *U[PF_MAXS]; };
static const PfRaw PF_RAW[] = {
    /* K=1, P=2 -- Kianfar et al. (arXiv:1711.02035, Table 3) */
    {1, 2, 2, {"01", "10"},
              {"00", "00"},
              {"01", "01"}},
    /* K=2, P=3 -- Kianfar et al. */
    {2, 3, 3, {"012", "210", "120"},
              {"002", "000", "011"},
              {"012", "022", "012"}},
    /* K=3, P=5 -- SeqAn3 optimum_search_scheme<0,3> */
    {3, 5, 4, {"43210", "23410", "12340", "01234"},
              {"00000", "00111", "00022", "00003"},
              {"00333", "01123", "01223", "02233"}},
    /* K=4, P=5 -- Renders et al. (Columba search_schemes/multiple_opt/4, scheme1) */
    {4, 5, 5, {"01234", "12034", "21034", "34210", "43210"},
              {"00222", "00000", "01111", "00003", "01114"},
              {"02244", "01244", "01244", "01444", "01444"}},
};

/* Exhaustive check that the table is covering (every error distribution summing to <= K over the
 * P parts passes some search) and that every visiting order is contiguous. 0 = valid. A table that
 * is not covering would give silently WRONG output, never a crash, so this runs before every use. */
static int pf_validate(const PfRaw *r)
{
    int K = r->K, P = r->P;
    for (int s = 0; s < r->ns; ++s) {
        int seen[PF_MAXP] = {0};
        for (int j = 0; j < P; ++j) {
            int p = r->pi[s][j] - '0';
            if (p < 0 || p >= P) return 1;
            seen[p] = 1;
            int lo = -1, hi = -1, cnt = 0;
            for (int t = 0; t < P; ++t) if (seen[t]) { if (lo < 0) lo = t; hi = t; ++cnt; }
            if (hi - lo + 1 != cnt) return 1;
        }
    }
    std::vector<int> d(P, 0);
    for (;;) {
        int sum = 0;
        for (int i = 0; i < P; ++i) sum += d[i];
        if (sum <= K) {
            int ok = 0;
            for (int s = 0; s < r->ns && !ok; ++s) {
                int c = 0, good = 1;
                for (int j = 0; j < P && good; ++j) {
                    c += d[r->pi[s][j] - '0'];
                    if (c < r->L[s][j] - '0' || c > r->U[s][j] - '0') good = 0;
                }
                ok = good;
            }
            if (!ok) return 1;
        }
        int i = 0;                                   /* odometer over (K+1)^P */
        for (; i < P; ++i) { if (++d[i] <= K) break; d[i] = 0; }
        if (i == P) break;
    }
    return 0;
}

/* The node keeps interval sizes in 32 bits. The largest interval it ever stores is a single base's
 * (intervals only shrink), so the filter can run when no base occurs 2^32 times or more in the BWT
 * text T.revcomp(T). Each base count there is at most the reference length, so this always holds
 * for references up to 4.29 Gbp (human: 3.1 Gbp), and up to ~8.6 Gbp at balanced composition. */
static inline bool pf_supported(const uint64_t L2[5])
{
    for (int c = 0; c < 4; ++c) if (L2[c + 1] - L2[c] > 0xffffffffULL) return false;
    return true;
}

/* Validate the tables and load them, and the index's C() array, into the CURRENT device's constant
 * memory (call once per device). 0 = success. */
static int pf_init_device(const uint64_t L2[5])
{
    PfTab h[PF_MAXK + 1];
    memset(h, 0, sizeof(h));
    for (size_t i = 0; i < sizeof(PF_RAW) / sizeof(PF_RAW[0]); ++i) {
        const PfRaw *r = &PF_RAW[i];
        if (pf_validate(r)) { fprintf(stderr, "[galn] prefilter: scheme table for K=%d is not covering\n", r->K); return 1; }
        PfTab &t = h[r->K];
        t.P = r->P; t.ns = r->ns;
        for (int s = 0; s < r->ns; ++s)
            for (int j = 0; j < r->P; ++j) {
                t.pi[s][j] = (uint8_t)(r->pi[s][j] - '0');
                t.L [s][j] = (uint8_t)(r->L [s][j] - '0');
                t.U [s][j] = (uint8_t)(r->U [s][j] - '0');
            }
    }
    if (cudaMemcpyToSymbol(pf_sch, h, sizeof(h)) != cudaSuccess) return 1;
    if (cudaMemcpyToSymbol(pf_L2, L2, 5 * sizeof(uint64_t)) != cudaSuccess) return 1;
    return 0;
}

/* -------------------------------------------------------------------------------------------- */
/* one search of one scheme for one read, warp-cooperative                                       */

#define PF_SM 0        /* match/mismatch */
#define PF_SI 1        /* in an insertion */
#define PF_SD 2        /* in a deletion */
#define PF_SE 3        /* EMPTY: nothing matched yet */

/* 24-byte node: bidirectional interval (x0 = start for the pattern, x1 = start for its reverse
 * complement, x2 = size) + packed state. x2 fits 32 bits: the largest single-base interval is
 * ~1.6e9 and intervals only shrink; the full-range interval is never stored (PF_SE marks it). */
struct PfNode { uint64_t x0, x1; uint32_t x2, p; };

__device__ __forceinline__ uint32_t pf_pack(int lo, int hi, int mm, int go, int ge, int st, int j)
{
    return (uint32_t)lo | ((uint32_t)hi << 7) | ((uint32_t)mm << 14) | ((uint32_t)go << 19)
         | ((uint32_t)ge << 22) | ((uint32_t)st << 26) | ((uint32_t)j << 28);
}

/* Returns 1 if the whole read matched within this search's bounds, or if the stack or the budget
 * ran out (then *flagged = 1: the read is not answered); 0 if the search finished without a hit.
 * An error belongs to the part holding the read base it consumes (a deletion, which consumes none,
 * to the part being extended), so every alignment induces one error distribution over the parts
 * and a covering scheme admits it. */
__device__ inline int pf_search(const uint32_t *__restrict__ bwt, uint64_t primary, const uint8_t *seq, int len,
                                int K, int si, const int *pb, int max_gapo, int max_gape,
                                PfNode *st, int cap, unsigned long long budget, int *flagged, unsigned long long *nn_io)
{
    const unsigned FULL = 0xffffffffu;
    int lane = threadIdx.x & 31;
    const PfTab &T = pf_sch[K];
    int P = T.P;
    unsigned long long nn = *nn_io;

    int sp = 0;                 /* seed: EMPTY pattern at the left edge of the first part */
    if (lane == 0) {
        int a = pb[T.pi[si][0]];
        st[0].x0 = 0; st[0].x1 = 0; st[0].x2 = 0;
        st[0].p = pf_pack(a, a, 0, 0, 0, PF_SE, 0);
        sp = 1;
    }
    sp = __shfl_sync(FULL, sp, 0);
    __syncwarp();

    while (sp > 0) {
        int room = cap - sp;
        int n_active = sp < 32 ? sp : 32;
        int r9 = room / 9; if (n_active > r9) n_active = r9;   /* a node has at most 9 children */
        if (n_active < 1) { if (lane == 0) *flagged = 1; *nn_io = nn; return 1; }
        bool active = lane < n_active;

        uint64_t x0 = 0, x1 = 0; uint32_t x2 = 0;
        int lo = 0, hi = 0, mm = 0, go = 0, ge = 0, stt = 0, j = 0;
        if (active) {
            PfNode nd = st[sp - 1 - lane];
            x0 = nd.x0; x1 = nd.x1; x2 = nd.x2; uint32_t q = nd.p;
            lo = q & 0x7f; hi = (q >> 7) & 0x7f; mm = (q >> 14) & 0x1f; go = (q >> 19) & 0x7;
            ge = (q >> 22) & 0xf; stt = (q >> 26) & 0x3; j = (q >> 28) & 0x7;
        }
        sp -= n_active; nn += n_active;
        if (nn > budget) { if (lane == 0) *flagged = 1; *nn_io = nn; return 1; }

        /* direction: pi is contiguous, so the target part is adjacent to the matched window */
        bool done = false, gen = false; int is_back = 0, pos = 0, tgt_lo = 0, tgt_hi = 0, cap_e = 0, lob = 0;
        if (active) {
            if (j >= P) done = true;
            else {
                int pp = T.pi[si][j]; tgt_lo = pb[pp]; tgt_hi = pb[pp + 1];
                cap_e = T.U[si][j]; lob = T.L[si][j];
                if (tgt_hi > hi)      { is_back = 0; pos = hi; }
                else if (tgt_lo < lo) { is_back = 1; pos = lo - 1; }
                else                  { done = true; }        /* zero-length part */
                gen = !done && pos >= 0 && pos < len;
            }
        }
        if (__any_sync(FULL, done)) { *nn_io = nn; return 1; }

        /* one bidirectional extension (bwt_extend): 8 counts + the 4 opposite-direction starts */
        uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0, u0 = 0, u1 = 0, u2 = 0, u3 = 0, r0 = 0, r1 = 0, r2 = 0, r3 = 0;
        if (gen && stt != PF_SE) {
            uint64_t drive = is_back ? x0 : x1;
            pf_2occ4(bwt, primary, drive - 1, drive - 1 + x2, t0, t1, t2, t3, u0, u1, u2, u3);
            uint64_t base = (is_back ? x1 : x0) + (drive <= primary && drive + x2 - 1 >= primary);
            r3 = base;
            r2 = r3 + (u3 - t3);
            r1 = r2 + (u2 - t2);
            r0 = r1 + (u1 - t1);
        }
        #define PF_K(b) ((b) == 0 ? t0 : (b) == 1 ? t1 : (b) == 2 ? t2 : t3)
        #define PF_L(b) ((b) == 0 ? u0 : (b) == 1 ? u1 : (b) == 2 ? u2 : u3)
        #define PF_R(b) ((b) == 0 ? r0 : (b) == 1 ? r1 : (b) == 2 ? r2 : r3)

        /* PF_CH(EMIT) enumerates bwt_match_gap's children under the search's bounds. M and I consume a
         * read base (the window grows); D consumes a reference base only. */
        #define PF_CH(EMIT) do { \
            int rb = seq[pos]; \
            for (int b = 0; b < 4; ++b) {   /* match / mismatch (and, from EMPTY, the seed) */ \
                int nmm = mm + (b != rb || rb > 3); \
                int ne = nmm + go + ge; \
                if (ne > cap_e) continue; \
                uint64_t cx0, cx1; uint32_t cx2; \
                if (stt == PF_SE) { \
                    cx0 = pf_L2[b] + 1; cx1 = pf_L2[3 - b] + 1; \
                    cx2 = (uint32_t)(pf_L2[b + 1] - pf_L2[b]); \
                } else { \
                    int ix = is_back ? b : 3 - b; \
                    uint32_t sz = (uint32_t)(PF_L(ix) - PF_K(ix)); \
                    if (sz == 0) continue; \
                    uint64_t fwd = pf_L2[ix] + 1 + PF_K(ix); \
                    cx0 = is_back ? fwd : PF_R(ix); cx1 = is_back ? PF_R(ix) : fwd; cx2 = sz; \
                } \
                if (cx2 == 0) continue; \
                int nlo = is_back ? lo - 1 : lo, nhi = is_back ? hi : hi + 1, nj = j; \
                if ((is_back && nlo == tgt_lo) || (!is_back && nhi == tgt_hi)) { \
                    if (ne < lob) continue; nj = j + 1; } \
                EMIT(cx0, cx1, cx2, nlo, nhi, nmm, go, ge, PF_SM, nj); \
            } \
            if (stt == PF_SM || stt == PF_SE) {   /* insertion: read base consumed, reference not */ \
                if (go < max_gapo) { int ne = mm + go + 1 + ge; \
                    if (ne <= cap_e) { int nlo = is_back ? lo - 1 : lo, nhi = is_back ? hi : hi + 1, nj = j; \
                        int okp = 1; \
                        if ((is_back && nlo == tgt_lo) || (!is_back && nhi == tgt_hi)) { \
                            if (ne < lob) okp = 0; else nj = j + 1; } \
                        if (okp) EMIT(x0, x1, x2, nlo, nhi, mm, go + 1, ge, \
                                      stt == PF_SE ? PF_SE : PF_SI, nj); } } \
            } else if (stt == PF_SI) { \
                if (ge < max_gape) { int ne = mm + go + ge + 1; \
                    if (ne <= cap_e) { int nlo = is_back ? lo - 1 : lo, nhi = is_back ? hi : hi + 1, nj = j; \
                        int okp = 1; \
                        if ((is_back && nlo == tgt_lo) || (!is_back && nhi == tgt_hi)) { \
                            if (ne < lob) okp = 0; else nj = j + 1; } \
                        if (okp) EMIT(x0, x1, x2, nlo, nhi, mm, go, ge + 1, PF_SI, nj); } } \
            } \
            if (stt != PF_SE) {   /* deletion: reference base consumed, read not (window unchanged) */ \
                int opening = (stt == PF_SM); \
                int allow = opening ? (go < max_gapo) : (stt == PF_SD && ge < max_gape); \
                if (allow) { \
                    int ngo = go + (opening ? 1 : 0), nge = ge + (opening ? 0 : 1); \
                    int ne = mm + ngo + nge; \
                    if (ne <= cap_e) for (int b = 0; b < 4; ++b) { \
                        int ix = is_back ? b : 3 - b; \
                        uint32_t sz = (uint32_t)(PF_L(ix) - PF_K(ix)); \
                        if (sz == 0) continue; \
                        uint64_t fwd = pf_L2[ix] + 1 + PF_K(ix); \
                        uint64_t cx0 = is_back ? fwd : PF_R(ix), cx1 = is_back ? PF_R(ix) : fwd; \
                        EMIT(cx0, cx1, sz, lo, hi, mm, ngo, nge, PF_SD, j); \
                    } \
                } \
            } \
        } while (0)

        int nc = 0;
        #define PF_CNT(a, b, c, d, e, f, g, h, i, k) (void)(++nc)
        if (gen) PF_CH(PF_CNT);
        #undef PF_CNT

        int incl = nc;   /* warp prefix sum of child counts -> each lane's write offset */
        for (int d = 1; d < 32; d <<= 1) { int y = __shfl_up_sync(FULL, incl, d); if (lane >= d) incl += y; }
        int total = __shfl_sync(FULL, incl, 31);
        if (sp + total > cap) { if (lane == 0) *flagged = 1; *nn_io = nn; return 1; }
        int wp = sp + incl - nc;

        #define PF_PUT(cx0, cx1, cx2, nlo, nhi, nmm, ngo, nge, nst, nj) do { \
            st[wp].x0 = (cx0); st[wp].x1 = (cx1); st[wp].x2 = (cx2); \
            st[wp].p = pf_pack((nlo), (nhi), (nmm), (ngo), (nge), (nst), (nj)); ++wp; } while (0)
        if (gen) PF_CH(PF_PUT);
        #undef PF_PUT
        #undef PF_CH
        #undef PF_K
        #undef PF_L
        #undef PF_R

        sp += total;
        __syncwarp();
    }
    *nn_io = nn;
    return 0;
}

/* all searches of the read's scheme; a hit in any of them ends the read */
__device__ inline int pf_has_hit(const uint32_t *__restrict__ bwt, uint64_t primary, const uint8_t *seq, int len,
                                 int K, int max_gapo, int max_gape, PfNode *st, int cap, unsigned long long budget,
                                 int *flagged)
{
    int nN = 0;
    for (int j = 0; j < len; ++j) if (seq[j] > 3) ++nN;
    if (nN > K) return 0;      /* every N costs a difference: more Ns than max_diff cannot align */
    const PfTab &T = pf_sch[K];
    int pb[PF_MAXP + 1];
    for (int t = 0; t <= T.P; ++t) pb[t] = (int)(((long long)len * t) / T.P);
    unsigned long long nn = 0;
    for (int si = 0; si < T.ns; ++si) {
        int fl = 0;
        int h = pf_search(bwt, primary, seq, len, K, si, pb, max_gapo, max_gape, st, cap, budget, &fl, &nn);
        fl = __shfl_sync(0xffffffffu, fl, 0);
        if (fl) { *flagged = 1; return 1; }
        if (h) return 1;
    }
    return 0;
}

/* One thread per read, before the scheme search: a read that occurs exactly certainly has a hit (bwa
 * reports its zero-difference alignment), so it is PF_MAYBE without a scheme search -- in endogenous
 * libraries that is most mapped reads, which would otherwise pay for the filter and gain nothing.
 * Reads the scheme search cannot answer are PF_MAYBE here as well; all others are PF_TODO. This
 * kernel never answers PF_NONE, so it cannot make the output wrong, only the filter cheaper. */
__global__ void pf_exact_kernel(const uint32_t *bwt, uint64_t primary, const uint8_t *query, const uint64_t *qoff,
                                const int *qlen, const int *maxdiff, const int *list, int n, int seed_len, uint8_t *verdict)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= n) return;
    int r = list[pos], len = qlen[r], K = maxdiff[r];
    if (K < 1 || K > PF_MAXK || len > PF_MAXLEN || len < 1 || len > seed_len) { verdict[pos] = PF_MAYBE; return; }
    const uint8_t *q = query + qoff[r];
    /* backward search of the query, as bwt_match_gap consumes it (from its last base) */
    int c = q[len - 1];
    uint8_t v = PF_EXACT;
    if (c > 3) v = PF_TODO;
    else {
        uint64_t k = pf_L2[c] + 1, l = pf_L2[c + 1];
        for (int i = len - 2; i >= 0; --i) {
            c = q[i];
            if (c > 3) { v = PF_TODO; break; }
            uint64_t a0, a1, a2, a3, b0, b1, b2, b3;
            pf_2occ4(bwt, primary, k - 1, l, a0, a1, a2, a3, b0, b1, b2, b3);
            k = pf_L2[c] + (c == 0 ? a0 : c == 1 ? a1 : c == 2 ? a2 : a3) + 1;
            l = pf_L2[c] + (c == 0 ? b0 : c == 1 ? b1 : c == 2 ? b2 : b3);
            if (k > l) { v = PF_TODO; break; }
        }
    }
    verdict[pos] = v;
}

/* Persistent kernel, one warp per read. list[0..n) are read numbers; order (may be NULL) is the
 * sequence in which list positions are taken (longest reads first balances the tail). verdict[pos]
 * gets PF_NONE or PF_MAYBE for list position pos. */
__global__ void pf_kernel(const uint32_t *bwt, uint64_t primary, const uint8_t *query, const uint64_t *qoff,
                          const int *qlen, const int *maxdiff, const int *bid, const bwtgap_opt_t *bopts,
                          const int *list, const int *order, int n, int *workctr, uint8_t *verdict, int cap)
{
    extern __shared__ unsigned char pf_smem[];
    int wib = threadIdx.x >> 5, lane = threadIdx.x & 31;
    PfNode *st = (PfNode *)pf_smem + (size_t)wib * cap;
    for (;;) {
        int idx = 0;
        if (lane == 0) idx = atomicAdd(workctr, 1);
        idx = __shfl_sync(0xffffffffu, idx, 0);
        if (idx >= n) break;
        int pos = order ? order[idx] : idx;
        int r = list[pos], len = qlen[r], K = maxdiff[r];
        uint8_t v = PF_MAYBE;
        if (K >= 1 && K <= PF_MAXK && len <= PF_MAXLEN) {
            const bwtgap_opt_t o = bopts[bid[r]];
            int flagged = 0;
            int hh = pf_has_hit(bwt, primary, query + qoff[r], len, K, o.max_gapo, o.max_gape, st, cap, PF_BUDGET, &flagged);
            v = hh ? PF_MAYBE : PF_NONE;
        }
        if (lane == 0) verdict[pos] = v;
    }
}

/* Run the filter over list[0..n) on the current device (pf_init_device done, device arrays as in
 * light_args). verdict (host, n entries) receives one answer per list position; *n_exact (may be
 * NULL) the number of reads found to occur exactly. Returns the kernel time in seconds, or -1 if
 * the device cannot run it (then every verdict is PF_MAYBE). */
static double pf_run(cudaStream_t s, const uint32_t *d_bwt, uint64_t primary, const uint8_t *d_q, const uint64_t *d_off,
                     const int *d_len, const int *d_maxdiff, const int *d_bid, const bwtgap_opt_t *d_bopts,
                     const int *h_list, const int *h_len, int n, int seed_len, uint8_t *verdict, long *n_exact)
{
    for (int i = 0; i < n; ++i) verdict[i] = PF_MAYBE;
    if (n_exact) *n_exact = 0;
    if (n <= 0) return 0;
    int dev = 0, nsm = 0, optin = 0, bps = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return -1;
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    int cap = PF_CAP;
    while (cap > 256 && (size_t)PF_WPB * cap * sizeof(PfNode) > (size_t)optin) cap -= 128;
    size_t smem = (size_t)PF_WPB * cap * sizeof(PfNode);
    if (smem > (size_t)optin) return -1;
    if (cudaFuncSetAttribute(pf_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) != cudaSuccess
        || cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, pf_kernel, PF_WPB * 32, smem) != cudaSuccess || bps < 1) {
        cudaGetLastError();
        return -1;
    }
    int *d_list, *d_order, *d_ctr; uint8_t *d_v;
    if (cudaMalloc(&d_list, (size_t)n * sizeof(int)) != cudaSuccess) { cudaGetLastError(); return -1; }
    if (cudaMalloc(&d_order, (size_t)n * sizeof(int)) != cudaSuccess) { cudaGetLastError(); cudaFree(d_list); return -1; }
    if (cudaMalloc(&d_ctr, sizeof(int)) != cudaSuccess) { cudaGetLastError(); cudaFree(d_list); cudaFree(d_order); return -1; }
    if (cudaMalloc(&d_v, (size_t)n) != cudaSuccess) { cudaGetLastError(); cudaFree(d_list); cudaFree(d_order); cudaFree(d_ctr); return -1; }
    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);
    float ms1 = 0, ms2 = 0;
    std::vector<int> order;
    /* 1: exact occurrence, one thread per read. Every answer starts as PF_MAYBE, so a kernel that
     * does not run can never leave a read marked as having no alignment. */
    cudaMemcpyAsync(d_list, h_list, (size_t)n * sizeof(int), cudaMemcpyHostToDevice, s);
    cudaMemsetAsync(d_v, PF_MAYBE, (size_t)n, s);
    cudaEventRecord(e0, s);
    pf_exact_kernel<<<(unsigned)((n + 127) / 128), 128, 0, s>>>(d_bwt, primary, d_q, d_off, d_len, d_maxdiff, d_list, n, seed_len, d_v);
    cudaError_t err = cudaGetLastError();   /* launch errors are not reported by the synchronisation */
    cudaEventRecord(e1, s);
    cudaMemcpyAsync(verdict, d_v, (size_t)n, cudaMemcpyDeviceToHost, s);
    cudaError_t serr = cudaStreamSynchronize(s);
    if (err == cudaSuccess) err = serr;
    if (err == cudaSuccess) {
        cudaEventElapsedTime(&ms1, e0, e1);
        /* 2: scheme search of the rest, longest reads first (max_diff, hence the tree size, is a step
         * function of read length) */
        for (int i = 0; i < n; ++i) if (verdict[i] == PF_TODO) order.push_back(i);
        if (n_exact) { long ne = 0; for (int i = 0; i < n; ++i) ne += verdict[i] == PF_EXACT; *n_exact = ne; }
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return h_len[h_list[a]] > h_len[h_list[b]]; });
        int m = (int)order.size();
        if (m > 0) {
            cudaMemcpyAsync(d_order, order.data(), (size_t)m * sizeof(int), cudaMemcpyHostToDevice, s);
            cudaMemsetAsync(d_ctr, 0, sizeof(int), s);
            cudaEventRecord(e2, s);
            pf_kernel<<<(unsigned)(bps * nsm), PF_WPB * 32, smem, s>>>(d_bwt, primary, d_q, d_off, d_len, d_maxdiff, d_bid, d_bopts,
                                                                      d_list, d_order, m, d_ctr, d_v, cap);
            err = cudaGetLastError();
            cudaEventRecord(e3, s);
            cudaMemcpyAsync(verdict, d_v, (size_t)n, cudaMemcpyDeviceToHost, s);
            serr = cudaStreamSynchronize(s);
            if (err == cudaSuccess) err = serr;
            if (err == cudaSuccess) cudaEventElapsedTime(&ms2, e2, e3);
        }
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    cudaFree(d_list); cudaFree(d_order); cudaFree(d_ctr); cudaFree(d_v);
    if (err != cudaSuccess) {   /* never trust a partial answer */
        fprintf(stderr, "[galn] prefilter: %s; searching every read\n", cudaGetErrorString(err));
        for (int i = 0; i < n; ++i) verdict[i] = PF_MAYBE;
        return -1;
    }
    for (int i = 0; i < n; ++i) if (verdict[i] != PF_NONE) verdict[i] = PF_MAYBE;
    return (ms1 + ms2) / 1e3;
}

#endif /* PREFILTER_CUH */
