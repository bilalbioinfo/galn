/* gapw.cuh — warp-per-read GPU engine for heavy reads (device only).
 *
 * One warp searches one read with exactly the sequential algorithm of
 * bwt_match_gap (bwtgap_hd.h): same pops, same pushes, same order, same
 * cut-offs. All 32 lanes run the scalar search redundantly in lock-step, so
 * every branch is warp-uniform and every lane reads back its own (identical)
 * writes; the lanes split only the memory-heavy work:
 *
 *   occurrence counts   lanes 0-15 load the 64-byte BWT block of k-1, lanes
 *                       16-31 the block of l, in one round trip, and count
 *                       bases with popcount masks (the same integers as
 *                       bwt_occ4 / bwt_2occ4 / bwt_2occ).
 *   chunk bookkeeping   lane-parallel copies of chunk ids under a lock.
 *
 * The priority stack: each score bucket is a LIFO of 64-entry chunks (16-byte
 * packed entries, no per-entry link; the score is the bucket, so it is not stored), so a pop reads the top slot of the lowest
 * non-empty bucket and a push appends to its bucket, i.e. upstream's order.
 * Bucket tops, the non-empty bitmask and width[] / seed_width[] live in shared
 * memory. Chunks come from the warp's own arena or, in shared-pool mode, from one
 * device-wide pool (a warp takes 32 ids at a time and returns every chunk it used
 * when the read ends). If no chunk is left the read is abandoned and re-run on
 * the CPU (never waits, so no deadlock).
 *
 * Reads the engine does not take (length > GAPW_MAXLEN, n_stacks >
 * GAPW_MAXSTACK) and reads that run out of stack or alignment space return an
 * error code and are re-run on the CPU from scratch.
 */
#ifndef GAPW_CUH
#define GAPW_CUH

#include <stdint.h>
#include "bwtgap_hd.h"

#define GAPW_CHUNK      64
#define GAPW_MAXSTACK   256
#define GAPW_MASKW      (GAPW_MAXSTACK / 32)
#define GAPW_MAXLEN     256
#define GAPW_WOFF       (GAPW_MAXLEN + 1)
#define GAPW_FULL       0xffffffffu
#define GAPW_ERR_UNSUP  (-5)    /* read not taken by the warp engine */
#define GAPW_ERR_CANCEL (-6)    /* abandoned: the host finished this read already */
#define GAPW_FAIL_CPU   0x40000000   /* flag in fail[]: the read cannot run in the warp engine at all */
#define GAPW_ERR_HANDOFF (-7)   /* handed to the CPU: stack past the hand-off size while the CPU has room */

/* Stack entry as stored: 16 bytes, two words (upstream gap_entry_t is 32 bytes).
 *   w0 = k | i<<38 | last_diff_pos<<47 | n_mm<<56 | state<<62
 *   w1 = l | n_gapo<<38 | n_gape<<44 | n_ins<<50 | n_del<<57
 * The score is the bucket index. A read whose values may not fit (a text of 2^38 or more,
 * about 2.7e11 symbols; n_mm/n_gapo/n_gape > 63, hence n_ins/n_del > 126) is given up with
 * GAPW_ERR_UNSUP, so the CPU searches it instead and the output stays exact. */
#define GAPW_KBITS 38
#define GAPW_KMASK ((1ull << GAPW_KBITS) - 1)
struct __align__(16) gapw_entry {
    uint64_t w0, w1;
};

/* the same entry unpacked, as the search uses it (field meaning as in upstream gap_entry_t) */
struct gapw_eview {
    uint64_t k, l;
    uint32_t info;                  /* score<<21 | i */
    int n_ins, n_del;
    int last_diff_pos;              /* <= len <= GAPW_MAXLEN */
    int n_mm, n_gapo, n_gape, state;
};

/* per-warp shared memory */
struct gapw_smem {
    int      top_chunk[GAPW_MAXSTACK];   /* top chunk of each bucket, -1 = empty */
    uint16_t top_cnt[GAPW_MAXSTACK];     /* entries in the top chunk */
    uint32_t mask[GAPW_MASKW];           /* bit b set <=> bucket b non-empty */
    int      cache[32];                  /* free chunk ids held by this warp */
    uint64_t w[2 * GAPW_WOFF];           /* width[].w, then seed_width[].w */
    int      bid[2 * GAPW_WOFF];         /* width[].bid, then seed_width[].bid */
    bwtgap_opt_t opt;                    /* this read's options (batch options, per-read max_diff/seed_len) */
};

/* device-wide control block */
struct gapw_ctl {
    int lock;       /* protects gfree/top */
    int top;        /* ids in gfree */
    int bump;       /* next never-used chunk id */
    int qnext;      /* next list position to claim */
    unsigned long long out_top;   /* alignment records reserved in the output arena (64-bit: never wraps) */
};

struct gapw_args {
    const uint32_t *bwt;
    const uint8_t  *query;
    const uint64_t *qoff;
    const int      *qlen;
    const int      *maxdiff;
    const bwtgap_opt_t *bopts;      /* per upstream batch */
    const int      *nstacks;        /* per upstream batch */
    const int      *bid;            /* per read: upstream batch index */
    int seed_len;
    /* work: reads list[0..list_n). Static mode: warps claim positions 0,1,2,... through
     * ctl->qnext. Dynamic mode (qctl != NULL): list and qctl live in host-mapped memory and
     * the 64-bit word qctl = front<<32 | back is shared with the CPU workers, who claim from
     * the front while the warps claim from the back, each with a compare-and-swap. */
    const int *list;
    int list_n;
    unsigned long long *qctl;
    /* device mode (portable, used by galn): dback counts down from list_n in device memory;
     * the host keeps *dfront (device memory, updated with async copies) at the number of
     * positions the CPU has claimed from the front; a warp stops when it reaches them */
    int *dback;
    const int *dfront;
    /* optional per-position flags (device memory, refreshed by the host): when set, the
     * warp working on that position abandons it (checked every 1024 pops) */
    const uint8_t *cancel;
    /* optional hand-off: *handoff is a budget (device word the host rewrites every poll from
     * the length of the CPU queues). A read whose stack holds more than handoff_entries entries
     * takes one unit and is given up (published like any give-up) so a CPU core, much faster
     * per read, takes it; when the CPU is saturated the budget is 0 and the GPU keeps every read */
    int *handoff;
    int handoff_entries;
    /* reads the warps give up on, published as they happen so the CPU starts on them at once
     * (the host copies them out while the kernel runs): fail[] starts at -1, fail_n counts reserved slots */
    int *fail;
    unsigned int *fail_n;
    /* stack memory */
    gapw_entry *chunks;             /* n_chunks * GAPW_CHUNK */
    int *chunk_prev;                /* n_chunks */
    int *gfree;                     /* n_chunks */
    int n_chunks;
    int *owned;                     /* owned_cap per warp (shared-pool mode) */
    int owned_cap;
    int arena_chunks;               /* > 0: arena mode, warp w owns chunks [w*arena_chunks, (w+1)*arena_chunks) */
    gapw_ctl *ctl;
    /* per-warp alignment scratch */
    bwtgap_aln_t *aln_scratch;
    int aln_cap;
    /* results, per list position */
    int *out_n;                     /* n_aln or error code */
    int *out_off;                   /* offset into out_rec */
    unsigned long long *out_pops, *out_trace;
    bwtgap_aln_t *out_rec;
    int out_cap;
};

/* ---------------------------------------------------------------------- */

__device__ __forceinline__ int gapw_lane() { return threadIdx.x & 31; }

/* lane 0 only: claim the next list position, or -1 when the list is exhausted */
__device__ int gapw_claim(const gapw_args &a)
{
    if (a.dback) {
        int p = atomicSub(a.dback, 1) - 1;
        if (p < 0 || p < *(volatile const int*)a.dfront) return -1;
        return p;
    }
    if (!a.qctl) {
        int p = atomicAdd(&a.ctl->qnext, 1);
        return p < a.list_n? p : -1;
    }
    unsigned long long old = *(volatile unsigned long long*)a.qctl;
    for (;;) {
        unsigned front = (unsigned)(old >> 32), back = (unsigned)old;
        if (front >= back) return -1;
        unsigned long long nv = ((unsigned long long)front << 32) | (back - 1);
        unsigned long long seen = atomicCAS_system(a.qctl, old, nv);
        if (seen == old) return (int)(back - 1);
        old = seen;
    }
}

/* occurrence counts of all four bases for positions k and l (either may be (uint64)-1).
 * Lanes 0-15 work on k, lanes 16-31 on l; every lane gets all eight results. */
__device__ __forceinline__ void gapw_2occ4(const uint32_t *__restrict__ bwt, const bwtgpu_meta_t *m,
                                           uint64_t k, uint64_t l, uint64_t ck[4], uint64_t cl[4])
{
    int lane = gapw_lane(), half = lane >> 4, h = lane & 15;
    uint64_t p = half? l : k;
    bool none = p == (uint64_t)(-1);
    uint64_t pp = none? 0 : p - (p >= m->primary);
    uint64_t widx = ((pp >> 7) << 4) + h;
    uint32_t w = (!none && widx < m->bwt_size)? __ldg(bwt + widx) : 0;
    uint32_t packed = 0;
    if (h >= 8 && !none) {
        int off = (int)(pp & 127), wi = h - 8, pw = off >> 4;
        if (wi <= pw) {
            uint32_t valid = (wi < pw)? 0xffffffffu : ~((1u << ((~off & 15) << 1)) - 1);
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                uint32_t x = w ^ (0x55555555u * (uint32_t)c);
                uint32_t y = ~(x | (x >> 1)) & 0x55555555u & valid;
                packed |= (uint32_t)__popc(y) << (8 * c);
            }
        }
    }
    /* sum the per-word counts over the 16 lanes of each half (lanes 0-7 contribute 0) */
    packed += __shfl_xor_sync(GAPW_FULL, packed, 1);
    packed += __shfl_xor_sync(GAPW_FULL, packed, 2);
    packed += __shfl_xor_sync(GAPW_FULL, packed, 4);
    packed += __shfl_xor_sync(GAPW_FULL, packed, 8);
    /* cnt[c] is the uint64 at words 2c, 2c+1 of the block */
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        uint32_t lo = __shfl_sync(GAPW_FULL, w, (half << 4) + 2 * c);
        uint32_t hi = __shfl_sync(GAPW_FULL, w, (half << 4) + 2 * c + 1);
        uint64_t v = none? 0 : (((uint64_t)hi << 32 | lo) + ((packed >> (8 * c)) & 0xff));
        ck[c] = __shfl_sync(GAPW_FULL, v, 0);
        cl[c] = __shfl_sync(GAPW_FULL, v, 16);
    }
}


/* warp sum of a 32-bit value (every lane gets the sum) */
__device__ __forceinline__ uint32_t gapw_wsum(uint32_t v)
{
#if __CUDA_ARCH__ >= 800 && !defined(GAPW_NO_REDUX)
    return __reduce_add_sync(GAPW_FULL, v);
#else
    v += __shfl_xor_sync(GAPW_FULL, v, 1);
    v += __shfl_xor_sync(GAPW_FULL, v, 2);
    v += __shfl_xor_sync(GAPW_FULL, v, 4);
    v += __shfl_xor_sync(GAPW_FULL, v, 8);
    v += __shfl_xor_sync(GAPW_FULL, v, 16);
    return v;
#endif
}

/* bases in data word `wi` (0..7) of the block of pp that count for Occ(pp): the same mask as
 * gapw_2occ4 / bwt_occ4 (bases at offsets <= pp & 127 of the block) */
__device__ __forceinline__ uint32_t gapw_valid(uint64_t pp, int wi)
{
    int off = (int)(pp & 127), pw = off >> 4;
    return wi < pw? 0xffffffffu : (wi == pw? ~((1u << ((~off & 15) << 1)) - 1) : 0u);
}
__device__ __forceinline__ uint32_t gapw_count(uint32_t w, uint32_t valid, int c)
{
    uint32_t x = w ^ (0x55555555u * (uint32_t)c);
    return (uint32_t)__popc(~(x | (x >> 1)) & 0x55555555u & valid);
}
/* uint64 count of base c stored at the head of the block of pp (0 for none / past the end) */
__device__ __forceinline__ uint64_t gapw_blkcnt(const uint32_t *__restrict__ bwt, const bwtgpu_meta_t *m,
                                                bool none, uint64_t pp, int c)
{
    return none? 0 : __ldg((const unsigned long long*)(bwt + ((pp >> 7) << 4)) + c);
}

/* One base: the SA interval of base c (uniform) prepended to (k,l):
 *   *nk = L2[c] + Occ(c, k-1) + 1,  *nl = L2[c] + Occ(c, l)   (the same integers as bwt_2occ)
 * Lanes 0-7 count the data words of the block of k-1, lanes 8-15 those of the block of l (one warp sum);
 * lanes 0 and 8 load the stored counts of the two blocks. Every lane gets the result.
 * The device copy of the .bwt is followed by GAPW_BWT_PAD zero bytes, so words past bwt_size read as 0,
 * which is what the bounds test of gapw_2occ4 substituted. */
#define GAPW_BWT_PAD 256
__device__ __forceinline__ void gapw_occ1(const uint32_t *__restrict__ bwt, const bwtgpu_meta_t *m,
                                          uint64_t k1, uint64_t l, int c, uint64_t *nk, uint64_t *nl)
{
    int lane = gapw_lane(), sel = (lane >> 3) & 1, wi = lane & 7;
    uint64_t p = sel? l : k1;
    bool none = p == (uint64_t)(-1);
    uint64_t pp = p - (p >= m->primary);
    const uint32_t *blk = bwt + ((pp >> 7) << 4);
    uint32_t v = 0;
    uint64_t cw = 0;
    if (lane < 16 && !none) {
        uint32_t valid = gapw_valid(pp, wi);
        if (valid) v = gapw_count(__ldg(blk + 8 + wi), valid, c) << (16 * sel);
        if (wi == 0) cw = __ldg((const unsigned long long*)blk + c);
    }
    uint32_t r = gapw_wsum(v);
    uint64_t ck = __shfl_sync(GAPW_FULL, cw, 0), cl = __shfl_sync(GAPW_FULL, cw, 8);
    uint64_t L2c = m->L2[c];
    *nk = L2c + ck + (r & 0xffff) + 1;
    *nl = L2c + cl + (r >> 16);
}

/* All four bases, each lane for its own base c = lane & 3:
 *   *kc = L2[c] + Occ(c, k-1) + 1,  *lc = L2[c] + Occ(c, l)
 * Lane roles: sel = (lane>>3)&1 picks the block of k-1 or of l, wi = lane&7 its data word,
 * bp = lane>>4 the base pair {0,1} or {2,3}; byte b of the two warp sums is the in-block count of
 * base b (at most 128, so bytes cannot carry). A consumer takes base c from lane c (or c+4, ...). */
__device__ __forceinline__ void gapw_occ4own(const uint32_t *__restrict__ bwt, const bwtgpu_meta_t *m,
                                             uint64_t k1, uint64_t l, uint64_t L2c, uint64_t *kc, uint64_t *lc)
{
    int lane = gapw_lane(), sel = (lane >> 3) & 1, wi = lane & 7, bp = lane >> 4, c = lane & 3;
    bool nonek = k1 == (uint64_t)(-1), nonel = l == (uint64_t)(-1);
    uint64_t ppk = nonek? 0 : k1 - (k1 >= m->primary), ppl = nonel? 0 : l - (l >= m->primary);
    uint64_t pp = sel? ppl : ppk;
    bool none = sel? nonel : nonek;
    uint32_t v = 0;
    if (!none) {
        uint32_t valid = gapw_valid(pp, wi);
        if (valid) {
            uint32_t w = __ldg(bwt + ((pp >> 7) << 4) + 8 + wi);
            v = (gapw_count(w, valid, 2 * bp) | gapw_count(w, valid, 2 * bp + 1) << 8) << (16 * bp);
        }
    }
    uint32_t rk = gapw_wsum(sel? 0u : v), rl = gapw_wsum(sel? v : 0u);
    *kc = L2c + gapw_blkcnt(bwt, m, nonek, ppk, c) + ((rk >> (8 * c)) & 0xff) + 1;
    *lc = L2c + gapw_blkcnt(bwt, m, nonel, ppl, c) + ((rl >> (8 * c)) & 0xff);
}

/* ---- chunk allocator ---------------------------------------------------- */

struct gapw_warpstate {
    int ncache;     /* ids in S->cache */
    int lfree;      /* local free list (linked through chunk_prev), -1 = empty */
    int n_owned;    /* chunks taken from the device pool for this read */
    int *owned;
    int base, bump; /* arena mode: first chunk of this warp's arena, chunks handed out so far */
};

__device__ __forceinline__ void gapw_lock(int *lk)
{
    while (atomicCAS(lk, 0, 1) != 0) __nanosleep(32);
    __threadfence();
}
__device__ __forceinline__ void gapw_unlock(int *lk)
{
    __threadfence();
    atomicExch(lk, 0);
}

/* uniform call; returns a chunk id, or -1 when the arena or the pool is exhausted */
__device__ int gapw_alloc_chunk(const gapw_args &a, gapw_smem *S, gapw_warpstate &W)
{
    if (W.ncache > 0) { --W.ncache; return S->cache[W.ncache]; }
    if (W.lfree >= 0) { int c = W.lfree; W.lfree = a.chunk_prev[c]; return c; }
    if (a.arena_chunks > 0) return W.bump < a.arena_chunks? W.base + W.bump++ : -1;
    int lane = gapw_lane();
    int got = 0, base = 0;
    if (lane == 0) {
        gapw_lock(&a.ctl->lock);
        int top = atomicAdd(&a.ctl->top, 0);
        got = top < 32? top : 32;
        base = top - got;
        if (got > 0) atomicExch(&a.ctl->top, base);
    }
    got = __shfl_sync(GAPW_FULL, got, 0);
    base = __shfl_sync(GAPW_FULL, base, 0);
    if (got > 0) {
        int id = 0;
        if (lane < got) { id = ((volatile int*)a.gfree)[base + lane]; S->cache[lane] = id; }
        __syncwarp();
        if (lane == 0) gapw_unlock(&a.ctl->lock);
    } else {
        if (lane == 0) {
            gapw_unlock(&a.ctl->lock);
            int b = atomicAdd(&a.ctl->bump, 32);
            got = b >= a.n_chunks? 0 : (a.n_chunks - b < 32? a.n_chunks - b : 32);
            base = b;
        }
        got = __shfl_sync(GAPW_FULL, got, 0);
        base = __shfl_sync(GAPW_FULL, base, 0);
        if (got > 0 && lane < got) S->cache[lane] = base + lane;
        __syncwarp();
    }
    if (got == 0) return -1;
    if (W.n_owned + got > a.owned_cap) {   /* cannot happen with owned_cap >= the per-read bound */
        return -1;
    }
    if (lane < got) W.owned[W.n_owned + lane] = S->cache[lane];
    __syncwarp();
    W.n_owned += got;
    W.ncache = got - 1;
    return S->cache[got - 1];
}

__device__ __forceinline__ void gapw_free_chunk(const gapw_args &a, gapw_smem *S, gapw_warpstate &W, int c)
{
    if (W.ncache < 32) { S->cache[W.ncache] = c; ++W.ncache; }
    else { a.chunk_prev[c] = W.lfree; W.lfree = c; }
}

/* give every chunk this read took back to the device pool (arena mode: reset the arena) */
__device__ void gapw_release_all(const gapw_args &a, gapw_warpstate &W)
{
    int lane = gapw_lane();
    if (a.arena_chunks > 0) { W.bump = 0; W.ncache = 0; W.lfree = -1; return; }
    if (W.n_owned > 0) {
        int top = 0;
        if (lane == 0) { gapw_lock(&a.ctl->lock); top = atomicAdd(&a.ctl->top, 0); }
        top = __shfl_sync(GAPW_FULL, top, 0);
        for (int j = lane; j < W.n_owned; j += 32) a.gfree[top + j] = W.owned[j];
        __syncwarp();
        if (lane == 0) { __threadfence(); atomicExch(&a.ctl->top, top + W.n_owned); gapw_unlock(&a.ctl->lock); }
        __syncwarp();
    }
    W.n_owned = 0; W.ncache = 0; W.lfree = -1;
}

/* ---- search state ------------------------------------------------------- */

struct gapw_ctx {
    const gapw_args *a;
    gapw_smem *S;
    gapw_warpstate W;
    int n_stacks, best, n_entries;
    uint64_t n_pops, trace;
    const uint8_t *cancel_flag;   /* this read's cancel flag, or NULL */
    uint64_t L2own;               /* L2[lane & 3] */
};

/* gap_push: uniform; returns 0 or an error code. `score` = bwtgap_score of the entry, which the caller
 * gets as the parent's score (its bucket) plus the child's penalty (the score is linear in the counts).
 * Every field fits the 16-byte entry: gapw_kernel only starts reads whose options bound them
 * (n_mm <= max_diff+1 <= 63, n_gapo <= max_gapo <= 63, n_gape <= max_gape <= 63, n_ins and n_del <=
 * n_gapo + n_gape <= 126, i <= len <= 256, k and l <= seq_len < 2^38); other reads go to the CPU. */
__device__ __forceinline__ int gapw_push(gapw_ctx &C, int score, int i, uint64_t k, uint64_t l, int n_mm, int n_gapo, int n_gape,
                                         int n_ins, int n_del, int state, int is_diff)
{
    if ((unsigned)score >= (unsigned)C.n_stacks) return BWTGAP_ERR_SCORE;
    gapw_smem *S = C.S;
    int c = S->top_chunk[score], cnt = S->top_cnt[score];
    if (c < 0 || cnt == GAPW_CHUNK) {
        int nc = gapw_alloc_chunk(*C.a, S, C.W);
        if (nc < 0) return BWTGAP_ERR_STACK;
        C.a->chunk_prev[nc] = c;
        S->top_chunk[score] = nc;
        if (c < 0) S->mask[score >> 5] |= 1u << (score & 31);   /* the bucket was empty */
        c = nc; cnt = 0;
    }
    gapw_entry e;
    e.w0 = k | (uint64_t)i << 38 | (uint64_t)(is_diff? i : 0) << 47 | (uint64_t)n_mm << 56 | (uint64_t)state << 62;
    e.w1 = l | (uint64_t)n_gapo << 38 | (uint64_t)n_gape << 44 | (uint64_t)n_ins << 50 | (uint64_t)n_del << 57;
    C.a->chunks[(size_t)c * GAPW_CHUNK + cnt] = e;
    S->top_cnt[score] = (uint16_t)(cnt + 1);
    ++C.n_entries;
    if (C.best > score) C.best = score;
    return 0;
}

/* gap_pop: uniform. TRACE: fold the pop into the per-read trace hash (only --verify reads it) */
template <bool TRACE>
__device__ __forceinline__ void gapw_pop(gapw_ctx &C, gapw_eview &e)
{
    gapw_smem *S = C.S;
    int b = C.best;
    int c = S->top_chunk[b], cnt = S->top_cnt[b] - 1;
    {
        const gapw_entry p = C.a->chunks[(size_t)c * GAPW_CHUNK + cnt];
        e.k = p.w0 & GAPW_KMASK;
        e.l = p.w1 & GAPW_KMASK;
        int i = (int)(p.w0 >> 38 & 511);
        e.info = (uint32_t)b << 21 | (uint32_t)i;      /* the bucket is the score */
        e.last_diff_pos = (int)(p.w0 >> 47 & 511);
        e.n_mm = (int)(p.w0 >> 56 & 63);
        e.state = (int)(p.w0 >> 62 & 3);
        e.n_gapo = (int)(p.w1 >> 38 & 63);
        e.n_gape = (int)(p.w1 >> 44 & 63);
        e.n_ins = (int)(p.w1 >> 50 & 127);
        e.n_del = (int)(p.w1 >> 57 & 127);
    }
    ++C.n_pops;
    bool emptied = false;
    if (cnt == 0) {
        int prev = C.a->chunk_prev[c];
        gapw_free_chunk(*C.a, S, C.W, c);
        S->top_chunk[b] = prev;
        S->top_cnt[b] = prev >= 0? GAPW_CHUNK : 0;
        if (prev < 0) { S->mask[b >> 5] &= ~(1u << (b & 31)); emptied = true; }
    } else S->top_cnt[b] = (uint16_t)cnt;
    --C.n_entries;
    if (TRACE)
        C.trace = bwtgap_trace_pop(C.trace, e.info, e.k, e.l, e.n_mm, e.n_gapo, e.n_gape, e.state,
                                   e.n_ins, e.n_del, e.last_diff_pos, C.n_entries);
    if (emptied && C.n_entries) { /* reset best: smallest non-empty bucket above b */
        int from = b + 1, wi = from >> 5, nw = (C.n_stacks + 31) >> 5;
        uint32_t mk = (wi < nw)? S->mask[wi] & (~0u << (from & 31)) : 0;
        while (!mk && ++wi < nw) mk = S->mask[wi];
        C.best = mk? (wi << 5) + __ffs(mk) - 1 : C.n_stacks;
    } else if (C.n_entries == 0) C.best = C.n_stacks;
}

/* bwt_cal_width into shared memory (w, bid at offset `o`); str is the reversed+complemented read */
__device__ void gapw_cal_width(const uint32_t *bwt, const bwtgpu_meta_t *m, int len, const uint8_t *str,
                               uint64_t *w, int *bid_out)
{
    uint64_t k = 0, l = m->seq_len;
    int bid = 0;
    for (int i = 0; i < len; ++i) {
        uint8_t c = str[i];
        c = c > 3? 4 : 3 - c;              /* undo the complement, as bwtgap_cal_width(comp=1) */
        if (c < 4) gapw_occ1(bwt, m, k - 1, l, c, &k, &l);
        if (k > l || c > 3) { k = 0; l = m->seq_len; ++bid; }
        w[i] = l - k + 1;
        bid_out[i] = bid;
    }
    w[len] = 0;
    bid_out[len] = ++bid;
}

/* bwt_match_exact_alt */
__device__ int gapw_match_exact_alt(const uint32_t *bwt, const bwtgpu_meta_t *m, int len, const uint8_t *str,
                                    uint64_t *k0, uint64_t *l0)
{
    uint64_t k = *k0, l = *l0;
    for (int i = len - 1; i >= 0; --i) {
        uint8_t c = str[i];
        if (c > 3) return 0;
        gapw_occ1(bwt, m, k - 1, l, c, &k, &l);
        if (k > l) return 0;
    }
    *k0 = k; *l0 = l;
    return (int)(l - k + 1);
}

/* gap_shadow on shared-memory width */
__device__ __forceinline__ void gapw_shadow(int x, uint64_t max, int last_diff_pos, uint64_t *w, int *bid)
{
    int j = 0;
    for (int i = 0; i < last_diff_pos; ++i) {
        if (w[i] > (uint64_t)(int64_t)x) w[i] -= x;
        else if (w[i] == (uint64_t)(int64_t)x) { bid[i] = 1; w[i] = max - (++j); }
    }
}

#define GAPW_PUSH(...) do { int _r = gapw_push(C, __VA_ARGS__); if (_r) return _r; } while (0)

/* bwt_match_gap for one read; uniform across the warp. Returns n_aln or an error code. */
template <bool TRACE>
__device__ int gapw_match_gap(gapw_ctx &C, const bwtgpu_meta_t *bm, int len, const uint8_t *__restrict__ seq,
                              bool use_seed, const bwtgap_opt_t *__restrict__ opt, bwtgap_aln_t *aln, int m_aln)
{
    const uint32_t *bwt = C.a->bwt;
    gapw_smem *S = C.S;
    uint64_t *W = S->w, *SW = S->w + GAPW_WOFF;
    int *WB = S->bid, *SWB = S->bid + GAPW_WOFF;
    /* the options as values: one load per read, then registers (through a pointer into shared memory,
     * every use would be reloaded after each stack write) */
    const bwtgap_opt_t O = *opt;
    int best_score = (O.max_diff+1) * O.s_mm + (O.max_gapo+1) * O.s_gapo + (O.max_gape+1) * O.s_gape;
    int best_diff = O.max_diff + 1, max_diff = O.max_diff;
    int best_cnt = 0;
    int j, _j, n_aln = 0;
    (void)best_diff;

    for (j = _j = 0; j < len; ++j)
        if (seq[j] > 3) ++_j;
    if (_j > max_diff) return 0;

    /* reset the stack */
    for (j = gapw_lane(); j < C.n_stacks; j += 32) { S->top_chunk[j] = -1; S->top_cnt[j] = 0; }
    if (gapw_lane() < GAPW_MASKW) S->mask[gapw_lane()] = 0;
    __syncwarp();
    C.best = C.n_stacks; C.n_entries = 0; C.n_pops = 0; C.trace = BWTGAP_TRACE_INIT;
    GAPW_PUSH(0, len, 0, bm->seq_len, 0, 0, 0, 0, 0, 0, 0);

    /* Exact-child fusion. The exact (no-difference) child of an expansion has the parent's score
     * b0 and is pushed last; with non-negative penalties no child scores below b0 and b0 is the
     * lowest non-empty bucket, so upstream pops exactly this child next (LIFO top of bucket b0).
     * It is kept in `e` instead (pend), counted in n_entries for every test at the loop top, and
     * "popped" without touching the stack; C.best needs no change because the stack itself is
     * the same as after upstream's push + pop. */
    const bool fuse = O.s_mm >= 0 && O.s_gapo >= 0 && O.s_gape >= 0;
    bool pend = false;
    gapw_eview e;
#define GAPW_FUSE(K, L) do { e.k = (K); e.l = (L); e.info = (e.info >> 21) << 21 | (uint32_t)i; \
        e.last_diff_pos = 0; e.state = BWTGAP_STATE_M; ++C.n_entries; pend = true; } while (0)

    while (C.n_entries) {
        int i, m, m_seed = 0, hit_found, allow_diff, allow_M, tmp;
        uint64_t k, l, occ;

        if (C.n_entries > O.max_entries) break;
        if ((C.n_pops & 1023) == 0) {
            if (C.cancel_flag && *(volatile const uint8_t*)C.cancel_flag) return GAPW_ERR_CANCEL;
            if (C.a->handoff && C.n_entries > C.a->handoff_entries && *(volatile const int*)C.a->handoff > 0) {
                int took = 0;
                if (gapw_lane() == 0) took = atomicSub(C.a->handoff, 1) > 0;
                if (__shfl_sync(GAPW_FULL, took, 0)) return GAPW_ERR_HANDOFF;
            }
        }
        if (pend) {     /* the fused exact child: upstream's pop of it */
            pend = false;
            ++C.n_pops;
            --C.n_entries;
            if (TRACE)
                C.trace = bwtgap_trace_pop(C.trace, e.info, e.k, e.l, e.n_mm, e.n_gapo, e.n_gape, e.state,
                                           e.n_ins, e.n_del, e.last_diff_pos, C.n_entries);
            if (C.n_entries == 0) C.best = C.n_stacks;
        } else gapw_pop<TRACE>(C, e);
        k = e.k; l = e.l;
        i = e.info & 0xffff;
        if (!(O.mode & BWTGAP_MODE_NONSTOP) && e.info>>21 > (uint32_t)(best_score + O.s_mm)) break;

        m = max_diff - (e.n_mm + e.n_gapo);
        if (O.mode & BWTGAP_MODE_GAPE) m -= e.n_gape;
        if (m < 0) continue;
        if (use_seed) {
            m_seed = O.max_seed_diff - (e.n_mm + e.n_gapo);
            if (O.mode & BWTGAP_MODE_GAPE) m_seed -= e.n_gape;
        }
        if (i > 0 && m < WB[i-1]) continue;

        hit_found = 0;
        if (i == 0) hit_found = 1;
        else if (m == 0 && (e.state == BWTGAP_STATE_M || (O.mode&BWTGAP_MODE_GAPE) || e.n_gape == O.max_gape)) {
            if (gapw_match_exact_alt(bwt, bm, i, seq, &k, &l)) hit_found = 1;
            else continue;
        }

        if (hit_found) {
            int score = bwtgap_score(e.n_mm, e.n_gapo, e.n_gape, opt);
            int do_add = 1;
            if (n_aln == 0) {
                best_score = score;
                best_diff = e.n_mm + e.n_gapo;
                if (O.mode & BWTGAP_MODE_GAPE) best_diff += e.n_gape;
                if (!(O.mode & BWTGAP_MODE_NONSTOP))
                    max_diff = (best_diff + 1 > O.max_diff)? O.max_diff : best_diff + 1;
            }
            if (score == best_score) best_cnt += l - k + 1;
            else if (best_cnt > O.max_top2) break;
            if (e.n_gapo) {
                for (j = 0; j != n_aln; ++j)
                    if (aln[j].k == k && aln[j].l == l) break;
                if (j < n_aln) do_add = 0;
            }
            if (do_add) {
                gapw_shadow(l - k + 1, bm->seq_len, e.last_diff_pos, W, WB);
                if (n_aln == m_aln) return BWTGAP_ERR_ALN;
                bwtgap_aln_t rec;
                rec.packed = bwtgap_pack(e.n_mm, e.n_gapo, e.n_gape, score, e.n_ins, e.n_del);
                rec.k = k; rec.l = l;
                aln[n_aln] = rec;
                ++n_aln;
            }
            continue;
        }

        --i;
        const int s = seq[i];
        occ = l - k + 1;
        allow_diff = allow_M = 1;
        if (i > 0) {
            int ii = i - (len - O.seed_len);
            if (WB[i-1] > m-1) allow_diff = 0;
            else if (WB[i-1] == m-1 && WB[i] == m-1 && W[i-1] == W[i]) allow_M = 0;
            if (use_seed && ii > 0) {
                if (SWB[ii-1] > m_seed-1) allow_diff = 0;
                else if (SWB[ii-1] == m_seed-1 && SWB[ii] == m_seed-1 && SW[ii-1] == SW[ii]) allow_M = 0;
            }
        }
        tmp = (O.mode & BWTGAP_MODE_LOGGAP)? bwtgap_int_log2(e.n_gape + e.n_gapo)/2+1 : e.n_gapo + e.n_gape;
        /* child scores: the parent's bucket plus the penalty of the child's new difference */
        const int b0 = (int)(e.info >> 21), s_o = b0 + O.s_gapo, s_e = b0 + O.s_gape, s_m = b0 + O.s_mm;
        const bool indel = allow_diff && i >= O.indel_end_skip + tmp && len - i >= O.indel_end_skip + tmp;
        /* deletion children (they need all four bases) exactly when upstream would push them */
        const bool delc = indel && ((e.state == BWTGAP_STATE_M && e.n_gapo < O.max_gapo)
            || (e.state == BWTGAP_STATE_D && e.n_gape < O.max_gape
                && (e.n_gape + e.n_gapo < max_diff || occ < (uint64_t)(int64_t)O.max_del_occ)));
        if (delc || (allow_diff && allow_M)) {
            /* lane c holds the child interval of base c */
            uint64_t kc, lc;
            gapw_occ4own(bwt, bm, k - 1, l, C.L2own, &kc, &lc);
            if (indel) {
                if (e.state == BWTGAP_STATE_M) {
                    if (e.n_gapo < O.max_gapo) {
                        GAPW_PUSH(s_o, i, k, l, e.n_mm, e.n_gapo + 1, e.n_gape, e.n_ins + 1, e.n_del, BWTGAP_STATE_I, 1);
                        for (j = 0; j != 4; ++j) {
                            k = __shfl_sync(GAPW_FULL, kc, j);
                            l = __shfl_sync(GAPW_FULL, lc, j);
                            if (k <= l) GAPW_PUSH(s_o, i + 1, k, l, e.n_mm, e.n_gapo + 1, e.n_gape, e.n_ins, e.n_del + 1, BWTGAP_STATE_D, 1);
                        }
                    }
                } else if (e.state == BWTGAP_STATE_I) {
                    if (e.n_gape < O.max_gape)
                        GAPW_PUSH(s_e, i, k, l, e.n_mm, e.n_gapo, e.n_gape + 1, e.n_ins + 1, e.n_del, BWTGAP_STATE_I, 1);
                } else if (e.state == BWTGAP_STATE_D) {
                    if (delc) {
                        for (j = 0; j != 4; ++j) {
                            k = __shfl_sync(GAPW_FULL, kc, j);
                            l = __shfl_sync(GAPW_FULL, lc, j);
                            if (k <= l) GAPW_PUSH(s_e, i + 1, k, l, e.n_mm, e.n_gapo, e.n_gape + 1, e.n_ins, e.n_del + 1, BWTGAP_STATE_D, 1);
                        }
                    }
                }
            }
            if (allow_diff && allow_M) {
                for (j = 1; j <= 4; ++j) {
                    int c = (s + j) & 3;
                    int is_mm = (j != 4 || s > 3);
                    k = __shfl_sync(GAPW_FULL, kc, c);
                    l = __shfl_sync(GAPW_FULL, lc, c);
                    if (k <= l) {
                        if (!is_mm && fuse) GAPW_FUSE(k, l);    /* j == 4, s < 4: the last push */
                        else GAPW_PUSH(is_mm? s_m : b0, i, k, l, e.n_mm + is_mm, e.n_gapo, e.n_gape, e.n_ins, e.n_del, BWTGAP_STATE_M, is_mm);
                    }
                }
            } else if (s < 4) {
                k = __shfl_sync(GAPW_FULL, kc, s);
                l = __shfl_sync(GAPW_FULL, lc, s);
                if (k <= l) {
                    if (fuse) GAPW_FUSE(k, l);
                    else GAPW_PUSH(b0, i, k, l, e.n_mm, e.n_gapo, e.n_gape, e.n_ins, e.n_del, BWTGAP_STATE_M, 0);
                }
            }
        } else {
            /* no deletion and no mismatch children: at most an insertion extension, then the exact child */
            uint64_t nk = 1, nl = 0;
            if (s < 4) {
                gapw_occ1(bwt, bm, k - 1, l, s, &nk, &nl);
            }
            if (indel && e.state == BWTGAP_STATE_I && e.n_gape < O.max_gape)
                GAPW_PUSH(s_e, i, k, l, e.n_mm, e.n_gapo, e.n_gape + 1, e.n_ins + 1, e.n_del, BWTGAP_STATE_I, 1);
            if (s < 4) {
                if (nk <= nl) {
                    if (fuse) GAPW_FUSE(nk, nl);
                    else GAPW_PUSH(b0, i, nk, nl, e.n_mm, e.n_gapo, e.n_gape, e.n_ins, e.n_del, BWTGAP_STATE_M, 0);
                }
            }
        }
    }
    return n_aln;
}

#undef GAPW_PUSH
#undef GAPW_FUSE

/* persistent kernel: each warp claims list positions until the list is exhausted */
#ifndef GAPW_MINBLOCKS
#define GAPW_MINBLOCKS 5    /* caps registers at 96 so 5 blocks (20 warps) fit per SM: best on GH200 */
#endif
template <bool TRACE>
__global__ void __launch_bounds__(128, GAPW_MINBLOCKS) gapw_kernel(gapw_args a)
{
    extern __shared__ __align__(16) unsigned char gapw_smem_raw[];
    int lane = gapw_lane();
    gapw_smem *S = (gapw_smem*)gapw_smem_raw + (threadIdx.x >> 5);
    int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const bwtgpu_meta_t *bm = &g_meta;

    gapw_ctx C;
    C.a = &a;
    C.S = S;
    C.W.ncache = 0; C.W.lfree = -1; C.W.n_owned = 0;
    C.W.owned = a.owned + (size_t)gwarp * a.owned_cap;
    C.W.base = a.arena_chunks > 0? gwarp * a.arena_chunks : 0;
    C.W.bump = 0;
    C.L2own = bm->L2[lane & 3];
    bwtgap_aln_t *aln = a.aln_scratch + (size_t)gwarp * a.aln_cap;

    for (;;) {
        int pos = 0;
        if (lane == 0) pos = gapw_claim(a);
        pos = __shfl_sync(GAPW_FULL, pos, 0);
        if (pos < 0) break;
        int r = ((volatile const int*)a.list)[pos];
        int b = a.bid[r];
        int len = a.qlen[r];
        const uint8_t *q = a.query + a.qoff[r];
        int ns = a.nstacks[b];
        int ret;
        C.n_pops = 0; C.trace = BWTGAP_TRACE_INIT;
        C.cancel_flag = a.cancel? a.cancel + pos : NULL;
        const bwtgap_opt_t *bo = a.bopts + b;
        if (len > GAPW_MAXLEN || ns > GAPW_MAXSTACK || a.maxdiff[r] > 62 || bo->max_gapo > 63 || bo->max_gape > 63
            || (bm->seq_len >> GAPW_KBITS))
            ret = GAPW_ERR_UNSUP;   /* the 16-byte entry could not hold every push: the CPU takes the read */
        else {
            if (lane == 0) {
                S->opt = a.bopts[b];
                S->opt.max_diff = a.maxdiff[r];
                S->opt.seed_len = a.seed_len < len? a.seed_len : 0x7fffffff;
            }
            __syncwarp();
            C.n_stacks = ns;
            gapw_cal_width(a.bwt, bm, len, q, S->w, S->bid);
            bool use_seed = len > a.seed_len;
            if (use_seed) gapw_cal_width(a.bwt, bm, a.seed_len, q + (len - a.seed_len), S->w + GAPW_WOFF, S->bid + GAPW_WOFF);
            ret = gapw_match_gap<TRACE>(C, bm, len, q, use_seed, &S->opt, aln, a.aln_cap);
        }
        gapw_release_all(a, C.W);
        /* publish */
        int off = 0;
        if (ret > 0) {
            unsigned long long o64 = 0;
            if (lane == 0) o64 = atomicAdd(&a.ctl->out_top, (unsigned long long)ret);
            o64 = __shfl_sync(GAPW_FULL, o64, 0);
            if (o64 + (unsigned long long)ret > (unsigned long long)a.out_cap) ret = BWTGAP_ERR_ALN;
            else {
                off = (int)o64;
                for (int j = lane; j < ret; j += 32) a.out_rec[off + j] = aln[j];
            }
        }
        if (ret < 0 && ret != GAPW_ERR_CANCEL && a.fail && lane == 0) {
            unsigned int fi = atomicAdd_system(a.fail_n, 1u);
            ((volatile int*)a.fail)[fi] = ret == GAPW_ERR_UNSUP? (r | GAPW_FAIL_CPU) : r;   /* tag: only the CPU can take it */
            __threadfence_system();
        }
        if (lane == 0) {
            a.out_n[pos] = ret;
            a.out_off[pos] = off;
            a.out_pops[pos] = C.n_pops;
            a.out_trace[pos] = C.trace;
        }
        __syncwarp();
    }
}

#endif /* GAPW_CUH */
