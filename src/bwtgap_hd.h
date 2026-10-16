/* bwtgap_hd.h — DUAL-TARGET port of BWA's gapped backtracking search.
 *
 * Verbatim port of lh3/bwa 0.7.19-r1273:
 *   bwt.c      bwt_occ4, bwt_2occ4, bwt_match_exact_alt
 *   bwtaln.c   bwt_cal_width
 *   bwtgap.c   gap_push, gap_pop, gap_shadow, int_log2, bwt_match_gap
 * Compiles unchanged as host C (CPU reference / GPU host fallback) and as CUDA
 * device code (one thread per read), like bwtsearch.h.
 *
 * The only structural change is the priority stack. Upstream keeps one
 * growable LIFO array per score bucket; here each bucket is a singly-linked
 * LIFO threaded through a fixed-capacity, caller-provided entry pool (with a
 * free list). Pop order is identical: lowest score bucket first, most recently
 * pushed entry within it. When the pool or the caller's alignment array is
 * full, the search stops and returns an error code instead of reallocating;
 * the caller then re-runs that read with more space (the GPU driver hands it
 * to the host). Because a re-run starts from scratch, results never depend on
 * the capacities chosen.
 *
 * Upstream's own `n_entries > max_entries` cut-off is reproduced exactly as
 * long as the pool capacity is >= max_entries + 8 (at most 9 pushes follow
 * each pop).
 */
#ifndef BWTGAP_HD_H
#define BWTGAP_HD_H

#include <stdint.h>
#include "bwtsearch.h"

/* upstream mode bits (bwtaln.h) */
#define BWTGAP_MODE_GAPE     0x01
#define BWTGAP_MODE_COMPREAD 0x02
#define BWTGAP_MODE_LOGGAP   0x04
#define BWTGAP_MODE_NONSTOP  0x10

#define BWTGAP_STATE_M 0
#define BWTGAP_STATE_I 1
#define BWTGAP_STATE_D 2

/* negative return codes of bwtgap_match_gap */
#define BWTGAP_ERR_STACK (-1)   /* entry pool full: re-run with a larger pool */
#define BWTGAP_ERR_ALN   (-2)   /* alignment array full: re-run with a larger one */
#define BWTGAP_ERR_SCORE (-3)   /* score >= n_stacks: would be out of bounds upstream too */

/* bwt_width_t */
typedef struct {
    uint64_t w;
    int bid;
} bwtgap_width_t;

/* the gap_opt_t fields the search reads; max_diff and seed_len are per read */
typedef struct {
    int s_mm, s_gapo, s_gape;
    int mode;
    int indel_end_skip, max_del_occ, max_entries;
    int max_diff, max_gapo, max_gape;
    int max_seed_diff, seed_len;
    int max_top2;
} bwtgap_opt_t;

/* gap_entry_t plus a link for the bucket list / free list */
typedef struct {
    uint64_t k, l;          /* (k,l) is the SA region of [i,n-1] */
    uint32_t info;          /* score<<21 | i */
    int32_t  last_diff_pos;
    int32_t  next;
    uint8_t  n_mm, n_gapo, n_gape, state;
    uint16_t n_ins, n_del;
} bwtgap_entry_t;

typedef struct {
    bwtgap_entry_t *pool;   /* cap entries */
    int cap, top, free_head;
    int *heads;             /* n_stacks bucket heads, -1 == empty */
    int n_stacks, best, n_entries;
    uint64_t n_pops;        /* instrumentation only: pops performed by the last search */
    uint64_t trace;         /* instrumentation only: hash of every popped entry, in pop order */
} bwtgap_stack_t;

/* Trace hash: word-wise FNV-1a over each popped entry and the entry count after the pop.
 * Every engine (CPU, thread-per-read GPU, warp-per-read GPU) must produce the same value
 * for the same read; a mismatch pinpoints a read whose pop sequence differs. */
#define BWTGAP_TRACE_INIT 1469598103934665603ULL
BWTGPU_HD static inline uint64_t bwtgap_trace_mix(uint64_t h, uint64_t v)
{
    return (h ^ v) * 1099511628211ULL;
}
BWTGPU_HD static inline uint64_t bwtgap_trace_pop(uint64_t h, uint32_t info, uint64_t k, uint64_t l,
                                                  uint32_t n_mm, uint32_t n_gapo, uint32_t n_gape, uint32_t state,
                                                  uint32_t n_ins, uint32_t n_del, int last_diff_pos, int n_entries)
{
    h = bwtgap_trace_mix(h, info);
    h = bwtgap_trace_mix(h, k);
    h = bwtgap_trace_mix(h, l);
    h = bwtgap_trace_mix(h, n_mm | n_gapo << 8 | n_gape << 16 | state << 24);
    h = bwtgap_trace_mix(h, n_ins | (uint64_t)n_del << 16);
    h = bwtgap_trace_mix(h, (uint64_t)(uint32_t)last_diff_pos);
    h = bwtgap_trace_mix(h, (uint64_t)(uint32_t)n_entries);
    return h;
}

/* on-disk bwt_aln1_t: same 24-byte layout as sai_aln1_t */
typedef struct {
    uint64_t packed;        /* n_mm:8, n_gapo:8, n_gape:8, score:20, n_ins:10, n_del:10 */
    uint64_t k, l;
} bwtgap_aln_t;

BWTGPU_HD static inline uint64_t bwtgap_pack(uint32_t n_mm, uint32_t n_gapo, uint32_t n_gape,
                                             uint32_t score, uint32_t n_ins, uint32_t n_del)
{
    return ((uint64_t)(n_mm   & 0xff))
         | ((uint64_t)(n_gapo & 0xff)   << 8)
         | ((uint64_t)(n_gape & 0xff)   << 16)
         | ((uint64_t)(score  & 0xfffff)<< 24)
         | ((uint64_t)(n_ins  & 0x3ff)  << 44)
         | ((uint64_t)(n_del  & 0x3ff)  << 54);
}

/* bwt_gen_cnt_table */
static inline void bwtgap_gen_cnt_table(uint32_t cnt_table[256])
{
    int i, j;
    for (i = 0; i != 256; ++i) {
        uint32_t x = 0;
        for (j = 0; j != 4; ++j)
            x |= (((i&3) == j) + ((i>>2&3) == j) + ((i>>4&3) == j) + (i>>6 == j)) << (j<<3);
        cnt_table[i] = x;
    }
}

#define BWTGAP_OCC_AUX4(cnt, b) \
    ((cnt)[(b)&0xff] + (cnt)[(b)>>8&0xff] + (cnt)[(b)>>16&0xff] + (cnt)[(b)>>24])

/* bwt_occ4 */
BWTGPU_HD static inline void bwtgap_occ4(const uint32_t *bwt, const bwtgpu_meta_t *m, const uint32_t *cnt_table,
                                         uint64_t k, uint64_t cnt[4])
{
    uint64_t x;
    const uint32_t *p, *end;
    uint32_t tmp;
    if (k == (uint64_t)(-1)) {
        cnt[0] = cnt[1] = cnt[2] = cnt[3] = 0;
        return;
    }
    k -= (k >= m->primary); /* because $ is not in bwt */
    p = BWTGPU_OCC_INTV(bwt, k);
    cnt[0] = ((const uint64_t*)p)[0]; cnt[1] = ((const uint64_t*)p)[1];
    cnt[2] = ((const uint64_t*)p)[2]; cnt[3] = ((const uint64_t*)p)[3];
    p += sizeof(uint64_t);
    end = p + ((k>>4) - ((k&~BWTGPU_OCC_INTV_MASK)>>4));
    for (x = 0; p < end; ++p) x += BWTGAP_OCC_AUX4(cnt_table, *p);
    tmp = *p & ~((1U<<((~k&15)<<1)) - 1);
    x += BWTGAP_OCC_AUX4(cnt_table, tmp) - (~k&15);
    cnt[0] += x&0xff; cnt[1] += x>>8&0xff; cnt[2] += x>>16&0xff; cnt[3] += x>>24;
}

/* bwt_2occ4: an analogy to bwt_occ4() but more efficient, requiring k <= l */
BWTGPU_HD static inline void bwtgap_2occ4(const uint32_t *bwt, const bwtgpu_meta_t *m, const uint32_t *cnt_table,
                                          uint64_t k, uint64_t l, uint64_t cntk[4], uint64_t cntl[4])
{
    uint64_t _k, _l;
    _k = k - (k >= m->primary);
    _l = l - (l >= m->primary);
    if (_l>>7 != _k>>7 || k == (uint64_t)(-1) || l == (uint64_t)(-1)) {
        bwtgap_occ4(bwt, m, cnt_table, k, cntk);
        bwtgap_occ4(bwt, m, cnt_table, l, cntl);
    } else {
        uint64_t x, y;
        const uint32_t *p, *endk, *endl;
        uint32_t tmp;
        k -= (k >= m->primary); /* because $ is not in bwt */
        l -= (l >= m->primary);
        p = BWTGPU_OCC_INTV(bwt, k);
        cntk[0] = ((const uint64_t*)p)[0]; cntk[1] = ((const uint64_t*)p)[1];
        cntk[2] = ((const uint64_t*)p)[2]; cntk[3] = ((const uint64_t*)p)[3];
        p += sizeof(uint64_t);
        /* prepare cntk[] */
        endk = p + ((k>>4) - ((k&~BWTGPU_OCC_INTV_MASK)>>4));
        endl = p + ((l>>4) - ((l&~BWTGPU_OCC_INTV_MASK)>>4));
        for (x = 0; p < endk; ++p) x += BWTGAP_OCC_AUX4(cnt_table, *p);
        y = x;
        tmp = *p & ~((1U<<((~k&15)<<1)) - 1);
        x += BWTGAP_OCC_AUX4(cnt_table, tmp) - (~k&15);
        /* calculate cntl[] and finalize cntk[] */
        for (; p < endl; ++p) y += BWTGAP_OCC_AUX4(cnt_table, *p);
        tmp = *p & ~((1U<<((~l&15)<<1)) - 1);
        y += BWTGAP_OCC_AUX4(cnt_table, tmp) - (~l&15);
        cntl[0] = cntk[0]; cntl[1] = cntk[1]; cntl[2] = cntk[2]; cntl[3] = cntk[3];
        cntk[0] += x&0xff; cntk[1] += x>>8&0xff; cntk[2] += x>>16&0xff; cntk[3] += x>>24;
        cntl[0] += y&0xff; cntl[1] += y>>8&0xff; cntl[2] += y>>16&0xff; cntl[3] += y>>24;
    }
}

/* bwt_match_exact_alt: continues backward search from [*k0,*l0]. Returns int, as upstream. */
BWTGPU_HD static inline int bwtgap_match_exact_alt(const uint32_t *bwt, const bwtgpu_meta_t *m,
                                                   int len, const uint8_t *str, uint64_t *k0, uint64_t *l0)
{
    int i;
    uint64_t k, l, ok, ol;
    k = *k0; l = *l0;
    for (i = len - 1; i >= 0; --i) {
        uint8_t c = str[i];
        if (c > 3) return 0; /* there is an N here. no match */
        bwtgpu_2occ(bwt, m, k - 1, l, c, &ok, &ol);
        k = m->L2[c] + ok + 1;
        l = m->L2[c] + ol;
        if (k > l) return 0; /* no match */
    }
    *k0 = k; *l0 = l;
    return (int)(l - k + 1);
}

/* bwt_cal_width. Upstream runs it on the reversed, NOT yet complemented read.
 * Our stored query is reversed AND complemented, so pass comp=1 to undo the
 * complement on the fly (N stays >3; '-' (5) behaves like N here). */
BWTGPU_HD static inline int bwtgap_cal_width(const uint32_t *bwt, const bwtgpu_meta_t *m,
                                             int len, const uint8_t *str, int comp, bwtgap_width_t *width)
{
    uint64_t k, l, ok, ol;
    int i, bid;
    bid = 0;
    k = 0; l = m->seq_len;
    for (i = 0; i < len; ++i) {
        uint8_t c = str[i];
        if (comp) c = c > 3? 4 : 3 - c;
        if (c < 4) {
            bwtgpu_2occ(bwt, m, k - 1, l, c, &ok, &ol);
            k = m->L2[c] + ok + 1;
            l = m->L2[c] + ol;
        }
        if (k > l || c > 3) { /* then restart */
            k = 0;
            l = m->seq_len;
            ++bid;
        }
        width[i].w = l - k + 1;
        width[i].bid = bid;
    }
    width[len].w = 0;
    width[len].bid = ++bid;
    return bid;
}

/* aln_score */
BWTGPU_HD static inline int bwtgap_score(int m, int o, int e, const bwtgap_opt_t *p)
{
    return m * p->s_mm + o * p->s_gapo + e * p->s_gape;
}

/* gap_reset_stack */
BWTGPU_HD static inline void bwtgap_reset(bwtgap_stack_t *s)
{
    int i;
    for (i = 0; i != s->n_stacks; ++i) s->heads[i] = -1;
    s->best = s->n_stacks;
    s->n_entries = 0;
    s->top = 0;
    s->free_head = -1;
    s->n_pops = 0;
    s->trace = BWTGAP_TRACE_INIT;
}

/* gap_push. Returns 0, BWTGAP_ERR_STACK or BWTGAP_ERR_SCORE. */
BWTGPU_HD static inline int bwtgap_push(bwtgap_stack_t *s, int i, uint64_t k, uint64_t l,
                                        int n_mm, int n_gapo, int n_gape, int n_ins, int n_del,
                                        int state, int is_diff, const bwtgap_opt_t *opt)
{
    int score, idx;
    bwtgap_entry_t *p;
    score = bwtgap_score(n_mm, n_gapo, n_gape, opt);
    if (score < 0 || score >= s->n_stacks) return BWTGAP_ERR_SCORE;
    if (s->free_head >= 0) { idx = s->free_head; s->free_head = s->pool[idx].next; }
    else if (s->top < s->cap) idx = s->top++;
    else return BWTGAP_ERR_STACK;
    p = s->pool + idx;
    p->info = (uint32_t)score<<21 | i; p->k = k; p->l = l;
    p->n_mm = n_mm; p->n_gapo = n_gapo; p->n_gape = n_gape;
    p->n_ins = n_ins; p->n_del = n_del;
    p->state = state;
    p->last_diff_pos = is_diff? i : 0;
    p->next = s->heads[score];
    s->heads[score] = idx;
    ++(s->n_entries);
    if (s->best > score) s->best = score;
    return 0;
}

/* gap_pop */
BWTGPU_HD static inline void bwtgap_pop(bwtgap_stack_t *s, bwtgap_entry_t *e)
{
    int idx = s->heads[s->best];
    ++s->n_pops;
    *e = s->pool[idx];
    s->heads[s->best] = e->next;
    s->pool[idx].next = s->free_head;
    s->free_head = idx;
    --(s->n_entries);
    s->trace = bwtgap_trace_pop(s->trace, e->info, e->k, e->l, e->n_mm, e->n_gapo, e->n_gape, e->state,
                                e->n_ins, e->n_del, e->last_diff_pos, s->n_entries);
    if (s->heads[s->best] < 0 && s->n_entries) { /* reset best */
        int i;
        for (i = s->best + 1; i < s->n_stacks; ++i)
            if (s->heads[i] >= 0) break;
        s->best = i;
    } else if (s->n_entries == 0) s->best = s->n_stacks;
}

/* gap_shadow */
BWTGPU_HD static inline void bwtgap_shadow(int x, int len, uint64_t max, int last_diff_pos, bwtgap_width_t *w)
{
    int i, j;
    (void)len;
    for (i = j = 0; i < last_diff_pos; ++i) {
        if (w[i].w > (uint64_t)(int64_t)x) w[i].w -= x;
        else if (w[i].w == (uint64_t)(int64_t)x) {
            w[i].bid = 1;
            w[i].w = max - (++j);
        } /* else should not happen */
    }
}

BWTGPU_HD static inline int bwtgap_int_log2(uint32_t v)
{
    int c = 0;
    if (v & 0xffff0000u) { v >>= 16; c |= 16; }
    if (v & 0xff00) { v >>= 8; c |= 8; }
    if (v & 0xf0) { v >>= 4; c |= 4; }
    if (v & 0xc) { v >>= 2; c |= 2; }
    if (v & 0x2) c |= 1;
    return c;
}

#define BWTGAP_PUSH(...) do { int _r = bwtgap_push(stack, __VA_ARGS__, opt); if (_r) return _r; } while (0)

/* bwt_match_gap. seq is the reversed+complemented read. width/seed_width are
 * modified (gap_shadow), as upstream. Writes up to m_aln records to aln.
 * Returns n_aln (>= 0) or a negative BWTGAP_ERR_* code. */
BWTGPU_HD static inline int bwtgap_match_gap(const uint32_t *bwt, const bwtgpu_meta_t *bm, const uint32_t *cnt_table,
                                             int len, const uint8_t *seq, bwtgap_width_t *width,
                                             bwtgap_width_t *seed_width, const bwtgap_opt_t *opt,
                                             bwtgap_stack_t *stack, bwtgap_aln_t *aln, int m_aln)
{
    int best_score = bwtgap_score(opt->max_diff+1, opt->max_gapo+1, opt->max_gape+1, opt);
    int best_diff = opt->max_diff + 1, max_diff = opt->max_diff;
    int best_cnt = 0;
    int j, _j, n_aln = 0;
    (void)best_diff;

    /* check whether there are too many N */
    for (j = _j = 0; j < len; ++j)
        if (seq[j] > 3) ++_j;
    if (_j > max_diff) return 0;

    bwtgap_reset(stack);
    BWTGAP_PUSH(len, 0, bm->seq_len, 0, 0, 0, 0, 0, 0, 0);

    while (stack->n_entries) {
        bwtgap_entry_t e;
        int i, m, m_seed = 0, hit_found, allow_diff, allow_M, tmp;
        uint64_t k, l, cnt_k[4], cnt_l[4], occ;

        if (stack->n_entries > opt->max_entries) break;
        bwtgap_pop(stack, &e); /* get the best entry */
        k = e.k; l = e.l; /* SA interval */
        i = e.info&0xffff; /* length */
        if (!(opt->mode & BWTGAP_MODE_NONSTOP) && e.info>>21 > (uint32_t)(best_score + opt->s_mm)) break; /* no need to proceed */

        m = max_diff - (e.n_mm + e.n_gapo);
        if (opt->mode & BWTGAP_MODE_GAPE) m -= e.n_gape;
        if (m < 0) continue;
        if (seed_width) { /* apply seeding */
            m_seed = opt->max_seed_diff - (e.n_mm + e.n_gapo);
            if (opt->mode & BWTGAP_MODE_GAPE) m_seed -= e.n_gape;
        }
        if (i > 0 && m < width[i-1].bid) continue;

        /* check whether a hit is found */
        hit_found = 0;
        if (i == 0) hit_found = 1;
        else if (m == 0 && (e.state == BWTGAP_STATE_M || (opt->mode&BWTGAP_MODE_GAPE) || e.n_gape == opt->max_gape)) { /* no diff allowed */
            if (bwtgap_match_exact_alt(bwt, bm, i, seq, &k, &l)) hit_found = 1;
            else continue; /* no hit, skip */
        }

        if (hit_found) { /* action for found hits */
            int score = bwtgap_score(e.n_mm, e.n_gapo, e.n_gape, opt);
            int do_add = 1;
            if (n_aln == 0) {
                best_score = score;
                best_diff = e.n_mm + e.n_gapo;
                if (opt->mode & BWTGAP_MODE_GAPE) best_diff += e.n_gape;
                if (!(opt->mode & BWTGAP_MODE_NONSTOP))
                    max_diff = (best_diff + 1 > opt->max_diff)? opt->max_diff : best_diff + 1; /* top2 behaviour */
            }
            if (score == best_score) best_cnt += l - k + 1;
            else if (best_cnt > opt->max_top2) break; /* top2b behaviour */
            if (e.n_gapo) { /* check whether the hit has been found. this may happen when a gap occurs in a tandem repeat */
                for (j = 0; j != n_aln; ++j)
                    if (aln[j].k == k && aln[j].l == l) break;
                if (j < n_aln) do_add = 0;
            }
            if (do_add) { /* append */
                bwtgap_shadow(l - k + 1, len, bm->seq_len, e.last_diff_pos, width);
                if (n_aln == m_aln) return BWTGAP_ERR_ALN;
                aln[n_aln].packed = bwtgap_pack(e.n_mm, e.n_gapo, e.n_gape, score, e.n_ins, e.n_del);
                aln[n_aln].k = k; aln[n_aln].l = l;
                ++n_aln;
            }
            continue;
        }

        --i;
        bwtgap_2occ4(bwt, bm, cnt_table, k - 1, l, cnt_k, cnt_l); /* retrieve Occ values */
        occ = l - k + 1;
        /* test whether diff is allowed */
        allow_diff = allow_M = 1;
        if (i > 0) {
            int ii = i - (len - opt->seed_len);
            if (width[i-1].bid > m-1) allow_diff = 0;
            else if (width[i-1].bid == m-1 && width[i].bid == m-1 && width[i-1].w == width[i].w) allow_M = 0;
            if (seed_width && ii > 0) {
                if (seed_width[ii-1].bid > m_seed-1) allow_diff = 0;
                else if (seed_width[ii-1].bid == m_seed-1 && seed_width[ii].bid == m_seed-1
                         && seed_width[ii-1].w == seed_width[ii].w) allow_M = 0;
            }
        }
        /* indels */
        tmp = (opt->mode & BWTGAP_MODE_LOGGAP)? bwtgap_int_log2(e.n_gape + e.n_gapo)/2+1 : e.n_gapo + e.n_gape;
        if (allow_diff && i >= opt->indel_end_skip + tmp && len - i >= opt->indel_end_skip + tmp) {
            if (e.state == BWTGAP_STATE_M) { /* gap open */
                if (e.n_gapo < opt->max_gapo) { /* gap open is allowed */
                    /* insertion */
                    BWTGAP_PUSH(i, k, l, e.n_mm, e.n_gapo + 1, e.n_gape, e.n_ins + 1, e.n_del, BWTGAP_STATE_I, 1);
                    /* deletion */
                    for (j = 0; j != 4; ++j) {
                        k = bm->L2[j] + cnt_k[j] + 1;
                        l = bm->L2[j] + cnt_l[j];
                        if (k <= l) BWTGAP_PUSH(i + 1, k, l, e.n_mm, e.n_gapo + 1, e.n_gape, e.n_ins, e.n_del + 1, BWTGAP_STATE_D, 1);
                    }
                }
            } else if (e.state == BWTGAP_STATE_I) { /* extention of an insertion */
                if (e.n_gape < opt->max_gape) /* gap extention is allowed */
                    BWTGAP_PUSH(i, k, l, e.n_mm, e.n_gapo, e.n_gape + 1, e.n_ins + 1, e.n_del, BWTGAP_STATE_I, 1);
            } else if (e.state == BWTGAP_STATE_D) { /* extention of a deletion */
                if (e.n_gape < opt->max_gape) { /* gap extention is allowed */
                    if (e.n_gape + e.n_gapo < max_diff || occ < (uint64_t)(int64_t)opt->max_del_occ) {
                        for (j = 0; j != 4; ++j) {
                            k = bm->L2[j] + cnt_k[j] + 1;
                            l = bm->L2[j] + cnt_l[j];
                            if (k <= l) BWTGAP_PUSH(i + 1, k, l, e.n_mm, e.n_gapo, e.n_gape + 1, e.n_ins, e.n_del + 1, BWTGAP_STATE_D, 1);
                        }
                    }
                }
            }
        }
        /* mismatches */
        if (allow_diff && allow_M) { /* mismatch is allowed */
            for (j = 1; j <= 4; ++j) {
                int c = (seq[i] + j) & 3;
                int is_mm = (j != 4 || seq[i] > 3);
                k = bm->L2[c] + cnt_k[c] + 1;
                l = bm->L2[c] + cnt_l[c];
                if (k <= l) BWTGAP_PUSH(i, k, l, e.n_mm + is_mm, e.n_gapo, e.n_gape, e.n_ins, e.n_del, BWTGAP_STATE_M, is_mm);
            }
        } else if (seq[i] < 4) { /* try exact match only */
            int c = seq[i] & 3;
            k = bm->L2[c] + cnt_k[c] + 1;
            l = bm->L2[c] + cnt_l[c];
            if (k <= l) BWTGAP_PUSH(i, k, l, e.n_mm, e.n_gapo, e.n_gape, e.n_ins, e.n_del, BWTGAP_STATE_M, 0);
        }
    }
    return n_aln;
}

#undef BWTGAP_PUSH

#endif /* BWTGAP_HD_H */
