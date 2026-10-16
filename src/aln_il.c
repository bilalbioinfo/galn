/* aln_il.c — see aln_il.h. The two halves below are bwtgap_match_gap's loop body
 * (bwtgap_hd.h) cut at the bwtgap_2occ4 call; keep them in step with it. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aln_il.h"

static void *xrealloc(void *p, size_t n)
{
    void *q = realloc(p, n);
    if (!q) { fprintf(stderr, "[aln_il] out of memory (%zu bytes)\n", n); exit(20); }
    return q;
}

void aln_il_init(aln_il_t *il, const uint32_t *bwt, const bwtgpu_meta_t *bm, const uint32_t *cnt_table, int n,
                 int pool_cap, int n_stacks, int max_len, int seed_len)
{
    int s;
    memset(il, 0, sizeof(*il));
    il->bwt = bwt; il->bm = bm; il->cnt_table = cnt_table;
    il->n = n < 1? 1 : n > ALN_IL_MAX? ALN_IL_MAX : n;
    for (s = 0; s < il->n; ++s) {
        aln_ws_init(&il->slot[s].ws);
        aln_ws_prealloc(&il->slot[s].ws, pool_cap, n_stacks, max_len, seed_len);
    }
}

void aln_il_free(aln_il_t *il)
{
    int s;
    for (s = 0; s < il->n; ++s) aln_ws_free(&il->slot[s].ws);
}

static void finish(aln_il_slot_t *S, int ret)
{
    S->ret = ret;
    S->status = ALN_IL_DONE;
    if (ret >= 0) {
        S->ws.last_peak = S->st.top;
        S->ws.last_pops = S->st.n_pops;
        S->ws.last_trace = S->st.trace;
    }
}

/* the part of bwtgap_match_gap before its loop; returns 0, or 1 if the read is already finished */
static int begin(aln_il_t *il, aln_il_slot_t *S)
{
    const bwtgap_opt_t *opt = &S->lo;
    const uint8_t *seq = S->r->query;
    int len = S->r->len, j, _j;
    bwtgap_stack_t *stack = &S->st;
    int r;

    bwtgap_cal_width(il->bwt, il->bm, len, seq, 1, S->ws.w);
    if (len > S->o->seed_len)
        bwtgap_cal_width(il->bwt, il->bm, S->o->seed_len, seq + (len - S->o->seed_len), 1, S->ws.sw);
    S->seed_width = len <= S->o->seed_len? NULL : S->ws.sw;
    stack->pool = S->ws.pool; stack->cap = S->ws.pool_cap; stack->top = 0; stack->free_head = -1;
    stack->heads = S->ws.heads; stack->n_stacks = S->n_stacks; stack->best = S->n_stacks;
    stack->n_entries = 0; stack->n_pops = 0; stack->trace = BWTGAP_TRACE_INIT;

    S->best_score = bwtgap_score(opt->max_diff+1, opt->max_gapo+1, opt->max_gape+1, opt);
    S->best_diff = opt->max_diff + 1; S->max_diff = opt->max_diff;
    S->best_cnt = 0; S->n_aln = 0;
    for (j = _j = 0; j < len; ++j)
        if (seq[j] > 3) ++_j;
    if (_j > S->max_diff) { finish(S, 0); return 1; }
    bwtgap_reset(stack);
    r = bwtgap_push(stack, len, 0, il->bm->seq_len, 0, 0, 0, 0, 0, 0, 0, opt);
    if (r) { finish(S, r); return 1; }
    return 0;
}

void aln_il_start(aln_il_t *il, int s, const sai_gap_opt_t *o, const bwtgap_opt_t *b, int n_stacks,
                  const read_t *r, int tag)
{
    aln_il_slot_t *S = &il->slot[s];
    aln_ws_t *ws = &S->ws;
    int len = r->len;
    S->r = r; S->o = o; S->b = b; S->n_stacks = n_stacks; S->tag = tag;
    S->lo = *b;
    S->lo.max_diff = aln_read_maxdiff(o, b, len);
    S->lo.seed_len = o->seed_len < len? o->seed_len : 0x7fffffff;
    /* workspace as in aln_search_host */
    if (ws->heads_cap < n_stacks) { ws->heads = (int*)xrealloc(ws->heads, (size_t)n_stacks * sizeof(int)); ws->heads_cap = n_stacks; }
    if (ws->w_cap < len + 1) { ws->w = (bwtgap_width_t*)xrealloc(ws->w, (size_t)(len + 1) * sizeof(bwtgap_width_t)); ws->w_cap = len + 1; }
    if (len > o->seed_len && ws->sw_cap < o->seed_len + 1) {
        ws->sw = (bwtgap_width_t*)xrealloc(ws->sw, (size_t)(o->seed_len + 1) * sizeof(bwtgap_width_t));
        ws->sw_cap = o->seed_len + 1;
    }
    if (ws->pool_cap == 0) { ws->pool_cap = 4096; ws->pool = (bwtgap_entry_t*)xrealloc(ws->pool, (size_t)ws->pool_cap * sizeof(bwtgap_entry_t)); }
    if (ws->aln_cap == 0) { ws->aln_cap = 64; ws->aln = (bwtgap_aln_t*)xrealloc(ws->aln, (size_t)ws->aln_cap * sizeof(bwtgap_aln_t)); }
    S->status = ALN_IL_RUN;
    begin(il, S);
}

/* overflow of the stack pool or the record array: grow and search again from scratch, exactly
 * as aln_search_host does, so the result is the uncapped upstream result */
static int regrow(aln_il_t *il, aln_il_slot_t *S, int ret)
{
    aln_ws_t *ws = &S->ws;
    long max_pool = (long)S->o->max_entries + 16;
    if (ret == BWTGAP_ERR_STACK && ws->pool_cap < max_pool) {
        long cap = (long)ws->pool_cap * 2;
        if (cap > max_pool) cap = max_pool;
        ws->pool_cap = (int)cap;
        ws->pool = (bwtgap_entry_t*)xrealloc(ws->pool, (size_t)ws->pool_cap * sizeof(bwtgap_entry_t));
    } else if (ret == BWTGAP_ERR_ALN) {
        ws->aln_cap *= 2;
        ws->aln = (bwtgap_aln_t*)xrealloc(ws->aln, (size_t)ws->aln_cap * sizeof(bwtgap_aln_t));
    } else return 0;
    S->status = ALN_IL_RUN;
    begin(il, S);
    return 1;
}

static void fail(aln_il_t *il, aln_il_slot_t *S, int ret)
{
    if (!regrow(il, S, ret)) finish(S, ret);
}

static inline void prefetch_occ(const uint32_t *bwt, const bwtgpu_meta_t *m, uint64_t k)
{
    if (k == (uint64_t)(-1)) return;
    k -= (k >= m->primary);
    __builtin_prefetch(BWTGPU_OCC_INTV(bwt, k), 0, 0);
}

/* half A: pops until an entry needs occurrence counts (prefetched) or the search ends */
static void half_a(aln_il_t *il, aln_il_slot_t *S)
{
    const uint32_t *bwt = il->bwt;
    const bwtgpu_meta_t *bm = il->bm;
    const bwtgap_opt_t *opt = &S->lo;
    /* work on local copies (registers) and store them back on every exit; through the slot
     * pointer the compiler would have to reload them after every stack write */
    bwtgap_stack_t st = S->st, *stack = &st;
    int best_score = S->best_score, best_diff = S->best_diff, max_diff = S->max_diff;
    int best_cnt = S->best_cnt, n_aln = S->n_aln;
    const uint8_t *seq = S->r->query;
    bwtgap_width_t *width = S->ws.w, *seed_width = S->seed_width;
    bwtgap_aln_t *aln = S->ws.aln;
    int len = S->r->len, m_aln = S->ws.aln_cap, j;
#define IL_SAVE() do { S->st = st; S->best_score = best_score; S->best_diff = best_diff; S->max_diff = max_diff; \
                       S->best_cnt = best_cnt; S->n_aln = n_aln; } while (0)

    while (stack->n_entries) {
        bwtgap_entry_t e;
        int i, m, m_seed = 0, hit_found;
        uint64_t k, l;

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
                if (n_aln == m_aln) { IL_SAVE(); fail(il, S, BWTGAP_ERR_ALN); return; }
                aln[n_aln].packed = bwtgap_pack(e.n_mm, e.n_gapo, e.n_gape, score, e.n_ins, e.n_del);
                aln[n_aln].k = k; aln[n_aln].l = l;
                ++n_aln;
            }
            continue;
        }

        /* the occurrence lookup comes next: prefetch its blocks and let the other reads run */
        S->e = e; S->i = i; S->m = m; S->m_seed = m_seed; S->k = k; S->l = l;
        prefetch_occ(bwt, bm, k - 1);
        prefetch_occ(bwt, bm, l);
        S->status = ALN_IL_OCC;
        IL_SAVE();
        return;
    }
    IL_SAVE();
    finish(S, n_aln);
}

#undef IL_SAVE
#define IL_PUSH(...) do { int _r = bwtgap_push(stack, __VA_ARGS__, opt); if (_r) { S->st = st; fail(il, S, _r); return; } } while (0)

/* half B: occurrence counts and the pushes */
static void half_b(aln_il_t *il, aln_il_slot_t *S)
{
    const bwtgpu_meta_t *bm = il->bm;
    const bwtgap_opt_t *opt = &S->lo;
    bwtgap_stack_t st = S->st, *stack = &st;      /* local copy, stored back below */
    const uint8_t *seq = S->r->query;
    bwtgap_width_t *width = S->ws.w, *seed_width = S->seed_width;
    int len = S->r->len, j, allow_diff, allow_M, tmp;
    const bwtgap_entry_t e = S->e;
    int i = S->i, m = S->m, m_seed = S->m_seed, max_diff = S->max_diff;
    uint64_t k = S->k, l = S->l, cnt_k[4], cnt_l[4], occ;

    S->status = ALN_IL_RUN;
    --i;
    bwtgap_2occ4(il->bwt, bm, il->cnt_table, k - 1, l, cnt_k, cnt_l); /* retrieve Occ values */
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
                IL_PUSH(i, k, l, e.n_mm, e.n_gapo + 1, e.n_gape, e.n_ins + 1, e.n_del, BWTGAP_STATE_I, 1);
                /* deletion */
                for (j = 0; j != 4; ++j) {
                    k = bm->L2[j] + cnt_k[j] + 1;
                    l = bm->L2[j] + cnt_l[j];
                    if (k <= l) IL_PUSH(i + 1, k, l, e.n_mm, e.n_gapo + 1, e.n_gape, e.n_ins, e.n_del + 1, BWTGAP_STATE_D, 1);
                }
            }
        } else if (e.state == BWTGAP_STATE_I) { /* extention of an insertion */
            if (e.n_gape < opt->max_gape) /* gap extention is allowed */
                IL_PUSH(i, k, l, e.n_mm, e.n_gapo, e.n_gape + 1, e.n_ins + 1, e.n_del, BWTGAP_STATE_I, 1);
        } else if (e.state == BWTGAP_STATE_D) { /* extention of a deletion */
            if (e.n_gape < opt->max_gape) { /* gap extention is allowed */
                if (e.n_gape + e.n_gapo < max_diff || occ < (uint64_t)(int64_t)opt->max_del_occ) {
                    for (j = 0; j != 4; ++j) {
                        k = bm->L2[j] + cnt_k[j] + 1;
                        l = bm->L2[j] + cnt_l[j];
                        if (k <= l) IL_PUSH(i + 1, k, l, e.n_mm, e.n_gapo, e.n_gape + 1, e.n_ins, e.n_del + 1, BWTGAP_STATE_D, 1);
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
            if (k <= l) IL_PUSH(i, k, l, e.n_mm + is_mm, e.n_gapo, e.n_gape, e.n_ins, e.n_del, BWTGAP_STATE_M, is_mm);
        }
    } else if (seq[i] < 4) { /* try exact match only */
        int c = seq[i] & 3;
        k = bm->L2[c] + cnt_k[c] + 1;
        l = bm->L2[c] + cnt_l[c];
        if (k <= l) IL_PUSH(i, k, l, e.n_mm, e.n_gapo, e.n_gape, e.n_ins, e.n_del, BWTGAP_STATE_M, 0);
    }
    S->st = st;
}

#undef IL_PUSH

void aln_il_round(aln_il_t *il)
{
    int s;
    for (s = 0; s < il->n; ++s)
        if (il->slot[s].status == ALN_IL_RUN) half_a(il, &il->slot[s]);
    for (s = 0; s < il->n; ++s)
        if (il->slot[s].status == ALN_IL_OCC) half_b(il, &il->slot[s]);
}
