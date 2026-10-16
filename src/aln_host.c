/* aln_host.c — see aln_host.h. Compiled with -ffp-contract=off so
 * aln_cal_maxdiff rounds exactly like upstream on every host architecture. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aln_host.h"

int aln_cal_maxdiff(int l, double err, double thres)
{
    double elambda = exp(-l * err);
    double sum, y = 1.0;
    int k, x = 1;
    for (k = 1, sum = elambda; k < 1000; ++k) {
        y *= l * err;
        x = (int)((unsigned)x * (unsigned)k); /* upstream: x *= k (wraps on overflow) */
        sum += elambda * y / x;
        if (1.0 - sum < thres) return k;
    }
    return 2;
}

int aln_batch_opt(const sai_gap_opt_t *o, const read_t *reads, int n, bwtgap_opt_t *b)
{
    int i, max_len = 0;
    b->s_mm = o->s_mm; b->s_gapo = o->s_gapo; b->s_gape = o->s_gape;
    b->mode = o->mode;
    b->indel_end_skip = o->indel_end_skip; b->max_del_occ = o->max_del_occ; b->max_entries = o->max_entries;
    b->max_diff = o->max_diff; b->max_gapo = o->max_gapo; b->max_gape = o->max_gape;
    b->max_seed_diff = o->max_seed_diff; b->seed_len = o->seed_len;
    b->max_top2 = o->max_top2;
    for (i = 0; i != n; ++i)
        if (reads[i].len > max_len) max_len = reads[i].len;
    if (o->fnr > 0.0) b->max_diff = aln_cal_maxdiff(max_len, ALN_AVG_ERR, o->fnr);
    if (b->max_diff < b->max_gapo) b->max_gapo = b->max_diff;
    /* gap_init_stack(max_diff, max_gapo, max_gape) */
    return bwtgap_score(b->max_diff + 1, b->max_gapo + 1, b->max_gape + 1, b);
}

int aln_read_maxdiff(const sai_gap_opt_t *o, const bwtgap_opt_t *b, int len)
{
    return (o->fnr > 0.0)? aln_cal_maxdiff(len, ALN_AVG_ERR, o->fnr) : b->max_diff;
}

void aln_ws_init(aln_ws_t *ws)
{
    memset(ws, 0, sizeof(*ws));
}

void aln_ws_free(aln_ws_t *ws)
{
    free(ws->pool); free(ws->heads); free(ws->aln); free(ws->w); free(ws->sw);
    memset(ws, 0, sizeof(*ws));
}

static void *xrealloc(void *p, size_t n)
{
    void *q = realloc(p, n);
    if (!q) { fprintf(stderr, "[aln_host] out of memory (%zu bytes)\n", n); exit(20); }
    return q;
}

void aln_ws_prealloc(aln_ws_t *ws, int pool_cap, int n_stacks, int max_len, int seed_len)
{
    if (ws->pool_cap < pool_cap) {
        ws->pool = (bwtgap_entry_t*)xrealloc(ws->pool, (size_t)pool_cap * sizeof(bwtgap_entry_t));
        ws->pool_cap = pool_cap;
    }
    if (ws->heads_cap < n_stacks) {
        ws->heads = (int*)xrealloc(ws->heads, (size_t)n_stacks * sizeof(int));
        ws->heads_cap = n_stacks;
    }
    if (ws->w_cap < max_len + 1) {
        ws->w = (bwtgap_width_t*)xrealloc(ws->w, (size_t)(max_len + 1) * sizeof(bwtgap_width_t));
        ws->w_cap = max_len + 1;
    }
    if (max_len > seed_len && ws->sw_cap < seed_len + 1) {
        ws->sw = (bwtgap_width_t*)xrealloc(ws->sw, (size_t)(seed_len + 1) * sizeof(bwtgap_width_t));
        ws->sw_cap = seed_len + 1;
    }
    if (ws->aln_cap < 256) {
        ws->aln = (bwtgap_aln_t*)xrealloc(ws->aln, 256 * sizeof(bwtgap_aln_t));
        ws->aln_cap = 256;
    }
}

int aln_search_host(const uint32_t *bwt, const bwtgpu_meta_t *m, const uint32_t *cnt_table,
                    const sai_gap_opt_t *o, const bwtgap_opt_t *b, int n_stacks,
                    const read_t *r, aln_ws_t *ws)
{
    bwtgap_opt_t lo = *b;
    bwtgap_stack_t st;
    int ret, len = r->len;
    /* enough entries to reproduce upstream's max_entries cut-off exactly */
    long max_pool = (long)o->max_entries + 16;

    lo.max_diff = aln_read_maxdiff(o, b, len);
    lo.seed_len = o->seed_len < len? o->seed_len : 0x7fffffff;

    if (ws->heads_cap < n_stacks) {
        ws->heads = (int*)xrealloc(ws->heads, (size_t)n_stacks * sizeof(int));
        ws->heads_cap = n_stacks;
    }
    if (ws->w_cap < len + 1) {
        ws->w = (bwtgap_width_t*)xrealloc(ws->w, (size_t)(len + 1) * sizeof(bwtgap_width_t));
        ws->w_cap = len + 1;
    }
    if (len > o->seed_len && ws->sw_cap < o->seed_len + 1) {
        ws->sw = (bwtgap_width_t*)xrealloc(ws->sw, (size_t)(o->seed_len + 1) * sizeof(bwtgap_width_t));
        ws->sw_cap = o->seed_len + 1;
    }
    if (ws->pool_cap == 0) {
        ws->pool_cap = 4096;
        ws->pool = (bwtgap_entry_t*)xrealloc(ws->pool, (size_t)ws->pool_cap * sizeof(bwtgap_entry_t));
    }
    if (ws->aln_cap == 0) {
        ws->aln_cap = 64;
        ws->aln = (bwtgap_aln_t*)xrealloc(ws->aln, (size_t)ws->aln_cap * sizeof(bwtgap_aln_t));
    }

    for (;;) {
        /* widths are modified by gap_shadow, so recompute them on every attempt */
        bwtgap_cal_width(bwt, m, len, r->query, 1, ws->w);
        if (len > o->seed_len)
            bwtgap_cal_width(bwt, m, o->seed_len, r->query + (len - o->seed_len), 1, ws->sw);
        st.pool = ws->pool; st.cap = ws->pool_cap; st.top = 0; st.free_head = -1;
        st.heads = ws->heads; st.n_stacks = n_stacks; st.best = n_stacks; st.n_entries = 0; st.n_pops = 0; st.trace = BWTGAP_TRACE_INIT;
        ret = bwtgap_match_gap(bwt, m, cnt_table, len, r->query, ws->w,
                               len <= o->seed_len? NULL : ws->sw, &lo, &st, ws->aln, ws->aln_cap);
        if (ret == BWTGAP_ERR_STACK && ws->pool_cap < max_pool) {
            long cap = (long)ws->pool_cap * 2;
            if (cap > max_pool) cap = max_pool;
            ws->pool_cap = (int)cap;
            ws->pool = (bwtgap_entry_t*)xrealloc(ws->pool, (size_t)ws->pool_cap * sizeof(bwtgap_entry_t));
            continue;
        }
        if (ret == BWTGAP_ERR_ALN) {
            ws->aln_cap *= 2;
            ws->aln = (bwtgap_aln_t*)xrealloc(ws->aln, (size_t)ws->aln_cap * sizeof(bwtgap_aln_t));
            continue;
        }
        ws->last_peak = st.top; /* slots are recycled before new ones are taken: top == peak n_entries */
        ws->last_pops = st.n_pops;
        ws->last_trace = st.trace;
        return ret;
    }
}
