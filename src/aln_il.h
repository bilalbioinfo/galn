/* aln_il.h — interleaved host search: several reads advance in turn on one CPU core.
 *
 * bwt_match_gap (bwtgap_match_gap) spends most of its time waiting for one random BWT
 * block per pop. Here the loop body is split at that access into two halves:
 *   A  pop the best entry and run every test up to the occurrence lookup (hits, cut-offs,
 *      exact-match tails), then prefetch the BWT blocks of k-1 and l and stop;
 *   B  compute the occurrence counts and push the children.
 * A worker keeps up to ALN_IL_MAX reads in flight and runs A for all of them, then B for
 * all of them, so the memory latencies of different reads overlap. Each read still goes
 * through exactly the same sequence of pops and pushes as bwtgap_match_gap, so results,
 * pop counts and trace hashes are identical (aln_search_host stays the reference).
 */
#ifndef ALN_IL_H
#define ALN_IL_H

#include "aln_host.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ALN_IL_MAX 16

enum { ALN_IL_IDLE = 0, ALN_IL_RUN, ALN_IL_OCC, ALN_IL_DONE };

typedef struct {
    int status;                 /* ALN_IL_* */
    int ret;                    /* n_aln or BWTGAP_ERR_* once DONE */
    int tag;                    /* caller's read id */
    /* the read and its options */
    const read_t *r;
    const sai_gap_opt_t *o;
    const bwtgap_opt_t *b;
    bwtgap_opt_t lo;
    int n_stacks;
    /* workspace (stack pool, widths, records), as in aln_search_host */
    aln_ws_t ws;
    bwtgap_stack_t st;
    bwtgap_width_t *seed_width;
    /* bwt_match_gap locals that live across iterations */
    int best_score, best_diff, max_diff, best_cnt, n_aln;
    /* state carried from half A to half B */
    bwtgap_entry_t e;
    int i, m, m_seed;
    uint64_t k, l;
} aln_il_slot_t;

typedef struct {
    const uint32_t *bwt;
    const bwtgpu_meta_t *bm;
    const uint32_t *cnt_table;
    int n;                      /* slots in use (1..ALN_IL_MAX) */
    aln_il_slot_t slot[ALN_IL_MAX];
} aln_il_t;

void aln_il_init(aln_il_t *il, const uint32_t *bwt, const bwtgpu_meta_t *bm, const uint32_t *cnt_table, int n,
                 int pool_cap, int n_stacks, int max_len, int seed_len);
void aln_il_free(aln_il_t *il);

/* start read r (caller id `tag`) in an idle slot s */
void aln_il_start(aln_il_t *il, int s, const sai_gap_opt_t *o, const bwtgap_opt_t *b, int n_stacks,
                  const read_t *r, int tag);

/* one round: half A on every running slot, then half B on every slot waiting for its
 * counts. Slots that finish become ALN_IL_DONE (records in slot.ws.aln, count in ret). */
void aln_il_round(aln_il_t *il);

#ifdef __cplusplus
}
#endif

#endif
