/* aln_host.h — host-side pieces of bwa_cal_sa_reg_gap() (bwtaln.c), shared by
 * the CPU reference driver and the GPU driver's host fallback. */
#ifndef ALN_HOST_H
#define ALN_HOST_H

#include <stdint.h>
#include "bwtgap_hd.h"
#include "host_io.h"
#include "sai_format.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ALN_AVG_ERR 0.02   /* BWA_AVG_ERR */

/* bwa_cal_maxdiff */
int aln_cal_maxdiff(int l, double err, double thres);

/* Per-batch local_opt as set up at the top of bwa_cal_sa_reg_gap: max_diff from the
 * longest read in the batch, max_gapo clamped to it. Returns n_stacks for the batch. */
int aln_batch_opt(const sai_gap_opt_t *o, const read_t *reads, int n, bwtgap_opt_t *b);

/* Per-read max_diff (recomputed when -n is a fraction), as in the read loop. */
int aln_read_maxdiff(const sai_gap_opt_t *o, const bwtgap_opt_t *b, int len);

/* Per-thread workspace for host searches. */
typedef struct {
    bwtgap_entry_t *pool; int pool_cap;
    int *heads; int heads_cap;
    bwtgap_aln_t *aln; int aln_cap;
    bwtgap_width_t *w, *sw; int w_cap, sw_cap;
    int last_peak;  /* peak concurrent stack entries of the last successful search */
    uint64_t last_pops; /* pops of the last successful search */
    uint64_t last_trace; /* trace hash of the last successful search */
} aln_ws_t;

void aln_ws_init(aln_ws_t *ws);
void aln_ws_free(aln_ws_t *ws);
/* Size a workspace up front (pool for `pool_cap` entries, etc.) so searches never re-run to grow it. */
void aln_ws_prealloc(aln_ws_t *ws, int pool_cap, int n_stacks, int max_len, int seed_len);

/* Run the full per-read search on the host, growing pool/aln space and re-running
 * from scratch on overflow, so the result is the uncapped upstream result.
 * Returns n_aln (records in ws->aln) or a negative BWTGAP_ERR_* code on a hard error. */
int aln_search_host(const uint32_t *bwt, const bwtgpu_meta_t *m, const uint32_t *cnt_table,
                    const sai_gap_opt_t *o, const bwtgap_opt_t *b, int n_stacks,
                    const read_t *r, aln_ws_t *ws);

#ifdef __cplusplus
}
#endif

#endif
