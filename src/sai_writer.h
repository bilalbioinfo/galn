/* sai_writer.h — header-only .sai serialization (host-only). */
#ifndef SAI_WRITER_H
#define SAI_WRITER_H

#include <stdio.h>
#include <stdint.h>
#include "sai_format.h"

static inline void sai_write_header(FILE *fp, const sai_gap_opt_t *opt)
{
    fwrite(SAI_MAGIC_STR, 1, SAI_MAGIC_LEN, fp);
    fwrite(opt, sizeof(sai_gap_opt_t), 1, fp);
}

/* write one read's block: int32 n_aln then n_aln records */
static inline void sai_write_read(FILE *fp, int32_t n_aln, const sai_aln1_t *alns)
{
    fwrite(&n_aln, 4, 1, fp);
    for (int32_t i = 0; i < n_aln && alns; ++i)
        fwrite(&alns[i], sizeof(sai_aln1_t), 1, fp);
}

#endif
