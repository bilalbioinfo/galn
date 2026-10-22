/* gpu_sa.h -- GPU BWT of T = F . revcomp(F) for long texts (libcubwt is limited to n < 2^31). */
#ifndef GALN_GPU_SA_H
#define GALN_GPU_SA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* BWT of T$, T = F . revcomp(F) read from the forward .pac (l_pac bases), n = 2 * l_pac.
 * Writes L ($ removed, symbols 0..3) to the host buffer L[0..n-1] and returns primary (the row of $,
 * as libcubwt_bwt and libsais64 give). Returns -1, with nothing written, if the device memory is too
 * small (the caller then uses the CPU). */
int64_t galn_gpu_bwt(const uint8_t *pac, int64_t l_pac, uint8_t *L);

#ifdef __cplusplus
}
#endif

#endif
