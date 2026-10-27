/* gpu_sa.h -- GPU BWT of T = F . revcomp(F) for long texts (libcubwt is limited to n < 2^31). */
#ifndef GALN_GPU_SA_H
#define GALN_GPU_SA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* BWT and SA samples of T$, T = F . revcomp(F) read from the forward .pac (l_pac bases), n = 2 * l_pac.
 * Lf[0..n]: the BWT by matrix row, the $ included (at row primary, value 4); the caller leaves it out.
 * samples[k-1] = SA[32k], k = 1 .. (n + 32) / 32 - 1, as bwt_cal_sa(bwt, 32) gives.
 * Both are host buffers. Returns primary, or -1 (nothing usable written) if the device memory is too small;
 * the caller then uses the CPU. Device memory: about n/4 + n/8 bytes plus working space. */
int64_t galn_gpu_bwt(const uint8_t *pac, int64_t l_pac, uint8_t *Lf, uint64_t *samples);

/* G-ALN 0.2.4's builder (gpu_sa_ranks.cu): a rank for every position in device memory, about 6.25 bytes per
 * text symbol, faster while it fits. L[0..n-1] gets the BWT with $ removed; returns primary, or -1. */
int64_t galn_gpu_bwt_ranks(const uint8_t *pac, int64_t l_pac, uint8_t *L);

#ifdef __cplusplus
}
#endif

#endif
