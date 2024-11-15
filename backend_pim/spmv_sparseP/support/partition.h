#ifndef _PARTITION_H_
#define _PARTITION_H_

#include <stdio.h>

#if BLNC_TSKLT_NNZ_RGRN
/** 
 * @brief balance nnzs across tasklets at row granularity
 */
void partition_tsklt_by_nnz_rgrn(struct COOMatrix *A, uint32_t start_row, uint32_t prev_nnz, uint32_t last_nnz, uint32_t rows_per_dpu, uint32_t nnz_per_dpu, uint32_t *nnz_split_tasklet, int nr_of_tasklets);
#endif

#if BLNC_TSKLT_NNZ
/** 
 * @brief equally balance nnzs across tasklets
 */
void partition_tsklt_by_nnz(uint32_t nnz_per_dpu, uint32_t *nnz_split_tasklet, int nr_of_tasklets);
#endif

#endif
