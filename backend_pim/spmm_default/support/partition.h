#ifndef _PARTITION_H_
#define _PARTITION_H_

#include <stdio.h>

#if BLNC_ROW
/** 
 * @brief balance rows across DPUs
 */
void partition_by_row_csr(struct CSRMatrix *csrMtx, uint32_t *row_split, int nr_of_dpus);
#endif

#if BLNC_NNZ
/** 
 * @brief balance nnz in row granularity across DPUs
 */
void partition_by_nnz_csr(struct CSRMatrix *csrMtx, uint32_t *row_split, int nr_of_dpus);
#endif

#if BLNC_NNZ_RGRN
/** 
 * @brief balance nnz in row granularity across DPUs
 */
void partition_by_nnz_rgrn_coo(struct COOMatrix *cooMtx, uint32_t *row_split, int nr_of_dpus);
#endif

#if BLNC_TSKLT_ROW
/** 
 * @brief balance rows across tasklets
 */
void partition_tsklt_by_row_csr(int dpu, int rows_per_dpu, uint32_t *row_split_tasklet, int nr_of_tasklets);
#endif

#if BLNC_TSKLT_NNZ
/** 
 * @brief balance nnz in row granularity across tasklets
 */
void partition_tsklt_by_nnz_csr(struct CSRMatrix *csrMtx, int dpu, int rows_per_dpu, uint32_t *row_split_tasklet, int nnz_per_dpu, int prev_rows_dpu, int nr_of_tasklets);

/**
 * @brief equally balance nnzs across tasklets for coo
 */
void partition_tsklt_by_nnz_coo(uint32_t nnz_per_dpu, uint32_t *nnz_split_tasklet, int nr_of_tasklets);

#endif

#if BLNC_TSKLT_NNZ_RGRN
/**
 * @brief balance nnzs across tasklets at row granularity
 */
void partition_tsklt_by_nnz_rgrn_coo(struct COOMatrix *A, uint32_t start_row, uint32_t prev_nnz, uint32_t last_nnz, uint32_t rows_per_dpu, uint32_t nnz_per_dpu, uint32_t *nnz_split_tasklet, int nr_of_tasklets);
#endif

#endif
