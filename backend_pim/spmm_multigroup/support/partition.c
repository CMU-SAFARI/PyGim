#ifndef _PARTITION_H_
#define _PARTITION_H_

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "partition.h"
#include "matrix.h"

#if BLNC_ROW
/** 
 * @brief balance rows across DPUs
 */
void partition_by_row_csr(struct CSRMatrix *csrMtx, uint32_t *row_split, int nr_of_dpus) {

    if (nr_of_dpus == 1) {
        row_split[0] = 0;
        row_split[1] = csrMtx->nrows;
        return;
    }

    // Compute the matrix splits.
    uint32_t chunks = csrMtx->nrows / nr_of_dpus;
    uint32_t rest_rows = csrMtx->nrows % nr_of_dpus;
    uint32_t rows_per_dpu;
    uint32_t curr_row = 0;
    uint32_t i;

    row_split[0] = curr_row;
    for (i = 0; i < nr_of_dpus; i++) {
        rows_per_dpu = chunks;
        if (i < rest_rows)
            rows_per_dpu++;
        curr_row += rows_per_dpu;
        if (curr_row > csrMtx->nrows)
            curr_row = csrMtx->nrows;
        row_split[i+1] = curr_row;
    }

    // Print partitioning
    //for (i = 0; i < nr_of_dpus; i++) {
    //    printf("[%d]: [%d, %d)\n", i, row_split[i], row_split[i+1]);
    //}
}
#endif

#if BLNC_NNZ
/** 
 * @brief balance nnz in row granularity across DPUs
 */
void partition_by_nnz_csr(struct CSRMatrix *csrMtx, uint32_t *row_split, int nr_of_dpus) {

    if (nr_of_dpus == 1) {
        row_split[0] = 0;
        row_split[1] = csrMtx->nrows;
        return;
    }

    // Compute the matrix splits.
    uint32_t nnz_cnt = csrMtx->nnz;
    uint32_t nnz_per_split = nnz_cnt / nr_of_dpus;
    uint32_t curr_nnz = 0;
    uint32_t row_start = 0;
    uint32_t split_cnt = 0;
    uint32_t i;

    row_split[0] = row_start;
    for (i = 0; i < csrMtx->nrows; i++) {
        curr_nnz += csrMtx->rowptr[i+1] - csrMtx->rowptr[i];
        if (curr_nnz >= nnz_per_split) {
            row_start = i + 1;
            ++split_cnt;
            if (split_cnt <= nr_of_dpus)
                row_split[split_cnt] = row_start;
            curr_nnz = 0;
        }
    }

    // Fill the last split with remaining elements
    if (curr_nnz < nnz_per_split && split_cnt <= nr_of_dpus) {
        row_split[++split_cnt] = csrMtx->nrows;
    }

    // If there are any remaining rows merge them in last partition
    if (split_cnt > nr_of_dpus) {
        row_split[nr_of_dpus] = csrMtx->nrows;
    }

    // If there are remaining threads create empty partitions
    for (i = split_cnt + 1; i <= nr_of_dpus; i++) {
        row_split[i] = csrMtx->nrows;
    }

    // Print partitioning
    //for (i = 0; i < nr_of_dpus; i++) {
    //    printf("[%d]: [%d, %d)\n", i, row_split[i], row_split[i+1]);
    //}

}
#endif

#if BLNC_NNZ_RGRN
/** 
 * @brief balance nnz in row granularity across DPUs
 */
void partition_by_nnz_rgrn_coo(struct COOMatrix *cooMtx, uint32_t *row_split, int nr_of_dpus) {
    if (nr_of_dpus == 1) {
        row_split[0] = 0;
        row_split[1] = cooMtx->nrows;
        return;
    }

    // Compute the matrix splits.
    uint32_t nnz_cnt = cooMtx->nnz;
    uint32_t nnz_per_split = nnz_cnt / nr_of_dpus;
    uint32_t curr_nnz = 0;
    uint32_t row_start = 0;
    uint32_t split_cnt = 0;
    uint32_t i;

    row_split[0] = row_start;
    for (i = 0; i < cooMtx->nrows; i++) {
        curr_nnz += cooMtx->rows[i];
        if (curr_nnz >= nnz_per_split) {
            row_start = i + 1;
            ++split_cnt;
            if (split_cnt <= nr_of_dpus)
                row_split[split_cnt] = row_start;
            curr_nnz = 0;
        }
    }

    // Fill the last split with remaining elements
    if (curr_nnz < nnz_per_split && split_cnt <= nr_of_dpus) {
        row_split[++split_cnt] = cooMtx->nrows;
    }

    // If there are any remaining rows merge them in last partition
    if (split_cnt > nr_of_dpus) {
        row_split[nr_of_dpus] = cooMtx->nrows;
    }

    // If there are remaining threads create empty partitions
    for (i = split_cnt + 1; i <= nr_of_dpus; i++) {
        row_split[i] = cooMtx->nrows;
    }
}
#endif

#if BLNC_TSKLT_ROW
/** 
 * @brief balance rows across tasklets
 */
void partition_tsklt_by_row_csr(int dpu, int rows_per_dpu, uint32_t *row_split_tasklet, int nr_of_tasklets) {

    // Compute the matrix splits.
    uint32_t granularity = 1;
    granularity = 8 / byte_dt;
    uint32_t chunks = rows_per_dpu / (granularity * nr_of_tasklets); 
    uint32_t rest_rows = rows_per_dpu % (granularity * nr_of_tasklets); 
    uint32_t rows_per_tasklet;
    uint32_t curr_row = 0;

    row_split_tasklet[0] = curr_row;
    for(unsigned int i=0; i < nr_of_tasklets; i++) {
        rows_per_tasklet = (granularity * chunks);
        if (i < rest_rows)
            rows_per_tasklet += granularity;
        curr_row += rows_per_tasklet;
        if (curr_row > rows_per_dpu)
            curr_row = rows_per_dpu;
        row_split_tasklet[i+1] = curr_row;
    }

    // Print partitioning
    //for (unsigned int i = 0; i < nr_of_tasklets; i++) {
    //    printf("[%d]: [%d, %d)\n", i, row_split_tasklet[i], row_split_tasklet[i+1]);
    //}
}
#endif

#if BLNC_TSKLT_NNZ
/** 
 * @brief balance nnz in row granularity across tasklets
 */
void partition_tsklt_by_nnz_csr(struct CSRMatrix *csrMtx, int dpu, int rows_per_dpu, uint32_t *row_split_tasklet, int nnz_per_dpu, int prev_rows_dpu, int nr_of_tasklets) {

    // Compute the matrix splits.
    uint32_t granularity = 1;
    granularity = 8 / byte_dt;
    uint32_t nnz_per_split = nnz_per_dpu / nr_of_tasklets; 
    uint32_t curr_nnz = 0;
    uint32_t row_start = 0;
    uint32_t split_cnt = 0;
    uint32_t t;

    row_split_tasklet[0] = row_start;
    for (t = 0; t < rows_per_dpu; t++) {
        curr_nnz += csrMtx->rowptr[prev_rows_dpu+t+1] - csrMtx->rowptr[prev_rows_dpu+t];
        if ((curr_nnz >= nnz_per_split) && ((t+1) % granularity == 0)) {
            row_start = t + 1;
            ++split_cnt;
            if (split_cnt <= nr_of_tasklets) {
                row_split_tasklet[split_cnt] = row_start;
            }
            curr_nnz = 0;
        }
    }

    // Fill the last split with remaining elements
    if (curr_nnz < nnz_per_split && split_cnt <= nr_of_tasklets) {
        row_split_tasklet[++split_cnt] = rows_per_dpu;
    }

    // If there are any remaining rows merge them in last partition
    if (split_cnt > nr_of_tasklets) {
        row_split_tasklet[nr_of_tasklets] = rows_per_dpu;
    }

    // If there are remaining threads create empty partitions
    for (t = split_cnt + 1; t <= nr_of_tasklets; t++) {
        row_split_tasklet[t] = rows_per_dpu;
    }

    // Print partitioning
    //for (i = 0; i < nr_of_tasklets; i++) {
    //    printf("[%d]: [%d, %d)\n", i, row_split_tasklet[i], row_split_tasklet[i+1]);
    //}
}

void partition_tsklt_by_nnz_coo(uint32_t nnz_per_dpu, uint32_t *nnz_split_tasklet, int nr_of_tasklets) {

    nnz_split_tasklet[0] = 0;
    for(unsigned int tasklet_id=0; tasklet_id < nr_of_tasklets; tasklet_id++) {
        uint32_t nnz_chunks_tskl = nnz_per_dpu / nr_of_tasklets;
        uint32_t rest_nnzs_tskl = nnz_per_dpu % nr_of_tasklets;
        uint32_t nnz_per_tasklet = nnz_chunks_tskl;
        uint32_t prev_nnz_tskl;

        if (tasklet_id < rest_nnzs_tskl)
            nnz_per_tasklet++;
/*
        if (rest_nnzs_tskl > 0) {
            if (tasklet_id >= rest_nnzs_tskl)
                prev_nnz_tskl = rest_nnzs_tskl * (nnz_chunks_tskl + 1) + (tasklet_id - rest_nnzs_tskl) * nnz_chunks_tskl;
            else
                prev_nnz_tskl = tasklet_id * (nnz_chunks_tskl + 1);
        } else {
            prev_nnz_tskl = tasklet_id * nnz_chunks_tskl;
        }
*/
        nnz_split_tasklet[tasklet_id+1] = nnz_split_tasklet[tasklet_id] + nnz_per_tasklet;
    }

    // Print partitioning
    //for (unsigned int t = 0; t < nr_of_tasklets; t++) {
    //    printf("[%d]: [%d, %d) %d\n", t, nnz_split_tasklet[t], nnz_split_tasklet[t+1], nnz_split_tasklet[t+1] - nnz_split_tasklet[t]);
    //}
    assert(nnz_split_tasklet[nr_of_tasklets] = nnz_per_dpu);

}
#endif


#if BLNC_TSKLT_NNZ_RGRN
/**
 * @brief balance nnzs across tasklets at row granularity
 */
void partition_tsklt_by_nnz_rgrn_coo(struct COOMatrix *A, uint32_t start_row, uint32_t prev_nnz, uint32_t last_nnz, uint32_t rows_per_dpu, uint32_t nnz_per_dpu, uint32_t *nnz_split_tasklet, int nr_of_tasklets) {

    // Compute the matrix splits.
    uint32_t nnz_per_split = nnz_per_dpu / nr_of_tasklets;
    int32_t curr_nnz = 0;
    curr_nnz -= prev_nnz;
    uint32_t split_cnt = 0;
    uint32_t t;

    nnz_split_tasklet[0] = 0;
    for (t = start_row; t < start_row + rows_per_dpu - 1; t++) {
        curr_nnz += A->rows[t];
        if (curr_nnz >= nnz_per_split) {
            ++split_cnt;
            if (split_cnt <= nr_of_tasklets) {
                nnz_split_tasklet[split_cnt] = nnz_split_tasklet[split_cnt-1] + curr_nnz;
            }
            curr_nnz = 0;
        }
    }

    // Get nnzs for the last row
    if (last_nnz != 0)
        curr_nnz += last_nnz;
    else
        curr_nnz += A->rows[start_row + rows_per_dpu - 1];

    // Fill the last split with remaining elements
    if (curr_nnz < nnz_per_split && split_cnt <= nr_of_tasklets) {
        nnz_split_tasklet[++split_cnt] = nnz_per_dpu;
    }

    // If there are any remaining rows merge them in last partition
    if (split_cnt > nr_of_tasklets) {
        nnz_split_tasklet[nr_of_tasklets] = nnz_per_dpu;
    }

    // If there are remaining threads create empty partitions
    for (t = split_cnt + 1; t <= nr_of_tasklets; t++) {
        nnz_split_tasklet[t] = nnz_per_dpu;
    }

    // Print partitioning
    ///for (t = 0; t < nr_of_tasklets; t++) {
    //    printf("[%d]: [%d, %d) %d\n", t, nnz_split_tasklet[t], nnz_split_tasklet[t+1], nnz_split_tasklet[t+1] - nnz_split_tasklet[t]);
    //}
    assert(nnz_split_tasklet[nr_of_tasklets] = nnz_per_dpu);

}
#endif


#endif
