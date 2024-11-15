#ifndef _PARTITION_H_
#define _PARTITION_H_

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "partition.h"
#include "matrix.h"

#if BLNC_TSKLT_NNZ_RGRN
/**
 * @brief balance nnzs across tasklets at row granularity
 */
void partition_tsklt_by_nnz_rgrn(struct COOMatrix *A, uint32_t start_row, uint32_t prev_nnz, uint32_t last_nnz, uint32_t rows_per_dpu, uint32_t nnz_per_dpu, uint32_t *nnz_split_tasklet, int nr_of_tasklets) {

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

#if BLNC_TSKLT_NNZ
/**
 * @brief equally balance nnzs across tasklets
 */
void partition_tsklt_by_nnz(uint32_t nnz_per_dpu, uint32_t *nnz_split_tasklet, int nr_of_tasklets) {

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


#endif
