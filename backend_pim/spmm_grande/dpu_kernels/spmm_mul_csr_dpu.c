/**
 * cgiannoula: christina.giann@gmail.com
 * Christina Giannoula
 * Sparse matrix vector multiplication with multiple tasklets.
 */

#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>
#include <seqread.h>

#include "../support/common.h"

__host dpu_arguments_t_csr DPU_INPUT_ARGUMENTS;
__host dpu_results_t DPU_RESULTS[NR_TASKLETS];


// Global Variables
// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);
uint32_t nnz_offset;

#define SEQREAD_CACHE_SIZE PIM_SEQREAD_CACHE_SIZE

/**
 * @fn task_main
 * @brief main function executed by each tasklet
 * @output sparse matrix vector multiplication
 */
int main() {
    uint32_t tasklet_id = me();

    if (tasklet_id == 0){ // Initialize once the cycle counter
        mem_reset(); // Reset the heap
#if PERF
        perfcounter_config(COUNT_CYCLES, true);
#endif
    }

    // Barrier
    //barrier_wait(&my_barrier);

    uint32_t nrows = DPU_INPUT_ARGUMENTS.nrows;
    uint32_t max_rows = DPU_INPUT_ARGUMENTS.max_rows;
    uint32_t max_nnz_ind = DPU_INPUT_ARGUMENTS.max_nnz_ind;
    uint32_t tcols = DPU_INPUT_ARGUMENTS.tcols;
    uint32_t dense_size = DPU_INPUT_ARGUMENTS.dense_size;
    uint32_t nnz_pad = DPU_INPUT_ARGUMENTS.nnz_pad;
    unsigned int start_row = DPU_INPUT_ARGUMENTS.start_row[tasklet_id];
    unsigned int rows_per_tasklet = DPU_INPUT_ARGUMENTS.rows_per_tasklet[tasklet_id];
    //printf("tasklet_id = %u, nrows = %u, max_rows = %u, max_nnz_ind = %u, tcols = %u, nnz_pad = %u\n", tasklet_id, nrows, max_rows, max_nnz_ind, tcols, nnz_pad);

    dpu_results_t *result = &DPU_RESULTS[tasklet_id];
    result->cycles = 0;

    uint32_t dense_size_byte = dense_size * byte_dt;

    // Addresses in MRAM
    uint32_t mram_base_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER);
    uint32_t mram_base_addr_x = (uint32_t) (DPU_MRAM_HEAP_POINTER + (max_rows * dense_size_byte));
    uint32_t mram_base_addr_rowptr = (uint32_t) (mram_base_addr_x + (tcols * dense_size_byte));
    uint32_t mram_base_addr_colind = (uint32_t) (mram_base_addr_rowptr + (max_rows * sizeof(uint32_t)));
    uint32_t mram_base_addr_values = (uint32_t) (mram_base_addr_colind + (max_nnz_ind * sizeof(uint32_t)));

    uint32_t i, j, k;
    //uint32_t *cache_temp = mem_alloc(8);
    uint64_t temp;
    val_dt *cache_x = mem_alloc(dense_size_byte); // FIXME: we assume that this is a multiple of 8
    val_dt *cache_acc = mem_alloc(dense_size_byte); // FIXME: we assume that this is a multiple of 8
    for (k = 0; k < dense_size; k++)
        cache_acc[k] = 0;
     
    if (rows_per_tasklet == 0) {
        goto EXIT;
    }
    
    //mram_read((__mram_ptr void const *) (mram_base_addr_rowptr), (void *) (cache_temp), 8);
    //nnz_offset = cache_temp[0];
    mram_read((__mram_ptr void const *) (mram_base_addr_rowptr), (void *) (&temp), 8);
    nnz_offset = (uint32_t) temp; // FIXME send it as an argument from host CPU

    mram_base_addr_rowptr += (start_row * sizeof(uint32_t));
    seqreader_buffer_t cache_rowptr = seqread_alloc();
    seqreader_t sr_rowptr;
    uint32_t *current_row = seqread_init(cache_rowptr, (__mram_ptr void *) mram_base_addr_rowptr, &sr_rowptr);
    uint32_t prev_row = *current_row;

    mram_base_addr_colind += ((prev_row - nnz_offset) * sizeof(uint32_t));
    mram_base_addr_values += ((prev_row - nnz_offset) * byte_dt);
    mram_base_addr_y += (start_row * dense_size_byte); 

    seqreader_buffer_t cache_colind = seqread_alloc();
    seqreader_t sr_colind;
    uint32_t *current_colind = seqread_init(cache_colind, (__mram_ptr void *) mram_base_addr_colind, &sr_colind);

    seqreader_buffer_t cache_val = seqread_alloc();
    seqreader_t sr_val;
    val_dt *current_val = seqread_init(cache_val, (__mram_ptr void *) mram_base_addr_values, &sr_val);


    if (start_row + rows_per_tasklet > nrows)
        rows_per_tasklet = nrows - start_row;

    for (i=start_row; i < start_row + rows_per_tasklet; i++) {
        current_row = seqread_get(current_row, sizeof(*current_row), &sr_rowptr);
        for (j=0; j < *current_row - prev_row; j++) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (*current_colind) * dense_size_byte), cache_x, dense_size_byte);
            for (k = 0; k < dense_size; k++)
                cache_acc[k] += (*current_val) * cache_x[k];

            current_colind = seqread_get(current_colind, sizeof(*current_colind), &sr_colind);
            current_val = seqread_get(current_val, sizeof(*current_val), &sr_val);
        }
        // write row
        mram_write(cache_acc, (__mram_ptr void *) (mram_base_addr_y), dense_size_byte);
        mram_base_addr_y += dense_size_byte;
        for (k = 0; k < dense_size; k++)
            cache_acc[k] = 0;

        // move to the next row
        prev_row = *current_row;
    }

EXIT: 

#if PERF
    result->cycles = perfcounter_get();
#endif

    return 0;
}
