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
#include <mutex.h>

#include "../support/common.h"

__host dpu_arguments_t_coo DPU_INPUT_ARGUMENTS;
__host dpu_results_t DPU_RESULTS[NR_TASKLETS];


// Global Variables
// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

#define SEQREAD_CACHE_SIZE PIM_SEQREAD_CACHE_SIZE

#if CG_LOCK
MUTEX_INIT(my_mutex);
#endif

uint32_t *sync_rptr; // global 
val_dt *sync_values; // global

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
    uint32_t nnzs = DPU_INPUT_ARGUMENTS.nnzs;
    uint32_t max_rows_per_dpu = DPU_INPUT_ARGUMENTS.max_rows;
    uint32_t max_nnzs = DPU_INPUT_ARGUMENTS.max_nnzs;
    uint32_t tcols = DPU_INPUT_ARGUMENTS.tcols;
    uint32_t is_used = DPU_INPUT_ARGUMENTS.is_used;
    uint32_t tstart_row = DPU_INPUT_ARGUMENTS.tstart_row;
    unsigned int start_nnz = DPU_INPUT_ARGUMENTS.start_nnz[tasklet_id];
    unsigned int nnz_per_tasklet = DPU_INPUT_ARGUMENTS.nnz_per_tasklet[tasklet_id];
    //printf("tasklet_id = %u, nrows = %u, max_rows = %u, tcols = %u, start_nnz = %u, nnz_per_tasklet = %u\n", tasklet_id, nrows, max_rows_per_dpu, tcols, start_nnz, nnz_per_tasklet);

    dpu_results_t *result = &DPU_RESULTS[tasklet_id];
    result->cycles = 0;

    if (is_used == 0) {
        goto EXIT;
    }

    // Addresses in MRAM
    uint32_t mram_base_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER);
    uint32_t mram_temp_addr_y;
    uint32_t mram_base_addr_x = (uint32_t) (DPU_MRAM_HEAP_POINTER + (max_rows_per_dpu * sizeof(val_dt)));
    uint32_t mram_base_addr_rowind = (uint32_t) (mram_base_addr_x + (tcols * sizeof(val_dt)));
    uint32_t mram_base_addr_colind = (uint32_t) (mram_base_addr_rowind + (max_nnzs * sizeof(uint32_t)));
    uint32_t mram_base_addr_val = (uint32_t) (mram_base_addr_colind + (max_nnzs * sizeof(uint32_t)));

    uint32_t i, j;
    val_dt *cache_y = mem_alloc(8);
    val_dt *cache_x = mem_alloc(8);

    // Initialize y vector in MRAM
    if(tasklet_id == 0) { 
#if INT8
        cache_y[0] = 0;
        cache_y[1] = 0;
        cache_y[2] = 0;
        cache_y[3] = 0;
        cache_y[4] = 0;
        cache_y[5] = 0;
        cache_y[6] = 0;
        cache_y[7] = 0;
#elif INT16
        cache_y[0] = 0;
        cache_y[1] = 0;
        cache_y[2] = 0;
        cache_y[3] = 0;
#elif INT32
        cache_y[0] = 0;
        cache_y[1] = 0;
#elif INT64
        cache_y[0] = 0;
#elif FLT32
        cache_y[0] = 0;
        cache_y[1] = 0;
#elif DBL64
        cache_y[0] = 0;
#else
        cache_y[0] = 0;
        cache_y[1] = 0;
#endif

        uint32_t iter = 0;
#if INT8
        iter = (max_rows_per_dpu >> 3);
#elif INT16
        iter = (max_rows_per_dpu >> 2);
#elif INT32
        iter = (max_rows_per_dpu >> 1);
#elif INT64
        iter = max_rows_per_dpu;
#elif FLT32
        iter = (max_rows_per_dpu >> 1);
#elif DBL64
        iter = max_rows_per_dpu;
#else
        iter = (max_rows_per_dpu >> 1);
#endif
        for(i=0; i < iter; i++) {
            mram_write(cache_y, (__mram_ptr void *) (mram_base_addr_y), 8);
            mram_base_addr_y += 8;
        }

        sync_rptr = mem_alloc(NR_TASKLETS * 8);
        sync_values = mem_alloc(NR_TASKLETS * 8);
        iter = NR_TASKLETS * (8 / byte_dt);
        for(i=0; i < iter; i++) {
            sync_rptr[i] = 0;
            sync_values[i] = 0;
        }
    }

    barrier_wait(&my_barrier);


    mram_base_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER);

    //if (nnz_per_tasklet == 0) {
    //    goto EXIT;
    //}

    mram_base_addr_rowind += (start_nnz * sizeof(uint32_t));
    mram_base_addr_colind += (start_nnz * sizeof(uint32_t));
    mram_base_addr_val += (start_nnz * sizeof(val_dt));
    seqreader_buffer_t cache_rowind = seqread_alloc();
    seqreader_t sr_rowind;
    uint32_t *cur_rowind = seqread_init(cache_rowind, (__mram_ptr void *) mram_base_addr_rowind, &sr_rowind);
    uint32_t prev_row = *cur_rowind;

    seqreader_buffer_t cache_colind = seqread_alloc();
    seqreader_t sr_colind;
    uint32_t *cur_colind = seqread_init(cache_colind, (__mram_ptr void *) mram_base_addr_colind, &sr_colind);

    seqreader_buffer_t cache_val = seqread_alloc();
    seqreader_t sr_val;
    val_dt *cur_val = seqread_init(cache_val, (__mram_ptr void *) mram_base_addr_val, &sr_val);


    uint32_t diff;
    val_dt acc = 0;
    uint32_t row_bound = 8 / byte_dt; 
    uint32_t row_indx = 0; 
    uint32_t sync_base = tasklet_id * row_bound;


#if INT8
    diff = prev_row - tstart_row;
    row_indx = (diff & 7);
    sync_rptr[tasklet_id] = diff - (diff & 7);
#elif INT16
    diff = prev_row - tstart_row;
    row_indx = (diff & 3);
    sync_rptr[tasklet_id] = diff - (diff & 3);
#elif INT32
    diff = prev_row - tstart_row;
    row_indx = (diff & 1);
    sync_rptr[tasklet_id] = diff - (diff & 1);
#elif INT64
    diff = prev_row - tstart_row;
    sync_rptr[tasklet_id] = diff;
#elif FLT32
    diff = prev_row - tstart_row;
    row_indx = (diff & 1);
    sync_rptr[tasklet_id] = diff - (diff & 1);
#elif DBL64
    diff = prev_row - tstart_row;
    sync_rptr[tasklet_id] = diff;
#else
    diff = prev_row - tstart_row;
    row_indx = (diff & 1);
    sync_rptr[tasklet_id] = diff - (diff & 1);
#endif


    for(i=0; i<nnz_per_tasklet; i++) {
        if((*cur_rowind) != prev_row) {
            sync_values[sync_base + row_indx] = acc;  
            acc = 0;

            row_indx += ((*cur_rowind)  - prev_row);
            if(row_indx >= row_bound) {
                prev_row = (*cur_rowind);
                break;
            }
            prev_row = (*cur_rowind);

        }
        
#if INT8
        if (((*cur_colind) & 7) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (*cur_colind) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[0];
        } else if (((*cur_colind) & 7) == 1) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 1) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[1];
        } else if (((*cur_colind)& 7) == 2) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 2) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[2];
        } else if (((*cur_colind) & 7) == 3) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 3) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[3];
        } else if (((*cur_colind) & 7) == 4) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 4) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[4];
        } else if (((*cur_colind) & 7) == 5) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 5) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[5];
        } else if (((*cur_colind) & 7) == 6) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 6) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[6];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 7) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[7];
        }
#elif INT16
        if (((*cur_colind) & 3) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (*cur_colind) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[0];
        } else if (((*cur_colind) & 3) == 1) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 1) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[1];
        } else if (((*cur_colind) & 3) == 2) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 2) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[2];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 3) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[3];
        } 
#elif INT32
        if (((*cur_colind) & 1) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (*cur_colind) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[0];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 1) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[1];
        } 
#elif INT64
        mram_read((__mram_ptr void const *) (mram_base_addr_x + (*cur_colind) * sizeof(val_dt)), cache_x, 8);
        acc += (*cur_val) * cache_x[0];
#elif FLT32
        if (((*cur_colind) & 1) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (*cur_colind) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[0];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 1) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[1];
        } 
#elif DBL64 
        mram_read((__mram_ptr void const *) (mram_base_addr_x + (*cur_colind) * sizeof(val_dt)), cache_x, 8);
        acc += (*cur_val) * cache_x[0];
#else
        if (((*cur_colind) & 1) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (*cur_colind) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[0];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 1) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[1];
        } 
#endif
        cur_rowind = seqread_get(cur_rowind, sizeof(uint32_t), &sr_rowind);
        cur_colind = seqread_get(cur_colind, sizeof(uint32_t), &sr_colind);
        cur_val = seqread_get(cur_val, sizeof(val_dt), &sr_val);
      }


    if (i == nnz_per_tasklet && row_indx < row_bound) {
        sync_values[sync_base + row_indx] = acc;  
        acc = 0;
    }

    for(i; i<nnz_per_tasklet; i++) {
        if((*cur_rowind) != prev_row) {
            diff = prev_row - tstart_row;
#if INT8
            if ((diff & 7) == 0) {
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[0] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 7) == 1) {
                diff -= 1; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[1] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 7) == 2) {
                diff -= 2; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[2] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 7) == 3) {
                diff -= 3; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[3] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 7) == 4) {
                diff -= 4; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[4] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 7) == 5) {
                diff -= 5; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[5] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 7) == 6) {
                diff -= 6; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[6] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else {
                diff -= 7; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[7] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            }

#elif INT16
            if ((diff & 3) == 0) {
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[0] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 3) == 1) {
                diff -= 1; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[1] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 3) == 2) {
                diff -= 2; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[2] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else {
                diff -= 3; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[3] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            }

#elif INT32
            if ((diff & 1) == 0) {
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8); 
                cache_y[0] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else {
                diff -= 1; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[1] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            }

#elif INT64
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 3)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[0] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);

#elif FLT32
            if ((diff & 1) == 0) {
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[0] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else {
                diff -= 1; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[1] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            }

#elif DBL64 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 3)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[0] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);

#else
            if ((diff & 1) == 0) {
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[0] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else {
                diff -= 1; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[1] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            }

#endif
            acc = 0;
            prev_row = (*cur_rowind);
        }

#if INT8
        if (((*cur_colind) & 7) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (*cur_colind) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[0];
        } else if (((*cur_colind) & 7) == 1) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 1) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[1];
        } else if (((*cur_colind) & 7) == 2) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 2) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[2];
        } else if (((*cur_colind) & 7) == 3) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 3) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[3];
        } else if (((*cur_colind) & 7) == 4) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 4) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[4];
        } else if (((*cur_colind) & 7) == 5) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 5) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[5];
        } else if (((*cur_colind) & 7) == 6) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 6) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[6];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 7) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[7];
        }
#elif INT16
        if (((*cur_colind) & 3) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (*cur_colind) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[0];
        } else if (((*cur_colind)& 3) == 1) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 1) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[1];
        } else if (((*cur_colind) & 3) == 2) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 2) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[2];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 3) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[3];
        } 
#elif INT32
        if (((*cur_colind) & 1) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (*cur_colind) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[0];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 1) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[1];
        } 
#elif INT64
        mram_read((__mram_ptr void const *) (mram_base_addr_x + (*cur_colind) * sizeof(val_dt)), cache_x, 8);
        acc += (*cur_val) * cache_x[0];
#elif FLT32
        if (((*cur_colind) & 1) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (*cur_colind) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[0];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 1) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[1];
        } 
#elif DBL64 
        mram_read((__mram_ptr void const *) (mram_base_addr_x + (*cur_colind) * sizeof(val_dt)), cache_x, 8);
        acc += (*cur_val) * cache_x[0];
#else
        if (((*cur_colind) & 1) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (*cur_colind) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[0];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*cur_colind) - 1) * sizeof(val_dt)), cache_x, 8);
            acc += (*cur_val) * cache_x[1];
        } 
#endif
        cur_rowind = seqread_get(cur_rowind, sizeof(uint32_t), &sr_rowind);
        cur_colind = seqread_get(cur_colind, sizeof(uint32_t), &sr_colind);
        cur_val = seqread_get(cur_val, sizeof(val_dt), &sr_val);
     }



    if (row_indx >= row_bound) {

        diff = prev_row - tstart_row;
#if INT8
        if ((diff & 7) == 0) {
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[0] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else if ((diff & 7) == 1) {
            diff -= 1; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[1] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else if ((diff & 7) == 2) {
            diff -= 2; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[2] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else if ((diff & 7) == 3) {
            diff -= 3; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[3] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else if ((diff & 7) == 4) {
            diff -= 4; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[4] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else if ((diff & 7) == 5) {
            diff -= 5; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[5] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else if ((diff & 7) == 6) {
            diff -= 6; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[6] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else {
            diff -= 7; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[7] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        }

#elif INT16
        if ((diff & 3) == 0) {
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[0] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else if ((diff & 3) == 1) {
            diff -= 1; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[1] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else if ((diff & 3) == 2) {
            diff -= 2; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[2] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else {
            diff -= 3; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[3] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        }

#elif INT32
        if ((diff & 1) == 0) {
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8); 
            cache_y[0] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else {
            diff -= 1; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[1] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        }

#elif INT64
        mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 3)); 
        mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
        cache_y[0] += acc;
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);

#elif FLT32
        if ((diff & 1) == 0) {
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[0] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else {
            diff -= 1; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[1] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        }

#elif DBL64 
        mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 3)); 
        mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
        cache_y[0] += acc;
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);

#else
        if ((diff & 1) == 0) {
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[0] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else {
            diff -= 1; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[1] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        }
#endif

    }

EXIT: 
    barrier_wait(&my_barrier);
    if ((tasklet_id == 0) && (is_used == 1)) {
        uint32_t t = 0;
        uint32_t iter = 8 / byte_dt;
        for(i = 0; i < NR_TASKLETS; i++) {
#if INT8
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (sync_rptr[i])); 
#elif INT16
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (sync_rptr[i] << 1)); 
#elif INT32
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (sync_rptr[i] << 2)); 
#elif INT64
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (sync_rptr[i] << 3)); 
#elif FLT32
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (sync_rptr[i] << 2)); 
#elif DBL64
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (sync_rptr[i] << 3)); 
#else
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (sync_rptr[i] << 2)); 
#endif
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            for(j=0; j < iter; j++) {
                if(sync_values[t] != 0)
                    cache_y[j] += sync_values[t]; 
                t++;
            }
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        }
    }        



#if PERF
    result->cycles = perfcounter_get();
#endif

    return 0;
}
