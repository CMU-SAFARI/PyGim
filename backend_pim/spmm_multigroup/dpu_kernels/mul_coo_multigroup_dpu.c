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
#if BLNC_TSKLT_NNZ
#if CG_LOCK
MUTEX_INIT(my_mutex);
#endif

#if LOCKFREE
uint32_t *sync_rptr; // global 
val_dt *sync_values; // global
#endif


#if LOCKFREEV2
val_dt *sync_values; // global
#endif
#endif

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
    uint32_t max_rows = DPU_INPUT_ARGUMENTS.max_rows;
    uint32_t max_nnzs = DPU_INPUT_ARGUMENTS.max_nnzs;
    uint32_t tcols = DPU_INPUT_ARGUMENTS.tcols;
    uint32_t dense_size = DPU_INPUT_ARGUMENTS.dense_size;
    uint32_t tstart_row = DPU_INPUT_ARGUMENTS.tstart_row;
    unsigned int start_nnz = DPU_INPUT_ARGUMENTS.start_nnz[tasklet_id];
    unsigned int nnz_per_tasklet = DPU_INPUT_ARGUMENTS.nnz_per_tasklet[tasklet_id];
    //printf("tasklet_id = %u, nrows = %u, max_rows = %u, max_nnz_ind = %u, tcols = %u, nnz_pad = %u\n", tasklet_id, nrows, max_rows, max_nnz_ind, tcols, nnz_pad);

    dpu_results_t *result = &DPU_RESULTS[tasklet_id];
    result->cycles = 0;

    uint32_t dense_size_byte = dense_size * byte_dt;

    // Addresses in MRAM
    uint32_t mram_base_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER);
    uint32_t mram_base_addr_x = (uint32_t) (DPU_MRAM_HEAP_POINTER + (max_rows * dense_size_byte));
    uint32_t mram_base_addr_rowind = (uint32_t) (mram_base_addr_x + (tcols * dense_size_byte));
    uint32_t mram_base_addr_colind = (uint32_t) (mram_base_addr_rowind + (max_nnzs * sizeof(uint32_t)));
    uint32_t mram_base_addr_val = (uint32_t) (mram_base_addr_colind + (max_nnzs * sizeof(uint32_t)));
     
    uint32_t i, j, k;
#if BLNC_TSKLT_NNZ_RGRN     
    val_dt *cache_x = mem_alloc(dense_size_byte); // FIXME: we assume that this is a multiple of 8
#endif
#if BLNC_TSKLT_NNZ
#if CG_LOCK
    val_dt *cache_x = mem_alloc(dense_size_byte); // FIXME: we assume that this is a multiple of 8
#endif
#if LOCKFREE
    val_dt *cache_x = mem_alloc(dense_size_byte); // FIXME: we assume that this is a multiple of 8
#endif
#endif
    val_dt *cache_acc = mem_alloc(dense_size_byte); // FIXME: we assume that this is a multiple of 8
    for (k = 0; k < dense_size; k++)
        cache_acc[k] = 0;
     
    // Initialize MRAM with 0s in the output matrix
    if(tasklet_id == 0) {
        uint32_t mram_temp_addr_y = mram_base_addr_y;
        for(i=0; i < nrows; i++) { // FIXME check if initialization by multiple threads is better or merge multiple rows as one mram_write
            mram_write(cache_acc, (__mram_ptr void *) (mram_temp_addr_y), dense_size_byte);
            mram_temp_addr_y += dense_size_byte;
        }
#if BLNC_TSKLT_NNZ
#if LOCKFREE
        sync_rptr = mem_alloc(NR_TASKLETS * 8);
        sync_values = mem_alloc(NR_TASKLETS * dense_size_byte);
#endif
#if LOCKFREEV2
        sync_values = mem_alloc(NR_TASKLETS * dense_size_byte);
#endif
#endif
    }
    barrier_wait(&my_barrier);

    if (nnz_per_tasklet == 0) {
        goto EXIT;
    }
        
    mram_base_addr_rowind += (start_nnz * sizeof(uint32_t)); // FIXME does this need to be a multiple of 8? for the sequential reader?
    mram_base_addr_colind += (start_nnz * sizeof(uint32_t));
    mram_base_addr_val += (start_nnz * sizeof(val_dt));
    seqreader_buffer_t cache_rowind = seqread_alloc();
    seqreader_t sr_rowind;
    uint32_t *current_rowind = seqread_init(cache_rowind, (__mram_ptr void *) mram_base_addr_rowind, &sr_rowind);
    uint32_t prev_row = *current_rowind;

    seqreader_buffer_t cache_colind = seqread_alloc();
    seqreader_t sr_colind;
    uint32_t *current_colind = seqread_init(cache_colind, (__mram_ptr void *) mram_base_addr_colind, &sr_colind);

    seqreader_buffer_t cache_val = seqread_alloc();
    seqreader_t sr_val;
    val_dt *current_val = seqread_init(cache_val, (__mram_ptr void *) mram_base_addr_val, &sr_val);

    mram_base_addr_y += ((prev_row - tstart_row) * dense_size_byte); 

#if BLNC_TSKLT_NNZ_RGRN 
    for (i=0; i < nnz_per_tasklet; i++) {
        if(*current_rowind != prev_row) {
            mram_write(cache_acc, (__mram_ptr void *) (mram_base_addr_y), dense_size_byte);
            mram_base_addr_y += (*current_rowind - prev_row) * dense_size_byte;
            for (k = 0; k < dense_size; k++)
                cache_acc[k] = 0;

            prev_row = *current_rowind;
        } 

        mram_read((__mram_ptr void const *) (mram_base_addr_x + (*current_colind) * dense_size_byte), cache_x, dense_size_byte);
        for (k = 0; k < dense_size; k++)
            cache_acc[k] += cache_x[k] * (*current_val);
        current_rowind = seqread_get(current_rowind, sizeof(uint32_t), &sr_rowind);
        current_colind = seqread_get(current_colind, sizeof(uint32_t), &sr_colind);
        current_val = seqread_get(current_val, sizeof(val_dt), &sr_val);
    }
    
    // Write final row
    mram_write(cache_acc, (__mram_ptr void *) (mram_base_addr_y), dense_size_byte);
#endif

#if BLNC_TSKLT_NNZ
#if CG_LOCK
    for (i=0; i < nnz_per_tasklet; i++) {
        if(*current_rowind != prev_row) {
            mutex_lock(my_mutex);
            mram_read((__mram_ptr void *) (mram_base_addr_y), cache_x, dense_size_byte);
            for (k = 0; k < dense_size; k++)
                cache_acc[k] += cache_x[k];
            mram_write(cache_acc, (__mram_ptr void *) (mram_base_addr_y), dense_size_byte);
            mutex_unlock(my_mutex);

            for (k = 0; k < dense_size; k++)
                cache_acc[k] = 0;

            mram_base_addr_y += (*current_rowind - prev_row) * dense_size_byte;
            prev_row = *current_rowind;
            break;
        } 

        mram_read((__mram_ptr void const *) (mram_base_addr_x + (*current_colind) * dense_size_byte), cache_x, dense_size_byte);
        for (k = 0; k < dense_size; k++)
            cache_acc[k] += cache_x[k] * (*current_val);
        current_rowind = seqread_get(current_rowind, sizeof(uint32_t), &sr_rowind);
        current_colind = seqread_get(current_colind, sizeof(uint32_t), &sr_colind);
        current_val = seqread_get(current_val, sizeof(val_dt), &sr_val);
    }
    
    // Write final row
    if (i == nnz_per_tasklet) {
        mutex_lock(my_mutex);
        mram_read((__mram_ptr void *) (mram_base_addr_y), cache_x, dense_size_byte);
        for (k = 0; k < dense_size; k++)
            cache_acc[k] += cache_x[k];
        mram_write(cache_acc, (__mram_ptr void *) (mram_base_addr_y), dense_size_byte);
        mutex_unlock(my_mutex);
        barrier_wait(&my_barrier);
        goto EXIT;
    } else {
        barrier_wait(&my_barrier);
    }

    for (i; i < nnz_per_tasklet; i++) {
        if(*current_rowind != prev_row) {
            mram_read((__mram_ptr void *) (mram_base_addr_y), cache_x, dense_size_byte);
            for (k = 0; k < dense_size; k++)
                cache_acc[k] += cache_x[k];
            mram_write(cache_acc, (__mram_ptr void *) (mram_base_addr_y), dense_size_byte);

            mram_base_addr_y += (*current_rowind - prev_row) * dense_size_byte;
            for (k = 0; k < dense_size; k++)
                cache_acc[k] = 0;

            prev_row = *current_rowind;
        } 

        mram_read((__mram_ptr void const *) (mram_base_addr_x + (*current_colind) * dense_size_byte), cache_x, dense_size_byte);
        for (k = 0; k < dense_size; k++)
            cache_acc[k] += cache_x[k] * (*current_val);
        current_rowind = seqread_get(current_rowind, sizeof(uint32_t), &sr_rowind);
        current_colind = seqread_get(current_colind, sizeof(uint32_t), &sr_colind);
        current_val = seqread_get(current_val, sizeof(val_dt), &sr_val);    
    }
    
    // Write final row
    mram_read((__mram_ptr void *) (mram_base_addr_y), cache_x, dense_size_byte);
    for (k = 0; k < dense_size; k++)
        cache_acc[k] += cache_x[k];
    mram_write(cache_acc, (__mram_ptr void *) (mram_base_addr_y), dense_size_byte);

#endif

#if LOCKFREE
    for (i=0; i < nnz_per_tasklet; i++) {
        if(*current_rowind != prev_row) {
            sync_rptr[tasklet_id] = prev_row - tstart_row;
            for (k = 0; k < dense_size; k++) {
                sync_values[tasklet_id * dense_size + k] = cache_acc[k];
                cache_acc[k] = 0;
            }

            mram_base_addr_y += (*current_rowind - prev_row) * dense_size_byte;
            prev_row = *current_rowind;
            break;
        } 

        mram_read((__mram_ptr void const *) (mram_base_addr_x + (*current_colind) * dense_size_byte), cache_x, dense_size_byte);
        for (k = 0; k < dense_size; k++)
            cache_acc[k] += cache_x[k] * (*current_val);
        current_rowind = seqread_get(current_rowind, sizeof(uint32_t), &sr_rowind);
        current_colind = seqread_get(current_colind, sizeof(uint32_t), &sr_colind);
        current_val = seqread_get(current_val, sizeof(val_dt), &sr_val);    
    }
    
    // Write final row
    if (i == nnz_per_tasklet) {
        sync_rptr[tasklet_id] = prev_row - tstart_row;
        for (k = 0; k < dense_size; k++)
            sync_values[tasklet_id * dense_size + k] = cache_acc[k];
        goto BAR;
    }

    for (i; i < nnz_per_tasklet; i++) {
        if(*current_rowind != prev_row) {
            mram_write(cache_acc, (__mram_ptr void *) (mram_base_addr_y), dense_size_byte);

            mram_base_addr_y += (*current_rowind - prev_row) * dense_size_byte;
            for (k = 0; k < dense_size; k++)
                cache_acc[k] = 0;

            prev_row = *current_rowind;
        } 

        mram_read((__mram_ptr void const *) (mram_base_addr_x + (*current_colind) * dense_size_byte), cache_x, dense_size_byte);
        for (k = 0; k < dense_size; k++)
            cache_acc[k] += cache_x[k] * (*current_val);
        current_rowind = seqread_get(current_rowind, sizeof(uint32_t), &sr_rowind);
        current_colind = seqread_get(current_colind, sizeof(uint32_t), &sr_colind);
        current_val = seqread_get(current_val, sizeof(val_dt), &sr_val);   
    }
    
    // Write final row
    mram_write(cache_acc, (__mram_ptr void *) (mram_base_addr_y), dense_size_byte);

BAR:
    barrier_wait(&my_barrier);
    
    // mram_base_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER + sync_rptr[tasklet_id] * dense_size_byte);
    // mram_read((__mram_ptr void const *) (mram_base_addr_y), cache_acc, dense_size_byte);
    // j = tasklet_id * dense_size;
    // for (k = 0; k < dense_size; k++)
    //     cache_acc[k] += sync_values[j + k];
    // mram_write(cache_acc, (__mram_ptr void *) (mram_base_addr_y), dense_size_byte);

    if(tasklet_id == NR_TASKLETS - 1) {
        for(i = 0; i < NR_TASKLETS; i++) {
            mram_base_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER + sync_rptr[i] * dense_size_byte);
            mram_read((__mram_ptr void const *) (mram_base_addr_y), cache_acc, dense_size_byte);
            j = i * dense_size;
            for (k = 0; k < dense_size; k++)
                cache_acc[k] += sync_values[j + k];
            mram_write(cache_acc, (__mram_ptr void *) (mram_base_addr_y), dense_size_byte);
        }

    }


#endif

#if LOCKFREEV2
    uint32_t offset = tasklet_id * dense_size;
    for (i=0; i < nnz_per_tasklet; i++) {
        if(*current_rowind != prev_row) {
            DPU_INPUT_ARGUMENTS.nnz_per_tasklet[tasklet_id] = prev_row - tstart_row;
            for (k = 0; k < dense_size; k++) {
                sync_values[offset + k] = cache_acc[k];
                cache_acc[k] = 0;
            }

            mram_base_addr_y += (*current_rowind - prev_row) * dense_size_byte;
            prev_row = *current_rowind;
            break;
        } 

        mram_read((__mram_ptr void const *) (mram_base_addr_x + (*current_colind) * dense_size_byte), &sync_values[offset], dense_size_byte);
        for (k = 0; k < dense_size; k++)
            cache_acc[k] += sync_values[offset + k] * (*current_val);
        current_rowind = seqread_get(current_rowind, sizeof(uint32_t), &sr_rowind);
        current_colind = seqread_get(current_colind, sizeof(uint32_t), &sr_colind);
        current_val = seqread_get(current_val, sizeof(val_dt), &sr_val);    
    }
    
    // Write final row
    if (i == nnz_per_tasklet) {
        DPU_INPUT_ARGUMENTS.nnz_per_tasklet[tasklet_id] = prev_row - tstart_row;
        for (k = 0; k < dense_size; k++)
            sync_values[offset + k] = cache_acc[k];
    }
    barrier_wait(&my_barrier);

    if(tasklet_id == 0) {
        uint32_t mram_base_addr_temp;
        for(uint32_t t = 0; t < NR_TASKLETS; t++) {
            mram_base_addr_temp = (uint32_t) (DPU_MRAM_HEAP_POINTER + DPU_INPUT_ARGUMENTS.nnz_per_tasklet[t] * dense_size_byte);
            mram_read((__mram_ptr void const *) (mram_base_addr_temp), cache_acc, dense_size_byte);
            j = t * dense_size;
            for (k = 0; k < dense_size; k++)
                cache_acc[k] += sync_values[j + k];
            mram_write(cache_acc, (__mram_ptr void *) (mram_base_addr_temp), dense_size_byte);
        }
        for (k = 0; k < dense_size; k++) {
            cache_acc[k] = 0;
        }
    }
    barrier_wait(&my_barrier);
    if (i == nnz_per_tasklet) 
        goto EXIT;


    for (i; i < nnz_per_tasklet; i++) {
        if(*current_rowind != prev_row) {
            mram_write(cache_acc, (__mram_ptr void *) (mram_base_addr_y), dense_size_byte);

            mram_base_addr_y += (*current_rowind - prev_row) * dense_size_byte;
            for (k = 0; k < dense_size; k++)
                cache_acc[k] = 0;

            prev_row = *current_rowind;
        } 

        mram_read((__mram_ptr void const *) (mram_base_addr_x + (*current_colind) * dense_size_byte), &sync_values[offset], dense_size_byte);
        for (k = 0; k < dense_size; k++)
            cache_acc[k] += sync_values[offset+k] * (*current_val);
        current_rowind = seqread_get(current_rowind, sizeof(uint32_t), &sr_rowind);
        current_colind = seqread_get(current_colind, sizeof(uint32_t), &sr_colind);
        current_val = seqread_get(current_val, sizeof(val_dt), &sr_val);    
    }
    
    // Write final row
    mram_read((__mram_ptr void const *) (mram_base_addr_y), &sync_values[offset], dense_size_byte);
    for (k = 0; k < dense_size; k++)
        cache_acc[k] += sync_values[offset+k];
    mram_write(cache_acc, (__mram_ptr void *) (mram_base_addr_y), dense_size_byte);



#endif
#endif

EXIT: 

#if PERF
    result->cycles = perfcounter_get();
#endif

    return 0;
}
