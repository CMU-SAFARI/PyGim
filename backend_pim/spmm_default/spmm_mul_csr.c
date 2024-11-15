/**
 * cgiannoula: christina.giann@gmail.com
 * Christina Giannoula
 */

/**
 *
 * @file spmv.c
 * @brief Host Application Source File for Sparse Matrix Vector Multiplication
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <time.h>
#include "dpu.h"
#include "spmm.h"
#include "./support/common.h"
#include "./support/timer.h"
#include "./support/matrix.h"
#include "./support/partition.h"


#define DPU_CAPACITY (64 << 20) // A DPU's capacity is 64 MiB

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"




/**
 * @brief merge 2D partitions host
 */
void add_2D(val_dt* A, val_dt* B, uint32_t A_ncols, uint32_t B_ncols, uint32_t off_x, uint32_t off_y, uint32_t len_x, uint32_t len_y) {
//	printf("%d %d %d %d %d %d\n",A_ncols ,B_ncols, off_x, off_y, len_x, len_y );
	// fflush(stdout);
	uint32_t i, j;
//#pragma omp parallel for num_threads(p.nthreads) shared(A, B, len_x, len_y, off_x, off_y, A_ncols, B_ncols) private(i, j) collapse(2)
	for (i = 0; i < len_x; i++)
		for (j = 0; j < len_y; j++){
			A[(off_x + i) * A_ncols + off_y + j] += B[i * B_ncols + j];
		}
}

/**
 * @brief Dense matrix addition, A = A + B, A and B have same size
 */
void matrix_add(val_dt* A, val_dt* B, uint32_t nrows, uint32_t ncols) {
	uint32_t i, len = nrows * ncols;
// #pragma omp parallel for num_threads(p.nthreads) shared(A, B, len) private(i) collapse(1)
	for (i = 0; i < len; i++)
		A[i] += B[i];
}

/**
 * @brief Add 2D partitions host
 */
void memadd_2D(val_dt* dest, val_dt* src, uint32_t dest_ncols, uint32_t src_ncols, uint32_t off_x, uint32_t off_y, uint32_t len_x, uint32_t len_y) {
	uint32_t i;
	dest = dest + (off_x * dest_ncols + off_y);
// #pragma omp parallel for num_threads(p.nthreads) shared(dest, src, len_x, dest_ncols, src_ncols) private(i) collapse(1)
	for (i = 0; i < len_x; i++)
		memadd(dest + (i * dest_ncols), src + (i * src_ncols), len_y);
	// memadd(dest + ((off_x + i) * dest_ncols + off_y), src + (i * src_ncols), size);

}

/**
 * @brief merge 2D partitions host
 */
void memcpy_2D(val_dt* dest, val_dt* src, uint32_t dest_ncols, uint32_t src_ncols, uint32_t off_x, uint32_t off_y, uint32_t len_x, uint32_t len_y) {
	uint32_t i, size = len_y * sizeof(val_dt);
	dest = dest + (off_x * dest_ncols + off_y);
// #pragma omp parallel for num_threads(p.nthreads) shared(dest, src, len_x, dest_ncols, src_ncols) private(i) collapse(1)
	for (i = 0; i < len_x; i++)
		memcpy(dest + (i * dest_ncols), src + (i * src_ncols), size);
	// memcpy(dest + ((off_x + i) * dest_ncols + off_y), src + (i * src_ncols), size);

}



void *pim_realloc(void *__ptr, size_t old_size, size_t new_size){
	void *new_ptr = malloc(new_size);
	memcpy(new_ptr, __ptr, old_size);
	return new_ptr;
}


/**
 * @brief compute output in the host
 */
void spmm_host_csr(val_dt* y, struct CSRMatrix *A, val_dt* x, uint32_t ncols) {

	for(unsigned int rowIndx = 0; rowIndx < A->nrows; rowIndx++) {
		for(unsigned int colIndx = 0; colIndx < ncols; colIndx++) {
			// elementwise addition
//			fflush(stdout);
			for(unsigned int i = A->rowptr[rowIndx]; i < A->rowptr[rowIndx + 1]; i++) {
				unsigned int colInd = A->colind[i];
				val_dt value = A->values[i]; // Asumming that values are 1s - not used
				y[rowIndx * ncols + colIndx] += x[colInd * ncols + colIndx];
			}
		}
	}
}

/**
 * @brief prepare PIM execution (matrix partitioning)
 */
void prepare_pim_csr(struct csr_info *sp_info, uint32_t dense_size, uint32_t sp_group_index, uint32_t ds_group_index, uint32_t nr_of_dense_groups) {
	struct dpu_devices_t *dpu_devices = sp_info->dpu_devices;
	struct dpu_set_t dpu, rank;
	uint32_t nr_of_dpus = dpu_devices->dpus_per_rank[sp_group_index * nr_of_dense_groups + ds_group_index];

	sp_info->dpu_info[ds_group_index] = (struct dpu_info_t_csr *) malloc(nr_of_dpus * sizeof(struct dpu_info_t_csr));
	sp_info->input_args[ds_group_index] = (dpu_arguments_t_csr *) malloc(nr_of_dpus * sizeof(dpu_arguments_t_csr));


	uint32_t *row_split = (uint32_t *) malloc((nr_of_dpus + 2) * sizeof(uint32_t));
	uint32_t *row_split_tasklet = (uint32_t *) malloc((NR_TASKLETS + 2) * sizeof(uint32_t));
#if BLNC_ROW
	// Balance row across dpus
    partition_by_row_csr(sp_info->A, row_split, nr_of_dpus);
#elif BLNC_NNZ
	// Balance nnz across dpus
	partition_by_nnz_csr(sp_info->A, row_split, nr_of_dpus);
#else
	printf("[ERROR]: Not find BLNC method for DPUs\n");
	exit(-1);
#endif

	sp_info->max_rows_per_dpu[ds_group_index] = 0;
	sp_info->max_nnz_ind_per_dpu[ds_group_index] = 0;
	sp_info->max_nnz_val_per_dpu[ds_group_index] = 0;
	sp_info->max_rows_per_tasklet[ds_group_index] = 0;
	unsigned int i = 0;
	unsigned int j = 0;

	i = 0;
	for(i = 0; i < nr_of_dpus; i++) {
		uint32_t rows_per_dpu = row_split[i+1] - row_split[i];
		uint32_t prev_rows_dpu = row_split[i];

		// Pad data to be transfered
		uint32_t rows_per_dpu_pad = rows_per_dpu + 1;
		if (rows_per_dpu_pad % (8 / byte_dt) != 0)
			rows_per_dpu_pad += ((8 / byte_dt) - (rows_per_dpu_pad % (8 / byte_dt)));
#if INT64 || DBL64
		if (rows_per_dpu_pad % 2 == 1)
            rows_per_dpu_pad++;
#endif
		if (rows_per_dpu_pad > sp_info->max_rows_per_dpu[ds_group_index])
			sp_info->max_rows_per_dpu[ds_group_index] = rows_per_dpu_pad;

		unsigned int nnz, nnz_ind_pad, nnz_val_pad;
		nnz = sp_info->A->rowptr[rows_per_dpu + prev_rows_dpu] - sp_info->A->rowptr[prev_rows_dpu];
		if (nnz % 2 != 0)
			nnz_ind_pad = nnz + 1;
		else
			nnz_ind_pad = nnz;
		if (nnz % (8 / byte_dt) != 0)
			nnz_val_pad = nnz + ((8 / byte_dt) - (nnz % (8 / byte_dt)));
		else
			nnz_val_pad = nnz;

#if INT64 || DBL64
		if (nnz_ind_pad % 2 == 1)
            nnz_ind_pad++;
        if (nnz_val_pad % 2 == 1)
            nnz_val_pad++;
#endif
		if (nnz_ind_pad > sp_info->max_nnz_ind_per_dpu[ds_group_index])
			sp_info->max_nnz_ind_per_dpu[ds_group_index] = nnz_ind_pad;
		if (nnz_val_pad > sp_info->max_nnz_val_per_dpu[ds_group_index])
			sp_info->max_nnz_val_per_dpu[ds_group_index] = nnz_val_pad;

		sp_info->dpu_info[ds_group_index][i].rows_per_dpu = rows_per_dpu;
		sp_info->dpu_info[ds_group_index][i].rows_per_dpu_pad = rows_per_dpu_pad;
		sp_info->dpu_info[ds_group_index][i].prev_rows_dpu = prev_rows_dpu;
		sp_info->dpu_info[ds_group_index][i].nnz = nnz;
		sp_info->dpu_info[ds_group_index][i].nnz_pad = nnz_ind_pad;
		sp_info->dpu_info[ds_group_index][i].results.cycles = 0;

		// Copy input arguments to DPU
		sp_info->input_args[ds_group_index][i].nrows = rows_per_dpu;
		sp_info->input_args[ds_group_index][i].tcols = sp_info->A->ncols;
		sp_info->input_args[ds_group_index][i].dense_size = dense_size;
		sp_info->input_args[ds_group_index][i].nnz_pad = nnz_ind_pad;

#if BLNC_TSKLT_ROW
		// Balance rows across tasklets
		partition_tsklt_by_row_csr(i, rows_per_dpu, row_split_tasklet, NR_TASKLETS);
#elif BLNC_TSKLT_NNZ
		// Balance nnz across tasklets
		partition_tsklt_by_nnz_csr(sp_info->A, i, rows_per_dpu, row_split_tasklet, nnz, prev_rows_dpu, NR_TASKLETS);
#else
		puts("One of BLNC_TSKLT_ROW and BLNC_TSKLT_NNZ must be 1(CSR)");
#endif
		uint32_t t;
		for (t = 0; t < NR_TASKLETS; t++) {
			sp_info->input_args[ds_group_index][i].start_row[t] = row_split_tasklet[t];
			sp_info->input_args[ds_group_index][i].rows_per_tasklet[t] = row_split_tasklet[t+1] - row_split_tasklet[t];

			if (sp_info->input_args[ds_group_index][i].rows_per_tasklet[t] > sp_info->max_rows_per_tasklet[ds_group_index])
				sp_info->max_rows_per_tasklet[ds_group_index] = sp_info->input_args[ds_group_index][i].rows_per_tasklet[t];
		}
	}

	// Initialization for parallel transfers
	if (sp_info->max_rows_per_dpu[ds_group_index] % 2 != 0)
		sp_info->max_rows_per_dpu[ds_group_index]++;
	if (sp_info->max_nnz_ind_per_dpu[ds_group_index] % 2 != 0)
		sp_info->max_nnz_ind_per_dpu[ds_group_index]++;
	if (sp_info->max_nnz_val_per_dpu[ds_group_index] % (8 / byte_dt) != 0)
		sp_info->max_nnz_val_per_dpu[ds_group_index] += ((8 / byte_dt) - (sp_info->max_nnz_val_per_dpu[ds_group_index] % (8 / byte_dt)));
	if (sp_info->max_rows_per_tasklet[ds_group_index] % (8 / byte_dt) != 0)
		sp_info->max_rows_per_tasklet[ds_group_index] += ((8 / byte_dt) - (sp_info->max_rows_per_tasklet[ds_group_index] % (8 / byte_dt)));

	// Set Input Max Arguments
	for(i = 0; i < nr_of_dpus; i++) {
		sp_info->input_args[ds_group_index][i].max_rows = sp_info->max_rows_per_dpu[ds_group_index];
		sp_info->input_args[ds_group_index][i].max_nnz_ind = sp_info->max_nnz_ind_per_dpu[ds_group_index];
		sp_info->input_args[ds_group_index][i].max_rows_tasklet = sp_info->max_rows_per_tasklet[ds_group_index];
	}

//	// Re-allocations
	if (sp_info->max_rows_per_dpu[ds_group_index] * nr_of_dpus > sp_info->A->rowptr_size) {
		sp_info->A->rowptr = (uint32_t *) pim_realloc(sp_info->A->rowptr, sp_info->A->rowptr_size * sizeof(uint32_t), (sp_info->max_rows_per_dpu[ds_group_index] * nr_of_dpus * sizeof(uint32_t)));
		sp_info->A->rowptr_size = sp_info->max_rows_per_dpu[ds_group_index] * nr_of_dpus;
	}
	if (sp_info->max_nnz_ind_per_dpu[ds_group_index] * nr_of_dpus > sp_info->A->colind_size) {
		sp_info->A->colind = (uint32_t *) pim_realloc(sp_info->A->colind, sp_info->A->colind_size * sizeof(uint32_t), (sp_info->max_nnz_ind_per_dpu[ds_group_index] * nr_of_dpus * sizeof(uint32_t)));
		sp_info->A->colind_size = sp_info->max_nnz_ind_per_dpu[ds_group_index] * nr_of_dpus;
	}
	if (sp_info->max_nnz_val_per_dpu[ds_group_index] * nr_of_dpus > sp_info->A->values_size) {
		sp_info->A->values = (val_dt *) pim_realloc(sp_info->A->values, sp_info->A->values_size * sizeof(val_dt), (sp_info->max_nnz_val_per_dpu[ds_group_index] * nr_of_dpus * sizeof(val_dt)));
		sp_info->A->values_size = sp_info->max_nnz_val_per_dpu[ds_group_index] * nr_of_dpus;
	}

	// Total bytes in MRAM of DPU
	unsigned long int total_bytes;
	total_bytes = ((sp_info->max_rows_per_dpu[ds_group_index]) * sizeof(uint32_t)) + (sp_info->max_nnz_ind_per_dpu[ds_group_index] * sizeof(uint32_t)) + (sp_info->A->ncols * dense_size * sizeof(val_dt)) + (sp_info->max_rows_per_dpu[ds_group_index] * dense_size * sizeof(val_dt));
	assert(total_bytes <= DPU_CAPACITY && "Bytes needed in benchmark exceeded MRAM size");

	// Free help arrays
	free(row_split);
	free(row_split_tasklet);


	return;
}

void copy_sparse_csr(struct csr_info_group *sp_group, uint32_t ds_parts){
	struct dpu_devices_t *dpu_devices = sp_group->sp_info[0]->dpu_devices;
	struct dpu_set_t dpu, rank;

	unsigned int i = 0;
	unsigned int j = 0;
//	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
//		DPU_FOREACH(dpu_devices->ranks[i], dpu, j) {
//			uint32_t sp_group_index = i / ds_parts;
//			uint32_t ds_group_index = i % ds_parts;
//			DPU_ASSERT(dpu_prepare_xfer(dpu, sp_group->sp_info[sp_group_index]->input_args[ds_group_index] + j));
//		}
//	}
//	DPU_ASSERT(dpu_push_xfer(dpu_devices->all_ranks, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(dpu_arguments_t_csr), DPU_XFER_DEFAULT));

	// Copy Rowptr
	i = 0;
	j = 0;
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		DPU_FOREACH(dpu_devices->ranks[i], dpu, j) {
			uint32_t sp_group_index = i / ds_parts;
			uint32_t ds_group_index = i % ds_parts;
			DPU_ASSERT(dpu_prepare_xfer(dpu, sp_group->sp_info[sp_group_index]->A->rowptr + sp_group->sp_info[sp_group_index]->dpu_info[ds_group_index][j].prev_rows_dpu));
		}
	}
	// Move some dummy data
	i = 0;
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		uint32_t sp_group_index = i / ds_parts;
		uint32_t ds_group_index = i % ds_parts;
		DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, (sp_group->sp_info[sp_group_index]->max_rows_per_dpu[ds_group_index] * sp_group->dense_ncols[ds_group_index] * sizeof(val_dt) + sp_group->sp_info[sp_group_index]->A->ncols * sp_group->dense_ncols[ds_group_index] * sizeof(val_dt)), sp_group->sp_info[sp_group_index]->max_rows_per_dpu[ds_group_index] * sizeof(uint32_t), DPU_XFER_ASYNC));
	}

	// Copy Colind
	i = 0;
	j = 0;
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		DPU_FOREACH(dpu_devices->ranks[i], dpu, j) {
			uint32_t sp_group_index = i / ds_parts;
			uint32_t ds_group_index = i % ds_parts;
			DPU_ASSERT(dpu_prepare_xfer(dpu, sp_group->sp_info[sp_group_index]->A->colind + sp_group->sp_info[sp_group_index]->A->rowptr[sp_group->sp_info[sp_group_index]->dpu_info[ds_group_index][j].prev_rows_dpu]));
		}
	}
	// Move some dummy data
	i = 0;
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		uint32_t sp_group_index = i / ds_parts;
		uint32_t ds_group_index = i % ds_parts;
		DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, (sp_group->sp_info[sp_group_index]->max_rows_per_dpu[ds_group_index] * sp_group->dense_ncols[ds_group_index] * sizeof(val_dt) + sp_group->sp_info[sp_group_index]->A->ncols * sp_group->dense_ncols[ds_group_index] * sizeof(val_dt) + sp_group->sp_info[sp_group_index]->max_rows_per_dpu[ds_group_index] * sizeof(uint32_t)), sp_group->sp_info[sp_group_index]->max_nnz_ind_per_dpu[ds_group_index] * sizeof(uint32_t), DPU_XFER_ASYNC));
	}

	// Copy Values
	i = 0;
	j = 0;
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		DPU_FOREACH(dpu_devices->ranks[i], dpu, j) {
			uint32_t sp_group_index = i / ds_parts;
			uint32_t ds_group_index = i % ds_parts;
			DPU_ASSERT(dpu_prepare_xfer(dpu, sp_group->sp_info[sp_group_index]->A->values + sp_group->sp_info[sp_group_index]->A->rowptr[sp_group->sp_info[sp_group_index]->dpu_info[ds_group_index][j].prev_rows_dpu]));
		}
	}
	// Move some dummy data
	i = 0;
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		uint32_t sp_group_index = i / ds_parts;
		uint32_t ds_group_index = i % ds_parts;
		DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, (sp_group->sp_info[sp_group_index]->max_rows_per_dpu[ds_group_index] * sp_group->dense_ncols[ds_group_index] * sizeof(val_dt) + sp_group->sp_info[sp_group_index]->A->ncols * sp_group->dense_ncols[ds_group_index] * sizeof(val_dt) + sp_group->sp_info[sp_group_index]->max_rows_per_dpu[ds_group_index] * sizeof(uint32_t) + sp_group->sp_info[sp_group_index]->max_nnz_ind_per_dpu[ds_group_index] * sizeof(uint32_t)), sp_group->sp_info[sp_group_index]->max_nnz_val_per_dpu[ds_group_index] * sizeof(val_dt), DPU_XFER_ASYNC));
	}
	DPU_ASSERT(dpu_sync(dpu_devices->all_ranks));
}

/**
 * @brief compute output in PIM
 */
void spmm_pim_csr(val_dt* y, struct csr_info_group *sp_group, struct dense_info_group *ds_group) {
	struct dpu_devices_t *dpu_devices = sp_group->sp_info[0]->dpu_devices;
	struct dpu_set_t dpu, rank;
	struct Timer timer;

	unsigned int i = 0;
	unsigned int j = 0;
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		DPU_FOREACH(dpu_devices->ranks[i], dpu, j) {
			uint32_t sp_group_index = i / ds_group->n_parts;
			uint32_t ds_group_index = i % ds_group->n_parts;
			DPU_ASSERT(dpu_prepare_xfer(dpu, sp_group->sp_info[sp_group_index]->input_args[ds_group_index] + j));
		}
	}
	DPU_ASSERT(dpu_push_xfer(dpu_devices->all_ranks, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(dpu_arguments_t_csr), DPU_XFER_DEFAULT));


	start(&timer, 2, 0);
	// Copy dense matrix x - We assume dense size is a multiple of 8bytes
	i = 0;
	uint32_t current_Brow = 0;
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		uint32_t sp_group_index = i / ds_group->n_parts;
		uint32_t ds_group_index = i % ds_group->n_parts;
		DPU_ASSERT(dpu_broadcast_to(rank, DPU_MRAM_HEAP_POINTER_NAME, sp_group->sp_info[sp_group_index]->max_rows_per_dpu[ds_group_index] * ds_group->ds_info[ds_group_index]->ncols * sizeof(val_dt), ds_group->ds_info[ds_group_index]->x + (current_Brow * ds_group->ds_info[ds_group_index]->ncols), sp_group->sp_info[sp_group_index]->A->ncols * ds_group->ds_info[ds_group_index]->ncols * sizeof(val_dt), DPU_XFER_ASYNC));
		if ((i+1) % ds_group->n_parts == 0) {
			current_Brow += sp_group->sp_info[sp_group_index]->A->ncols;
		}
	}
#if SYNC
	DPU_ASSERT(dpu_sync(dpu_devices->all_ranks));
#endif
	stop(&timer, 2);

	// Run kernel on DPUs
	start(&timer, 3, 0);
#if SYNC
	DPU_ASSERT(dpu_launch(dpu_devices->all_ranks, DPU_SYNCHRONOUS));
#else
	DPU_ASSERT(dpu_launch(dpu_devices->all_ranks, DPU_ASYNCHRONOUS));
#endif
	stop(&timer, 3);
#if LOG
	// Display DPU Log
    DPU_FOREACH(dpu_devices->all_ranks, dpu) {
        DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
    }
#endif

	// Retrieve results
	start(&timer, 4, 0);
	i = 0;
	j = 0;
	uint32_t max_h_size = ds_group->ds_info[0]->ncols;
	val_dt** y_temp = (val_dt **) malloc(dpu_devices->nr_of_ranks * sizeof(val_dt *));
	for (uint32_t k = 0; k < dpu_devices->nr_of_ranks; k++)
		y_temp[k] = (val_dt *) calloc(dpu_devices->dpus_per_rank[k] * sp_group->max_rows_per_dpu_all_groups * max_h_size, sizeof(val_dt));
	// Copy dense matrix y
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		DPU_FOREACH(dpu_devices->ranks[i], dpu, j) {
			uint64_t offset = (uint64_t) sp_group->max_rows_per_dpu_all_groups * (uint64_t) ds_group->ds_info[0]->ncols;
			DPU_ASSERT(dpu_prepare_xfer(dpu, y_temp[i] + (j * offset)));
		}
	}

	// Move some dummy data
	// FIXME max rows does not need to be a multiple of 2 (here) - it is only needed for copying rowptr but not for retrieve
	i = 0;
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		uint64_t sp_group_index = i / ds_group->n_parts;
		uint64_t ds_group_index = i % ds_group->n_parts;
		DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, (uint64_t) sp_group->sp_info[sp_group_index]->max_rows_per_dpu[ds_group_index] * (uint64_t) ds_group->ds_info[ds_group_index]->ncols * sizeof(val_dt), DPU_XFER_ASYNC));
	}
	DPU_ASSERT(dpu_sync(dpu_devices->all_ranks));

	stop(&timer, 4);

	puts("Result");fflush(stdout);

//	// Alignment cost
//	start(&timer, 5, 0);
//	// host data
//	val_dt *y_temp_h = (val_dt*) malloc(sizeof(val_dt) * ds_group->ds_info[0]->ncols * sp_group->sp_info[0]->A->nrows);
//	current_Brow = 0;
//
//	val_dt *y_temp_d = (val_dt*) malloc(sizeof(val_dt) * ds_group->ds_info[0]->ncols * sp_group->sp_info[0]->A->nrows);
//	uint32_t current_Acol = 0;
//	i = 0;
//	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
//		uint64_t sp_group_index = i / ds_group->n_parts;
//		uint64_t ds_group_index = i % ds_group->n_parts;
//
//		// Merge
//		memset(y_temp_d, 0, sizeof(val_dt) * ds_group->ds_info[i%ds_group->n_parts]->ncols * sp_group->sp_info[i/ds_group->n_parts]->A->nrows);
//#pragma omp parallel for num_threads(p.nthreads) shared(sp_group, ds_group, y_temp_d, y_temp, dpu_devices, i, sp_group_index, ds_group_index) private(j)
//		for(j = 0; j < dpu_devices->dpus_per_rank[i]; j++) {
//			memcpy(&y_temp_d[sp_group->sp_info[sp_group_index]->dpu_info[ds_group_index][j].prev_rows_dpu * ds_group->ds_info[ds_group_index]->ncols], &y_temp[i][j * sp_group->max_rows_per_dpu_all_groups * ds_group->ds_info[0]->ncols], sp_group->sp_info[sp_group_index]->dpu_info[ds_group_index][j].rows_per_dpu * ds_group->ds_info[ds_group_index]->ncols * sizeof(val_dt));
//		}
//
//		memset(y_temp_h, 0, sizeof(val_dt) * ds_group->ds_info[i%ds_group->n_parts]->ncols * sp_group->sp_info[i/ds_group->n_parts]->A->nrows);
//		spmm_host_csr(y_temp_h, sp_group->sp_info[i/ds_group->n_parts]->A, ds_group->ds_info[i%ds_group->n_parts]->x + (current_Brow * ds_group->ds_info[i%ds_group->n_parts]->ncols), ds_group->ds_info[i%ds_group->n_parts]->ncols);
//		if ((i+1) % ds_group->n_parts == 0)
//			current_Brow += sp_group->sp_info[i/ds_group->n_parts]->A->ncols;
//
//
//
//		bool status = true;
//		for (int k = 0; k < sp_group->sp_info[i/ds_group->n_parts]->A->nrows * ds_group->ds_info[i%ds_group->n_parts]->ncols; k++) {
//			if(y_temp_h[k] != y_temp_d[k]) {
//				printf("block %d %d\n", sp_group_index, ds_group_index);
//				status = false;
//				printf("%d (%d,%d): %f -- %f\n", k, k / ds_group->ds_info[i%ds_group->n_parts]->ncols, k % ds_group->ds_info[i%ds_group->n_parts]->ncols, y_temp_h[k], y_temp_d[k]);
//				break;
//			}
//		}
//		if (status) {
//			printf("[OK] Outputs are equal\n");
//		} else {
//			printf("[ERROR] Outputs differ!\n");
//		}
//
//		if(status == 0)
//			return;
//
//
//
//
//
//
//		add_2D(y, y_temp_d, ds_group->total_cols, ds_group->ds_info[ds_group_index]->ncols, 0, current_Acol, sp_group->sp_info[sp_group_index]->A->nrows, ds_group->ds_info[ds_group_index]->ncols);
//		if ((i+1) % ds_group->n_parts != 0)
//			current_Acol += ds_group->ds_info[i%ds_group->n_parts]->ncols;
//		else
//			current_Acol = 0;
//
//
//	}
//
//	free(y_temp_d);
//	free(y_temp_h);
//
//	stop(&timer, 5);

	// Alignment cost
	start(&timer, 5, 0);

#if BLOCK_MERGE
	for(uint32_t sp_idx = 0; sp_idx < sp_group->n_parts; sp_idx++){
		uint64_t current_Bcol = 0;

		for (uint32_t ds_idx = 0; ds_idx < ds_group->n_parts; ds_idx++){
			uint64_t rank_idx = sp_idx * ds_group->n_parts + ds_idx;
			uint64_t ds_ncols = ds_group->ds_info[ds_idx]->ncols;
			struct dpu_info_t_csr* dpu_info = sp_group->sp_info[sp_idx]->dpu_info[ds_idx];

			for (int thd = 0; thd < dpu_devices->dpus_per_rank[rank_idx]; thd++){
				if (sp_idx == 0){
					memcpy_2D(y, y_temp[rank_idx] + (thd * sp_group->max_rows_per_dpu_all_groups * ds_ncols),
				          ds_group->total_cols,  ds_ncols, //ncols
				          dpu_info[thd].prev_rows_dpu, current_Bcol,  //offset
				          dpu_info[thd].rows_per_dpu, ds_ncols); // len
				}
				else{
					memadd_2D(y, y_temp[rank_idx] + (thd * sp_group->max_rows_per_dpu_all_groups * ds_ncols),
				          ds_group->total_cols,  ds_ncols, //ncols
				          dpu_info[thd].prev_rows_dpu, current_Bcol,  //offset
				          dpu_info[thd].rows_per_dpu, ds_ncols); // len
				}
			}
			current_Bcol += ds_ncols;
		}
	}
#elif ROW_MERGE
 	uint32_t* thd_now = (uint32_t*) malloc(sizeof(uint32_t) * sp_group->n_parts * ds_group->n_parts);
 	uint32_t* rank_ids = (uint32_t*) malloc(sizeof(uint32_t) * ds_group->n_parts);
 	val_dt** y_temp_base = (val_dt **) malloc(ds_group->n_parts * sizeof(val_dt *));
 	struct dpu_info_t_csr** dpu_infos = (struct dpu_info_t_csr**) malloc(sizeof(struct dpu_info_t_csr*) * ds_group->n_parts);

 	for(uint32_t sp_idx = 0; sp_idx < sp_group->n_parts; sp_idx++) {
 		for (uint32_t ds_idx = 0; ds_idx < ds_group->n_parts; ds_idx++){
 			rank_ids[ds_idx] = sp_idx * ds_group->n_parts + ds_idx;
 			dpu_infos[ds_idx] = sp_group->sp_info[sp_idx]->dpu_info[ds_idx];
 			y_temp_base[ds_idx] = y_temp[rank_ids[ds_idx]];
 		}

 		memset(thd_now, 0, sizeof(uint32_t) * sp_group->n_parts * ds_group->n_parts);
 		for (i = 0; i < sp_group->total_rows; i++) {
 			val_dt* y_base= y + (i * ds_group->total_cols);
 			uint32_t current_Bcol=0;
 // #pragma omp parallel for num_threads(2) shared(y_base, y_temp, current_Bcol, thd_now, dpu_infos, sp_group, ds_group) private(ds_idx) collapse(1)
 			for (uint32_t ds_idx = 0; ds_idx < ds_group->n_parts; ds_idx++){
 				struct dpu_info_t_csr *dpu_info = &dpu_infos[ds_idx][thd_now[ds_idx]];
 				uint32_t ds_ncols = ds_group->ds_info[ds_idx]->ncols;
 				if (dpu_info->prev_rows_dpu + dpu_info->rows_per_dpu <= i){
 					thd_now[ds_idx]++;
 					dpu_info = &dpu_infos[ds_idx][thd_now[ds_idx]];
 					y_temp_base[ds_idx] += sp_group->max_rows_per_dpu_all_groups * ds_ncols;
 					// printf("%d %d\n", ds_idx, thd_now[ds_idx]);
 				}
 				// printf("%d %d %d\n", dpu_info->prev_rows_dpu, dpu_info->rows_per_dpu, i);
			  if (sp_idx == 0)
 				  memcpy(y_base+current_Bcol, y_temp_base[ds_idx] + ((i - dpu_info->prev_rows_dpu )* ds_ncols), ds_ncols * sizeof(val_dt));
				else
				  memadd(y_base+current_Bcol, y_temp_base[ds_idx] + ((i - dpu_info->prev_rows_dpu )* ds_ncols), ds_ncols);
 				current_Bcol += ds_ncols;
 			}
 		}
 	}

 	free(thd_now);
 	free(rank_ids);
 	free(y_temp_base);
	free(dpu_infos);
#else
	printf("[ERROR]: Not find MERGE method for results\n");
	exit(-1);
#endif


	stop(&timer, 5);

	puts("output------");
	print_results(&timer);
	puts("output------");
	return;

}

void print_results(struct Timer *timer){
	// Print timing results
	printf("\n");
//	printf("CPU ");
//	print(&timer, 0, 1);
	printf("[DATA]load_sparse_time:");
	print_time(timer, 1, 1);
	printf("[DATA]load_dense_time:");
	print_time(timer, 2, 1);
	printf("[DATA]kernel_time:");
	print_time(timer, 3, 1);
	printf("[DATA]retrieve_result_time:");
	print_time(timer, 4, 1);
	printf("[DATA]alignment_time:");
	print_time(timer, 5, 1);
	printf("\n\n");
	fflush(stdout);
}

