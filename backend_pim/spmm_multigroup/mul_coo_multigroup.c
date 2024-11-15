/**
 * cgiannoula: christina.giann@gmail.com
 * Christina Giannoula
 */

/**
 *
 * @file spmm.c
 * @brief Host Application Source File for Sparse Matrix Vector Multiplication
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu_log.h>
#include <dpu_management.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <omp.h>

#include "spmm.h"
#include "support/partition.h"

#define DPU_CAPACITY (64 << 20) // A DPU's capacity is 64 MiB

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"



/**
 * @brief compute output in the host
 */
void spmm_host_coo(val_dt* y, struct COOMatrix *A, val_dt* x, uint32_t ncols) {
	for(unsigned int n = 0; n < A->nnz; n++) {
		uint32_t rowInd = A->rowind[n];
		uint32_t colInd = A->colind[n];
		val_dt value = A->val[n]; // Asumming that values are 1s - not used
//		printf("%d %d %d %d\n", n, rowInd, colInd, rowInd * ncols + ncols -1);
		for(unsigned int colIndx = 0; colIndx < ncols; colIndx++) {
			// elementwise addition
			y[rowInd * ncols + colIndx] += x[colInd * ncols + colIndx] * value;
		}
	}
}

//
//
///**
// * @brief compute output in the host
// */
//static void spmm_host_group(val_dt* y, struct sparse_info_group *sp_group, struct dense_info_group *ds_group) {
//	uint32_t current_Brow = 0;
//	for (uint32_t i = 0; i < sp_group->n_parts; i++){
//		struct COOMatrix *A = sp_group->sp_info[i]->A;
//		uint32_t h_size = ds_group->ds_info[0]->ncols;
//		val_dt *y_temp = (val_dt*) malloc(sizeof(val_dt) * h_size * A->nrows);
//		uint32_t current_Acol = 0;
//		for (uint32_t j = 0; j < ds_group->n_parts; j++){
//			h_size = ds_group->ds_info[j]->ncols;
//			memset(y_temp, 0, sizeof(val_dt) * h_size * A->nrows);
//			//			printf("%d %d %d\n", i, j, current_Brow);
//			fflush(stdout);
//			spmm_host(y_temp, A, ds_group->ds_info[j]->x + (current_Brow * h_size), h_size);
//			add_2D(y, y_temp, ds_group->total_cols, h_size,
//					0, current_Acol, A->nrows, h_size);
//			current_Acol += h_size;
//		}
//		current_Brow += A->ncols;
//		free(y_temp);
//	}
//}

/**
 * @brief prepare PIM execution (matrix partitioning)
 */
void prepare_pim_coo(struct coo_info *sp_info, uint32_t dense_size, uint32_t sp_group_index, uint32_t ds_group_index, uint32_t nr_of_dense_groups) {
	struct dpu_devices_t *dpu_devices = sp_info->dpu_devices;
	struct dpu_set_t dpu;
	uint32_t rank_id = sp_group_index * (nr_of_dense_groups / dpu_devices->groups_per_rank) + ds_group_index;
	uint32_t total_dpus = dpu_devices->dpus_per_rank[rank_id];
	uint32_t grp_dpus_max = dpu_devices->grp_dpus_per_rank[rank_id * dpu_devices->groups_per_rank];

	sp_info->dpu_info[ds_group_index] = (struct dpu_info_t_coo *) malloc(total_dpus * sizeof(struct dpu_info_t_coo));
	sp_info->input_args[ds_group_index] = (dpu_arguments_t_coo *) malloc(total_dpus * sizeof(dpu_arguments_t_coo));

	    uint32_t *row_split = (uint32_t *) malloc((total_dpus + 2) * sizeof(uint32_t));
	uint32_t *nnz_split_tasklet = (uint32_t *) malloc((NR_TASKLETS + 2) * sizeof(uint32_t));

	sp_info->max_rows_per_dpu[ds_group_index] = 0;
	sp_info->max_nnz_per_dpu[ds_group_index] = 0;
	sp_info->max_nnz_per_tasklet[ds_group_index] = 0;
	unsigned int i = 0;
	unsigned int acc_dpus = 0;
	for(uint32_t group_id = 0; group_id < dpu_devices->groups_per_rank; group_id++) {
		uint32_t nr_of_dpus = dpu_devices->grp_dpus_per_rank[rank_id * dpu_devices->groups_per_rank + group_id];
		uint32_t prev_row = 0;
		uint32_t prev_nnz = 0;

#if BLNC_NNZ_RGRN	
        // Balance nnz across dpus
		partition_by_nnz_rgrn_coo(sp_info->A, row_split, nr_of_dpus);
#endif
		for(i = 0; i < nr_of_dpus; i++) {
#if BLNC_NNZ_RGRN				
        	uint32_t rows_per_dpu = row_split[i+1] - row_split[i];
        	uint32_t prev_rows_dpu = row_split[i];

			uint32_t nnz_per_dpu = 0;
        	for (uint32_t r = 0; r < rows_per_dpu; r++)
            	nnz_per_dpu += sp_info->A->rows[prev_rows_dpu + r];
		
			uint32_t prev_nnz_dpu = 0;
        	for(unsigned int r = 0; r < prev_rows_dpu; r++) 
            	prev_nnz_dpu += sp_info->A->rows[r];
			
			uint32_t temp_prev_row = prev_rows_dpu;
			uint32_t temp_prev_nnz = 0;
#else			
			// Distribute NNZ to DPUs equally
			uint32_t nnz_chunks = sp_info->A->nnz / nr_of_dpus;
			uint32_t rest_nnzs = sp_info->A->nnz % nr_of_dpus;
			uint32_t nnz_per_dpu = nnz_chunks;
			uint32_t prev_nnz_dpu;

			if (i < rest_nnzs)
				nnz_per_dpu++;
			if (rest_nnzs > 0) {
				if (i >= rest_nnzs)
					prev_nnz_dpu = rest_nnzs * (nnz_chunks + 1) + (i - rest_nnzs) * nnz_chunks;
				else
					prev_nnz_dpu = i * (nnz_chunks + 1);
			} else {
				prev_nnz_dpu = i * nnz_chunks;
			}

			uint32_t prev_rows_dpu = prev_row;
			uint32_t r = prev_row;
			uint32_t temp = prev_nnz;
			uint32_t cur_nnz = 0;
			while (cur_nnz < nnz_per_dpu) {
				cur_nnz += sp_info->A->rows[r] - temp;
				temp = 0;
				r++;
			}

			// Fix Last DPU
			if (i == nr_of_dpus - 1)
			    r = sp_info->A->nrows;

			uint32_t rows_per_dpu = r - prev_row;

			// update prev_row and prev_nnz for next iteration
			uint32_t temp_prev_row = prev_row;
			uint32_t temp_prev_nnz = prev_nnz;
			prev_row = r - 1;
			prev_nnz = sp_info->A->rows[r-1] - (cur_nnz - nnz_per_dpu);
			if (cur_nnz == nnz_per_dpu) {
				prev_row = r;
				prev_nnz = 0;
			}
			else{
				// printf("overlap: %d %d %d %d\n", group_id, rank_id, acc_dpus + i, prev_row);
			}
#endif
			if (rows_per_dpu > sp_info->max_rows_per_dpu[ds_group_index])
				sp_info->max_rows_per_dpu[ds_group_index] = rows_per_dpu;

			unsigned int nnz, nnz_pad;
			nnz = nnz_per_dpu;
			if (nnz % (8 / byte_dt) != 0)
				nnz_pad = nnz + ((8 / byte_dt) - (nnz % (8 / byte_dt)));
			else
				nnz_pad = nnz;

			if (nnz_pad > sp_info->max_nnz_per_dpu[ds_group_index])
				sp_info->max_nnz_per_dpu[ds_group_index] = nnz_pad;


			sp_info->dpu_info[ds_group_index][acc_dpus + i].rows_per_dpu = rows_per_dpu;
			sp_info->dpu_info[ds_group_index][acc_dpus + i].prev_rows_dpu = prev_rows_dpu;
			sp_info->dpu_info[ds_group_index][acc_dpus + i].prev_nnz_dpu = prev_nnz_dpu;
			sp_info->dpu_info[ds_group_index][acc_dpus + i].nnz = nnz;
			sp_info->dpu_info[ds_group_index][acc_dpus + i].nnz_pad = nnz_pad;
			sp_info->dpu_info[ds_group_index][acc_dpus + i].results.cycles = 0;

			// Copy input arguments to DPU
			sp_info->input_args[ds_group_index][acc_dpus + i].nrows = rows_per_dpu;
			sp_info->input_args[ds_group_index][acc_dpus + i].nnzs = nnz;
			sp_info->input_args[ds_group_index][acc_dpus + i].tcols = sp_info->A->ncols; 
			sp_info->input_args[ds_group_index][acc_dpus + i].dense_size = dense_size; 
			sp_info->input_args[ds_group_index][acc_dpus + i].tstart_row = sp_info->dpu_info[ds_group_index][acc_dpus + i].prev_rows_dpu;

#if BLNC_TSKLT_NNZ_RGRN
			// Balance nnzs at a row granularity across tasklets 
			partition_tsklt_by_nnz_rgrn_coo(sp_info->A, temp_prev_row, temp_prev_nnz, prev_nnz, rows_per_dpu, nnz_per_dpu, nnz_split_tasklet, NR_TASKLETS);
#elif BLNC_TSKLT_NNZ
			// Balance nnz across tasklets 
			partition_tsklt_by_nnz_coo(nnz_per_dpu, nnz_split_tasklet, NR_TASKLETS);
#else
			puts("One of BLNC_TSKLT_NNZ_RGRN and BLNC_TSKLT_NNZ must be 1(COO)");
			exit(-1);
#endif
			for (uint32_t t = 0; t < NR_TASKLETS; t++) {
				sp_info->input_args[ds_group_index][acc_dpus + i].start_nnz[t] = nnz_split_tasklet[t]; 
				sp_info->input_args[ds_group_index][acc_dpus + i].nnz_per_tasklet[t] = nnz_split_tasklet[t+1] - nnz_split_tasklet[t];

				if (sp_info->input_args[ds_group_index][acc_dpus + i].nnz_per_tasklet[t] > sp_info->max_nnz_per_tasklet[ds_group_index])
					sp_info->max_nnz_per_tasklet[ds_group_index] = sp_info->input_args[ds_group_index][acc_dpus + i].nnz_per_tasklet[t];
			}

		}
		
		acc_dpus += nr_of_dpus;
	}

	// Initialization for parallel transfers 
	//if (max_rows_per_dpu % 2 != 0)
	//    max_rows_per_dpu++;
	if (sp_info->max_nnz_per_dpu[ds_group_index] % (8 / byte_dt) != 0)
		sp_info->max_nnz_per_dpu[ds_group_index] += ((8 / byte_dt) - (sp_info->max_nnz_per_dpu[ds_group_index] % (8 / byte_dt)));

	// Set Input Max Arguments
	for(i = 0; i < total_dpus; i++) {
		sp_info->input_args[ds_group_index][i].max_rows = sp_info->max_rows_per_dpu[ds_group_index]; 
		sp_info->input_args[ds_group_index][i].max_nnzs = sp_info->max_nnz_per_dpu[ds_group_index]; 
	}

	// Re-allocations
	// if (sp_info->max_nnz_per_dpu[ds_group_index] * grp_dpus_max > sp_info->A->nnz_size) {
	// 	sp_info->A->rowind = (uint32_t *) realloc(sp_info->A->rowind, (sp_info->max_nnz_per_dpu[ds_group_index] * grp_dpus_max * sizeof(uint32_t)));
	// 	sp_info->A->colind = (uint32_t *) realloc(sp_info->A->colind, (sp_info->max_nnz_per_dpu[ds_group_index] * grp_dpus_max * sizeof(uint32_t)));
	// 	sp_info->A->val = (val_dt *) realloc(sp_info->A->val, (sp_info->max_nnz_per_dpu[ds_group_index] * grp_dpus_max * sizeof(val_dt)));
	// 	sp_info->A->nnz_size = sp_info->max_nnz_per_dpu[ds_group_index] * grp_dpus_max;
	// }
	if (sp_info->max_nnz_per_dpu[ds_group_index] * grp_dpus_max > sp_info->A->nnz_size) {
		sp_info->A->rowind = (uint32_t *) pim_realloc(sp_info->A->rowind, sp_info->A->nnz_size * sizeof(uint32_t), (sp_info->max_nnz_per_dpu[ds_group_index] * grp_dpus_max * sizeof(uint32_t)));
		sp_info->A->colind = (uint32_t *) pim_realloc(sp_info->A->colind, sp_info->A->nnz_size * sizeof(uint32_t), (sp_info->max_nnz_per_dpu[ds_group_index] * grp_dpus_max * sizeof(uint32_t)));
		sp_info->A->val = (val_dt *) pim_realloc(sp_info->A->val, sp_info->A->nnz_size * sizeof(val_dt),(sp_info->max_nnz_per_dpu[ds_group_index] * grp_dpus_max * sizeof(val_dt)));
		sp_info->A->nnz_size = sp_info->max_nnz_per_dpu[ds_group_index] * grp_dpus_max;
	}

	// Total bytes in MRAM of DPU
	unsigned long int total_bytes;
	// printf("here %ld %ld\n", sp_info->max_nnz_per_dpu[ds_group_index], sp_info->max_rows_per_dpu[ds_group_index] );
	total_bytes = ((sp_info->max_nnz_per_dpu[ds_group_index]) * (2 * sizeof(uint32_t) + sizeof(val_dt))) + (sp_info->A->ncols * dense_size * sizeof(val_dt)) + (sp_info->max_rows_per_dpu[ds_group_index] * dense_size * sizeof(val_dt));
	assert(total_bytes <= DPU_CAPACITY && "Bytes needed in benchmark exceeded MRAM size");

	free(row_split);
	free(nnz_split_tasklet);
	return;
}


/**
 * @brief compute output in PIM 
 */
void spmm_pim_coo(val_dt* y, struct coo_info_group *sp_group, struct dense_info_group *ds_group) {
	struct dpu_devices_t *dpu_devices = sp_group->sp_info[0]->dpu_devices;
	struct dpu_set_t dpu, rank;
	struct Timer timer;

	uint32_t groups_per_rank = dpu_devices->groups_per_rank;

	start(&timer, 1, 0);
	unsigned int i = 0;
	unsigned int j = 0;
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		uint32_t sp_group_index = i / (ds_group->n_parts / groups_per_rank);
		uint32_t ds_group_index = i % (ds_group->n_parts / groups_per_rank);
		DPU_FOREACH(dpu_devices->ranks[i], dpu, j) {
			DPU_ASSERT(dpu_prepare_xfer(dpu,  sp_group->sp_info[sp_group_index]->input_args[ds_group_index] + j));
		}
	}
	DPU_ASSERT(dpu_push_xfer(dpu_devices->all_ranks, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(dpu_arguments_t_coo), DPU_XFER_DEFAULT));

	// Copy Matrix
	i = 0;
	j = 0;
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		uint32_t sp_group_index = i / (ds_group->n_parts / groups_per_rank);
		uint32_t ds_group_index = i % (ds_group->n_parts / groups_per_rank);
		DPU_FOREACH(dpu_devices->ranks[i], dpu, j) {
			DPU_ASSERT(dpu_prepare_xfer(dpu,  sp_group->sp_info[sp_group_index]->A->rowind +  sp_group->sp_info[sp_group_index]->dpu_info[ds_group_index][j].prev_nnz_dpu));
		}
	}
	// Move some dummy data
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		uint32_t sp_group_index = i / (ds_group->n_parts / groups_per_rank);
		uint32_t ds_group_index = i % (ds_group->n_parts / groups_per_rank);
		DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, (sp_group->sp_info[sp_group_index]->max_rows_per_dpu[ds_group_index] * ds_group->ds_info[ds_group_index]->ncols * sizeof(val_dt) +  sp_group->sp_info[sp_group_index]->A->ncols * ds_group->ds_info[ds_group_index]->ncols * sizeof(val_dt)),  sp_group->sp_info[sp_group_index]->max_nnz_per_dpu[ds_group_index] * sizeof(uint32_t), DPU_XFER_ASYNC));
	}
#if SYNC	
	DPU_ASSERT(dpu_sync(dpu_devices->all_ranks));
#endif	

	i = 0;
	j = 0;
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		uint32_t sp_group_index = i / (ds_group->n_parts / groups_per_rank);
		uint32_t ds_group_index = i % (ds_group->n_parts / groups_per_rank);
		DPU_FOREACH(dpu_devices->ranks[i], dpu, j) {
			DPU_ASSERT(dpu_prepare_xfer(dpu,  sp_group->sp_info[sp_group_index]->A->colind +  sp_group->sp_info[sp_group_index]->dpu_info[ds_group_index][j].prev_nnz_dpu));
		}
	}
	// Move some dummy data
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		uint32_t sp_group_index = i / (ds_group->n_parts / groups_per_rank);
		uint32_t ds_group_index = i % (ds_group->n_parts / groups_per_rank);
		DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, (sp_group->sp_info[sp_group_index]->max_rows_per_dpu[ds_group_index] * ds_group->ds_info[ds_group_index]->ncols * sizeof(val_dt) +  sp_group->sp_info[sp_group_index]->A->ncols * ds_group->ds_info[ds_group_index]->ncols * sizeof(val_dt) + sp_group->sp_info[sp_group_index]->max_nnz_per_dpu[ds_group_index] * sizeof(uint32_t)),  sp_group->sp_info[sp_group_index]->max_nnz_per_dpu[ds_group_index] * sizeof(uint32_t), DPU_XFER_ASYNC));
	}
#if SYNC	
	DPU_ASSERT(dpu_sync(dpu_devices->all_ranks));
#endif	

	i = 0;
	j = 0;
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		uint32_t sp_group_index = i / (ds_group->n_parts / groups_per_rank);
		uint32_t ds_group_index = i % (ds_group->n_parts / groups_per_rank);
		DPU_FOREACH(dpu_devices->ranks[i], dpu, j) {
			DPU_ASSERT(dpu_prepare_xfer(dpu,  sp_group->sp_info[sp_group_index]->A->val +  sp_group->sp_info[sp_group_index]->dpu_info[ds_group_index][j].prev_nnz_dpu));
		}
	}
	// Move some dummy data
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		uint32_t sp_group_index = i / (ds_group->n_parts / groups_per_rank);
		uint32_t ds_group_index = i % (ds_group->n_parts / groups_per_rank);
		//printf("here1 %d %d\n", sp_group_index, ds_group_index);
		DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, (sp_group->sp_info[sp_group_index]->max_rows_per_dpu[ds_group_index] * ds_group->ds_info[ds_group_index]->ncols * sizeof(val_dt) +  sp_group->sp_info[sp_group_index]->A->ncols * ds_group->ds_info[ds_group_index]->ncols * sizeof(val_dt) + sp_group->sp_info[sp_group_index]->max_nnz_per_dpu[ds_group_index] * sizeof(uint32_t) + sp_group->sp_info[sp_group_index]->max_nnz_per_dpu[ds_group_index] * sizeof(uint32_t)),  sp_group->sp_info[sp_group_index]->max_nnz_per_dpu[ds_group_index] * sizeof(val_dt), DPU_XFER_ASYNC));
	}
	DPU_ASSERT(dpu_sync(dpu_devices->all_ranks));

	stop(&timer, 1);


	start(&timer, 2, 0);
	// Copy dense matrix x - We assume dense size is a multiple of 8bytes
	i = 0;
	/*uint32_t current_Brow = 0;
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
	    uint32_t sp_group_index = i / ds_group->n_parts;
	    uint32_t ds_group_index = i % ds_group->n_parts;
	    DPU_ASSERT(dpu_broadcast_to(rank, DPU_MRAM_HEAP_POINTER_NAME, sp_group->sp_info[sp_group_index]->max_rows_per_dpu[ds_group_index] * ds_group->ds_info[ds_group_index]->ncols * sizeof(val_dt), ds_group->ds_info[ds_group_index]->x + (current_Brow * ds_group->ds_info[ds_group_index]->ncols), sp_group->sp_info[sp_group_index]->A->ncols * ds_group->ds_info[ds_group_index]->ncols * sizeof(val_dt), DPU_XFER_ASYNC));
	    if ((i+1) % ds_group->n_parts == 0) {
	  	current_Brow += sp_group->sp_info[sp_group_index]->A->ncols;
	    }
	}
	DPU_ASSERT(dpu_sync(dpu_devices->all_ranks));
	*/


	i = 0;
	j = 0;
	/*
	uint32_t current_Brow = 0;
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		uint32_t sp_group_index = i / ds_group->n_parts;
		uint32_t ds_group_index = i % ds_group->n_parts;
		DPU_FOREACH(dpu_devices->ranks[i], dpu, j) {
			DPU_ASSERT(dpu_prepare_xfer(dpu, ds_group->ds_info[ds_group_index]->x + (current_Brow * ds_group->ds_info[ds_group_index]->ncols)));
		}
		if ((i+1) % ds_group->n_parts == 0) {
			current_Brow += sp_group->sp_info[sp_group_index]->A->ncols;
		}
	}
	// Move some dummy data
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		uint32_t sp_group_index = i / ds_group->n_parts;
		uint32_t ds_group_index = i % ds_group->n_parts;
		DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, (sp_group->sp_info[sp_group_index]->max_rows_per_dpu[ds_group_index] * ds_group->ds_info[ds_group_index]->ncols * sizeof(val_dt)), sp_group->sp_info[sp_group_index]->A->ncols * ds_group->ds_info[ds_group_index]->ncols * sizeof(val_dt), DPU_XFER_ASYNC));
	}
	DPU_ASSERT(dpu_sync(dpu_devices->all_ranks));
	*/

	uint32_t current_Brow = 0;
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		uint32_t rank_id = i;
		uint32_t group_id = 0;
		uint32_t dpus_acc = 0;
		uint32_t sp_group_index = i / (ds_group->n_parts / groups_per_rank);
		DPU_FOREACH(dpu_devices->ranks[i], dpu, j) {
			uint32_t fg_ds_group_index = (i * groups_per_rank + group_id) % ds_group->n_parts;
			uint32_t dpus_in_group = dpu_devices->grp_dpus_per_rank[rank_id * groups_per_rank + group_id];

			DPU_ASSERT(dpu_prepare_xfer(dpu, ds_group->ds_info[fg_ds_group_index]->x + (current_Brow * ds_group->ds_info[fg_ds_group_index]->ncols)));
			if ((j+1) == dpus_acc + dpus_in_group) {
				dpus_acc += dpus_in_group;
				group_id++;
				//printf("j %d group_id %d sp_group_index %d fg_ds_group_index %d\n", j, group_id, sp_group_index, fg_ds_group_index);
			}
		}
		if (((i+1) * groups_per_rank) % ds_group->n_parts == 0) {
			current_Brow += sp_group->sp_info[sp_group_index]->A->ncols;
			//printf("cur_Brow increase\n");
		}
	}
	// Move some dummy data
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		uint32_t sp_group_index = i / (ds_group->n_parts / groups_per_rank);
		uint32_t ds_group_index = i % (ds_group->n_parts / groups_per_rank);
		uint32_t abs_ds_group_index = (i % (ds_group->n_parts / groups_per_rank) * groups_per_rank);
		//printf("here2 %d %d %d\n", sp_group_index, ds_group_index, abs_ds_group_index);
		DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, (sp_group->sp_info[sp_group_index]->max_rows_per_dpu[ds_group_index] * ds_group->ds_info[abs_ds_group_index]->ncols * sizeof(val_dt)), sp_group->sp_info[sp_group_index]->A->ncols * ds_group->ds_info[abs_ds_group_index]->ncols * sizeof(val_dt), DPU_XFER_ASYNC));
	}
#if SYNC	
	DPU_ASSERT(dpu_sync(dpu_devices->all_ranks));
#endif	
	

	stop(&timer, 2);
puts("load finished");fflush(stdout);

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
puts("kernel finished");fflush(stdout);
	// Retrieve results
	start(&timer, 4, 0);

	i = 0;
	j = 0;

	uint32_t max_h_size = ds_group->ds_info[0]->ncols;
	val_dt** y_temp = (val_dt **) malloc(dpu_devices->nr_of_ranks * sizeof(val_dt *));

	for (uint32_t k = 0; k < dpu_devices->nr_of_ranks; k++){
		y_temp[k] = (val_dt *) calloc(dpu_devices->dpus_per_rank[k] * sp_group->max_rows_per_dpu_all_groups * max_h_size * 2, sizeof(val_dt));
		// printf("-->%u %u %u %u \n", k, dpu_devices->nr_of_ranks, dpu_devices->dpus_per_rank[k], dpu_devices->dpus_per_rank[k] * sp_group->max_rows_per_dpu_all_groups * max_h_size);
	}

	// Copy dense matrix y
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		DPU_FOREACH(dpu_devices->ranks[i], dpu, j) {
			uint64_t offset = (uint64_t) sp_group->max_rows_per_dpu_all_groups * (uint64_t) ds_group->ds_info[0]->ncols;
			DPU_ASSERT(dpu_prepare_xfer(dpu, y_temp[i] + (j * offset)));
		}
	}

	// Move some dummy data
	i = 0;
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		uint32_t sp_group_index = i / (ds_group->n_parts / groups_per_rank);
		uint32_t ds_group_index = i % (ds_group->n_parts / groups_per_rank);
		uint32_t abs_ds_group_index = (i % (ds_group->n_parts / groups_per_rank) * groups_per_rank);
		// printf("-->%u %u %u %u \n", i, abs_ds_group_index, sp_group->max_rows_per_dpu_all_groups * max_h_size, (uint64_t) sp_group->sp_info[sp_group_index]->max_rows_per_dpu[ds_group_index] * (uint64_t) ds_group->ds_info[abs_ds_group_index]->ncols);
		// printf("%u %u %u %u \n", i, sp_group_index, ds_group_index, abs_ds_group_index);
		DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, (uint64_t) sp_group->sp_info[sp_group_index]->max_rows_per_dpu[ds_group_index] * (uint64_t) ds_group->ds_info[abs_ds_group_index]->ncols * sizeof(val_dt), DPU_XFER_ASYNC));
	}
	DPU_ASSERT(dpu_sync(dpu_devices->all_ranks)); // Need to be synchronized
 		
	stop(&timer, 4);

	// Alignment cost
	start(&timer, 5, 0);

#if BLOCK_MERGE 
	uint32_t* thd_st = (uint32_t*) malloc(sizeof(uint32_t) *  ds_group->n_parts);

	for(uint32_t sp_idx = 0; sp_idx < sp_group->n_parts; sp_idx++){
		 for (uint32_t ds_idx = 0; ds_idx < ds_group->n_parts; ds_idx++){
            uint32_t group_ids = sp_idx * ds_group->n_parts + ds_idx;
			if (group_ids % groups_per_rank == 0)
				thd_st[ds_idx] = 0;
			else
				thd_st[ds_idx] = thd_st[ds_idx - 1] + dpu_devices->grp_dpus_per_rank[group_ids-1];
        }

		uint64_t current_Bcol = 0;
		for (uint32_t ds_idx = 0; ds_idx < ds_group->n_parts; ds_idx++){
			uint64_t rank_idx = (sp_idx * ds_group->n_parts + ds_idx);
			uint64_t group_idx = rank_idx % groups_per_rank;
			rank_idx = rank_idx / groups_per_rank;
            uint64_t info_rank_idx = rank_idx % (ds_group->n_parts / groups_per_rank);
			
			uint64_t ds_ncols = ds_group->ds_info[ds_idx]->ncols;
			struct dpu_info_t_coo* dpu_info = sp_group->sp_info[sp_idx]->dpu_info[info_rank_idx];

			uint32_t thd_start = thd_st[ds_idx];
			uint32_t thd_num = dpu_devices->grp_dpus_per_rank[rank_idx * groups_per_rank + group_idx];
			
			for (int thd = thd_start; thd < thd_start + thd_num; thd++){
#if BLNC_NNZ_RGRN
                if (sp_idx ==0){
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
#else
				memadd_2D(y, y_temp[rank_idx] + (thd * sp_group->max_rows_per_dpu_all_groups * ds_ncols),
							ds_group->total_cols,  ds_ncols, //ncols
							dpu_info[thd].prev_rows_dpu, current_Bcol,  //offset
							dpu_info[thd].rows_per_dpu, ds_ncols); // len
#endif
			}
			current_Bcol += ds_ncols;
		}
	}
	free(thd_st);
	// for(uint32_t sp_idx = 0; sp_idx < sp_group->n_parts; sp_idx++){
	// 	uint64_t current_Bcol = 0;
	// 	for (uint32_t ds_idx = 0; ds_idx < ds_group->n_parts; ds_idx++){
	// 		uint64_t rank_idx = (sp_idx * ds_group->n_parts + ds_idx);
	// 		uint64_t group_idx = rank_idx % groups_per_rank;
	// 		rank_idx = rank_idx / groups_per_rank;
			
	// 		uint64_t ds_ncols = ds_group->ds_info[ds_idx]->ncols;
	// 		struct dpu_info_t_coo* dpu_info = sp_group->sp_info[sp_idx]->dpu_info[rank_idx % (ds_group->n_parts / groups_per_rank)];

	// 		uint32_t thd_start = group_idx * dpu_devices->dpus_per_rank[rank_idx] / groups_per_rank;
	// 		uint32_t thd_num = dpu_devices->grp_dpus_per_rank[rank_idx * groups_per_rank + group_idx];
			
	// 		for (int thd = thd_start; thd < thd_start + thd_num; thd++){
	// 			memadd_2D(y, y_temp[rank_idx] + (thd * sp_group->max_rows_per_dpu_all_groups * ds_ncols),
	// 						ds_group->total_cols,  ds_ncols, //ncols
	// 						dpu_info[thd].prev_rows_dpu, current_Bcol,  //offset
	// 						dpu_info[thd].rows_per_dpu, ds_ncols); // len

	// 		}
	// 		current_Bcol += ds_ncols;
	// 	}
	// }

#elif ROW_MERGE

    //Assume All groups have same size!!!!
    uint32_t* thd_now = (uint32_t*) malloc(sizeof(uint32_t) *  ds_group->n_parts);
	uint32_t* thd_end = (uint32_t*) malloc(sizeof(uint32_t) *  ds_group->n_parts);
    uint32_t* group_ids = (uint32_t*) malloc(sizeof(uint32_t) * ds_group->n_parts);
    val_dt** y_temp_base = (val_dt **) malloc(ds_group->n_parts * sizeof(val_dt *));
    struct dpu_info_t_coo** dpu_infos = (struct dpu_info_t_coo**) malloc(sizeof(struct dpu_info_t_coo*) * ds_group->n_parts);
    memset(y, 0, sp_group->total_rows * ds_group->total_cols * sizeof(val_dt));

    for(uint32_t sp_idx = 0; sp_idx < sp_group->n_parts; sp_idx++) {

        for (uint32_t ds_idx = 0; ds_idx < ds_group->n_parts; ds_idx++){
            group_ids[ds_idx] = sp_idx * ds_group->n_parts + ds_idx;
            uint32_t rank_id = group_ids[ds_idx] / groups_per_rank;
            dpu_infos[ds_idx] = sp_group->sp_info[sp_idx]->dpu_info[rank_id % (ds_group->n_parts / groups_per_rank)];

			if (group_ids[ds_idx] % groups_per_rank == 0)
				thd_now[ds_idx] = 0;
			else
				thd_now[ds_idx] = thd_now[ds_idx - 1] + dpu_devices->grp_dpus_per_rank[group_ids[ds_idx]-1];

            y_temp_base[ds_idx] = y_temp[rank_id] + (thd_now[ds_idx] * sp_group->max_rows_per_dpu_all_groups * ds_group->ds_info[ds_idx]->ncols);
			// printf("[group_info] %d: %d %d->%d\n", ds_idx, rank_id, thd_now[ds_idx], thd_now[ds_idx] + dpu_infos[ds_idx]->rows_per_dpu);
        }
		for (uint32_t ds_idx = 0; ds_idx < ds_group->n_parts; ds_idx++){
			if (group_ids[ds_idx] % groups_per_rank == groups_per_rank-1)
				thd_end[ds_idx] = dpu_devices->dpus_per_rank[group_ids[ds_idx] / groups_per_rank];
			else
				thd_end[ds_idx] = thd_now[ds_idx + 1];
		}


        for (i = 0; i < sp_group->total_rows; i++) {
            val_dt* y_base= y + (i * ds_group->total_cols);
            uint32_t current_Bcol=0;
// #pragma omp parallel for num_threads(2) shared(y_base, y_temp_base, current_Bcol, thd_now, dpu_infos, sp_group, ds_group) private(ds_idx) collapse(1)
            for (uint32_t ds_idx = 0; ds_idx < ds_group->n_parts; ds_idx++){
                // 
                uint32_t ds_ncols = ds_group->ds_info[ds_idx]->ncols;
				uint32_t rank_id = group_ids[ds_idx] / groups_per_rank;
				
                struct dpu_info_t_coo *dpu_info = &dpu_infos[ds_idx][thd_now[ds_idx]];

#if BLNC_NNZ_RGRN
                if (sp_idx == 0)
                    memcpy(y_base+current_Bcol, y_temp_base[ds_idx] + ((i - dpu_info->prev_rows_dpu)* ds_ncols), ds_ncols * sizeof(val_dt));
                else
                    memadd(y_base+current_Bcol, y_temp_base[ds_idx] + ((i - dpu_info->prev_rows_dpu)* ds_ncols), ds_ncols);
				if (dpu_info->prev_rows_dpu + dpu_info->rows_per_dpu - 1 == i &&
					thd_now[ds_idx] < thd_end[ds_idx] - 1){
					thd_now[ds_idx]++;
                    y_temp_base[ds_idx] += sp_group->max_rows_per_dpu_all_groups * ds_ncols;
				}
#else
				// memcpy(y_base+current_Bcol, y_temp_base[ds_idx] + ((i - dpu_info->prev_rows_dpu)* ds_ncols), ds_ncols * sizeof(val_dt));
				memadd(y_base+current_Bcol, y_temp_base[ds_idx] + ((i - dpu_info->prev_rows_dpu)* ds_ncols), ds_ncols);

				if (dpu_info->prev_rows_dpu + dpu_info->rows_per_dpu - 1 == i &&
					thd_now[ds_idx] < thd_end[ds_idx] - 1){
					thd_now[ds_idx]++;
                    y_temp_base[ds_idx] += sp_group->max_rows_per_dpu_all_groups * ds_ncols;

					dpu_info = &dpu_infos[ds_idx][thd_now[ds_idx]];
					if (dpu_info->prev_rows_dpu == i){
						memadd(y_base+current_Bcol, y_temp_base[ds_idx] + ((i - dpu_info->prev_rows_dpu)* ds_ncols), ds_ncols);
						// printf("overlap: %d %d %d %d\n", group_ids[ds_idx], rank_id, thd_now[ds_idx]-1, i);
					}
				}
#endif
                current_Bcol += ds_ncols;
            }
        }
    }

    free(thd_now);
	free(thd_end);
    free(group_ids);
    free(y_temp_base);
	free(dpu_infos);

#else
	
	// host data
	val_dt *y_temp_h = (val_dt*) malloc(sizeof(val_dt) * ds_group->ds_info[0]->ncols * sp_group->sp_info[0]->A->nrows);
	current_Brow = 0;

	val_dt *y_temp_d = (val_dt*) malloc(sizeof(val_dt) * ds_group->ds_info[0]->ncols * sp_group->sp_info[0]->A->nrows);
	uint32_t current_Acol = 0;
	i = 0;
	DPU_RANK_FOREACH(dpu_devices->all_ranks, rank, i) {
		uint32_t sp_group_index = i / (ds_group->n_parts / groups_per_rank);
		uint32_t ds_group_index = i % (ds_group->n_parts / groups_per_rank);
		uint32_t abs_ds_group_index = (i % (ds_group->n_parts / groups_per_rank) * groups_per_rank);

		uint32_t dpus_acc = 0;
		for(uint32_t group_id = 0; group_id < groups_per_rank; group_id++) {
			uint32_t fg_ds_group_index = (i * groups_per_rank + group_id) % ds_group->n_parts;
#if BLNC_NNZ_RGRN
            // Merge
            memset(y_temp_d, 0, sizeof(val_dt) * ds_group->ds_info[abs_ds_group_index]->ncols * sp_group->sp_info[sp_group_index]->A->nrows);
            #pragma omp parallel for num_threads(p.nthreads) shared(sp_group, ds_group, y_temp_d, y_temp, dpu_devices, i, sp_group_index, ds_group_index) private(j) 
            for(j = 0; j < dpu_devices->grp_dpus_per_rank[i * dpu_devices->groups_per_rank + group_id]; j++) {
                memcpy(&y_temp_d[sp_group->sp_info[sp_group_index]->dpu_info[ds_group_index][dpus_acc+j].prev_rows_dpu * ds_group->ds_info[abs_ds_group_index]->ncols], &y_temp[i][(dpus_acc+j) * sp_group->max_rows_per_dpu_all_groups * ds_group->ds_info[0]->ncols], sp_group->sp_info[sp_group_index]->dpu_info[ds_group_index][dpus_acc+j].rows_per_dpu * ds_group->ds_info[abs_ds_group_index]->ncols * sizeof(val_dt));
            }
            dpus_acc += dpu_devices->grp_dpus_per_rank[i * groups_per_rank + group_id];

#else
			// Merge
			memset(y_temp_d, 0, sizeof(val_dt) * ds_group->ds_info[abs_ds_group_index]->ncols * sp_group->sp_info[sp_group_index]->A->nrows);
			#pragma omp parallel for num_threads(p.nthreads) shared(sp_group, ds_group, y_temp_d, y_temp, dpu_devices, i, sp_group_index, ds_group_index) private(j) 
			for(j = 0; j < dpu_devices->grp_dpus_per_rank[i * dpu_devices->groups_per_rank + group_id]; j++) {
				if (j != 0 && sp_group->sp_info[sp_group_index]->dpu_info[ds_group_index][dpus_acc+j].prev_rows_dpu == (sp_group->sp_info[sp_group_index]->dpu_info[ds_group_index][dpus_acc+j-1].prev_rows_dpu + sp_group->sp_info[sp_group_index]->dpu_info[ds_group_index][dpus_acc+j-1].rows_per_dpu - 1)) {
					memcpy(&y_temp_d[(sp_group->sp_info[sp_group_index]->dpu_info[ds_group_index][dpus_acc+j].prev_rows_dpu + 1) * ds_group->ds_info[abs_ds_group_index]->ncols], &y_temp[i][((dpus_acc+j) * sp_group->max_rows_per_dpu_all_groups + 1) * ds_group->ds_info[0]->ncols], (sp_group->sp_info[sp_group_index]->dpu_info[ds_group_index][dpus_acc+j].rows_per_dpu - 1) * ds_group->ds_info[abs_ds_group_index]->ncols * sizeof(val_dt));
				} else {
					memcpy(&y_temp_d[sp_group->sp_info[sp_group_index]->dpu_info[ds_group_index][dpus_acc+j].prev_rows_dpu * ds_group->ds_info[abs_ds_group_index]->ncols], &y_temp[i][(dpus_acc+j) * sp_group->max_rows_per_dpu_all_groups * ds_group->ds_info[0]->ncols], sp_group->sp_info[sp_group_index]->dpu_info[ds_group_index][dpus_acc+j].rows_per_dpu * ds_group->ds_info[abs_ds_group_index]->ncols * sizeof(val_dt));
				}
			}

			#pragma omp parallel for num_threads(p.nthreads) shared(sp_group, ds_group, y_temp_d, y_temp, dpu_devices, i, sp_group_index, ds_group_index) private(j) 
			for(j = 1; j < dpu_devices->grp_dpus_per_rank[i * dpu_devices->groups_per_rank + group_id]; j++) {
				if(sp_group->sp_info[sp_group_index]->dpu_info[ds_group_index][dpus_acc+j].prev_rows_dpu == (sp_group->sp_info[sp_group_index]->dpu_info[ds_group_index][dpus_acc+j-1].prev_rows_dpu + sp_group->sp_info[sp_group_index]->dpu_info[ds_group_index][dpus_acc+j-1].rows_per_dpu - 1)) { 
					for(uint32_t d = 0; d < ds_group->ds_info[abs_ds_group_index]->ncols; d++) {
						y_temp_d[sp_group->sp_info[sp_group_index]->dpu_info[ds_group_index][dpus_acc+j].prev_rows_dpu * ds_group->ds_info[abs_ds_group_index]->ncols + d] += y_temp[i][(dpus_acc+j) * sp_group->max_rows_per_dpu_all_groups * ds_group->ds_info[0]->ncols + d];
					}
				}
			}
			dpus_acc += dpu_devices->grp_dpus_per_rank[i * groups_per_rank + group_id];
#endif



			memset(y_temp_h, 0, sizeof(val_dt) * ds_group->ds_info[abs_ds_group_index]->ncols * sp_group->sp_info[sp_group_index]->A->nrows);
			spmm_host_coo(y_temp_h, sp_group->sp_info[sp_group_index]->A, ds_group->ds_info[fg_ds_group_index]->x + (current_Brow * ds_group->ds_info[fg_ds_group_index]->ncols), ds_group->ds_info[fg_ds_group_index]->ncols);

			bool status = true;
			for (int k = 0; k < sp_group->sp_info[sp_group_index]->A->nrows * ds_group->ds_info[fg_ds_group_index]->ncols; k++) {
				if(y_temp_h[k] != y_temp_d[k]) {
					printf("block %d %d %d\n", sp_group_index, ds_group_index, fg_ds_group_index);
					status = false;
					printf("%d (%d,%d): %f -- %f\n", k, k / ds_group->ds_info[fg_ds_group_index]->ncols, k % ds_group->ds_info[fg_ds_group_index]->ncols, y_temp_h[k], y_temp_d[k]);
					break;
				}
			}
			if (status) {
				printf("[OK] Outputs are equal\n");
			} else {
				printf("[ERROR] Outputs differ!\n");
			}

			if(status == 0)
				return;

			


			add_2D(y, y_temp_d, ds_group->total_cols, ds_group->ds_info[fg_ds_group_index]->ncols, 0, current_Acol, sp_group->sp_info[sp_group_index]->A->nrows, ds_group->ds_info[fg_ds_group_index]->ncols);
			if ((i * groups_per_rank + group_id + 1) % ds_group->n_parts != 0)
				current_Acol += ds_group->ds_info[fg_ds_group_index]->ncols;
			else
				current_Acol = 0;

		}
		// for correctness
		if (((i+1) * groups_per_rank ) % ds_group->n_parts == 0) {
			current_Brow += sp_group->sp_info[sp_group_index]->A->ncols;
			printf("cur_Brow increase\n");
		}



		/*add_2D(y, y_temp_d, ds_group->total_cols, ds_group->ds_info[ds_group_index]->ncols, 0, current_Acol, sp_group->sp_info[sp_group_index]->A->nrows, ds_group->ds_info[ds_group_index]->ncols);
		if ((i+1) % ds_group->n_parts != 0)
			current_Acol += ds_group->ds_info[i%ds_group->n_parts]->ncols;
		else
			current_Acol = 0;
		*/

	}
	

	free(y_temp_d);
	free(y_temp_h);


#endif
	stop(&timer, 5);

	print_results(&timer);

	return;
}
