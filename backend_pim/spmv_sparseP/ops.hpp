#ifndef OPS_H
#define OPS_H

#include <stdint.h>
#include "assert.h"

//#include <dpu>
extern "C" {
#include "support/common.h"
#include "spmm.h"
#include "dpu_log.h"
#include "dpu_management.h"
}

//void spmm_add_csr_sparse_to_device(struct csr_info *sp_info, uint32_t h_size){
//	puts("run prepare_pim");
//	prepare_pim(sp_info, h_size);
//	auto rowptr = (uint32_t*) malloc((sp_info->A->nrows + 1 + sp_info->max_rows_per_dpu) * sizeof(uint32_t));
//	auto colindptr =(uint32_t*) malloc((sp_info->A->nnz + sp_info->max_nnz_ind_per_dpu) * sizeof(uint32_t));
//	auto valuesptr = (val_dt *) malloc((sp_info->A->nnz + sp_info->max_nnz_val_per_dpu) * sizeof(val_dt));
//	memcpy(rowptr, sp_info->A->rowptr, (sp_info->A->nrows + 1) * sizeof(uint32_t));
//	sp_info->A->rowptr = rowptr;
//	memcpy(colindptr, sp_info->A->colind, sp_info->A->nnz * sizeof(uint32_t));
//	sp_info->A->colind = colindptr;
//	memcpy(valuesptr, sp_info->A->values, sp_info->A->nnz * sizeof(val_dt));
//	sp_info->A->values = valuesptr;
//}

// A[i + off_x, j + off_y] += B[i][j];
//void add_2D(val_dt* A, val_dt* B, uint32_t A_ncols, uint32_t B_ncols, uint32_t off_x, uint32_t off_y, uint32_t len_x, uint32_t len_y){
////	printf("%d %d %d %d %d %d\n",A_ncols ,B_ncols, off_x, off_y, len_x, len_y );
//	fflush(stdout);
//	for (int i = 0; i < len_x; i++)
//		for (int j = 0; j < len_y; j++){
//			A[(off_x + i) * A_ncols + off_y + j] += B[i * B_ncols + j];
//		}
//}


/**
 * @brief compute output in the host
 */
static void spmm_host_coo_group(val_dt* y, struct coo_info_group *sp_group, struct dense_info_group *ds_group) {
	uint32_t current_Brow = 0;
	for (uint32_t i = 0; i < sp_group->n_parts; i++){
		struct COOMatrix *A = sp_group->sp_info[i]->A;
		uint32_t h_size = ds_group->ds_info[0]->ncols;
		val_dt *y_temp = (val_dt*) malloc(sizeof(val_dt) * h_size * A->nrows);
		uint32_t current_Acol = 0;

		for (uint32_t j = 0; j < ds_group->n_parts; j++){
			h_size = ds_group->ds_info[j]->ncols;
			memset(y_temp, 0, sizeof(val_dt) * h_size * A->nrows);
//			printf("%d %d %d\n", i, j, current_Brow);
			fflush(stdout);
			spmm_host_coo(y_temp, A, ds_group->ds_info[j]->x + (current_Brow * h_size), h_size);
			add_2D(y, y_temp, ds_group->total_cols, h_size,
			       0, current_Acol, A->nrows, h_size);
			current_Acol += h_size;
		}
		current_Brow += A->ncols;
		free(y_temp);
	}
}


void spmv_coo_sparse_to_device_group(struct coo_info_group *sp_group, struct dpu_devices_t *dpu_devices, unsigned int ranks_per_spmv){

	for (uint32_t i = 0; i < sp_group->n_parts; i++) {
		struct coo_info *info = sp_group->sp_info[i];
		sp_group->sp_info[i]->dpu_devices = dpu_devices;
		sp_group->sp_info[i]->dpu_info = (struct dpu_info_t_coo **) malloc(sp_group->dense_parts * sizeof(struct dpu_info_t_coo *));
		sp_group->sp_info[i]->input_args = (dpu_arguments_t_coo **) malloc(sp_group->dense_parts * sizeof(dpu_arguments_t_coo *));
		sp_group->sp_info[i]->max_rows_per_dpu = (uint64_t *) malloc(sp_group->dense_parts * sizeof(uint64_t));
		sp_group->sp_info[i]->max_nnz_per_dpu = (uint64_t *) malloc(sp_group->dense_parts * sizeof(uint64_t));
		sp_group->sp_info[i]->max_nnz_per_tasklet = (uint64_t *) malloc(sp_group->dense_parts * sizeof(uint64_t));
		for (uint32_t j = 0; j < sp_group->dense_parts; j++) {
			uint32_t h_size = sp_group->dense_ncols[j];
			prepare_pim_coo(info, h_size, i, j, sp_group->dense_parts, ranks_per_spmv);
		}
	}

	sp_group->max_rows_per_dpu_all_groups = 0;
	for (uint32_t i = 0; i < sp_group->n_parts; i++) {
		struct coo_info *info = sp_group->sp_info[i];
		for (uint32_t j = 0; j < sp_group->dense_parts; j++) {
			if (info->max_rows_per_dpu[j] > sp_group->max_rows_per_dpu_all_groups) {
				sp_group->max_rows_per_dpu_all_groups = info->max_rows_per_dpu[j];
			}
		}
	}

	sp_group->ranks_per_spmv = ranks_per_spmv;
	// For each SpMV group running on N ranks, store the aggregated #dpus within the SpMV group
	sp_group->rank_to_dpu_acc = (uint32_t *) malloc(dpu_devices->nr_of_ranks * sizeof(uint32_t));
	sp_group->rank_to_dpu_acc[0] = 0;
	for (uint32_t i = 1; i < dpu_devices->nr_of_ranks; i++) {
		sp_group->rank_to_dpu_acc[i] = sp_group->rank_to_dpu_acc[i - 1] + dpu_devices->dpus_per_rank[i - 1];

		if (i % ranks_per_spmv == 0) {
			sp_group->rank_to_dpu_acc[i] = 0;
		}
	}

	// Assign each rank to the corresponding ds_group partition
	sp_group->rank_to_ds_group_id = (uint32_t *) malloc(dpu_devices->nr_of_ranks * sizeof(uint32_t));
	uint32_t group_num = 0;
	for (uint32_t i = 0; i < dpu_devices->nr_of_ranks; i++) {
		sp_group->rank_to_ds_group_id[i] = group_num;
		if ((i + 1) % ranks_per_spmv == 0) {
			group_num++;
		}
	}

	// Dummy input args - FIXME
	uint32_t ranks_used = sp_group->n_parts * sp_group->dense_parts * ranks_per_spmv;
	uint32_t max_nr_of_dpus = 0;
	for (uint32_t i = ranks_used; i < dpu_devices->nr_of_ranks; i++) {
		if (dpu_devices->dpus_per_rank[i] > max_nr_of_dpus)
			max_nr_of_dpus = dpu_devices->dpus_per_rank[i];
	}
	sp_group->dummy_input_args = (dpu_arguments_t_coo *) malloc(max_nr_of_dpus * sizeof(dpu_arguments_t_coo));
	for(uint32_t i = 0; i < max_nr_of_dpus; i++) {
		sp_group->dummy_input_args[i].nrows = 0;
		sp_group->dummy_input_args[i].nnzs = 0;
		sp_group->dummy_input_args[i].max_rows = 0;
		sp_group->dummy_input_args[i].tstart_row = 0;
		sp_group->dummy_input_args[i].tcols = 0;
		sp_group->dummy_input_args[i].is_used = 0;
	}

	copy_sparse_coo(sp_group, sp_group->dense_parts);

//	// Check
//	for (uint32_t i = 0; i < dpu_devices->nr_of_ranks; i++)
//		if (i >= ranks_used)
//			printf("[Not Used %d] acc %d, group %d\n", i, sp_group->rank_to_dpu_acc[i], sp_group->rank_to_ds_group_id[i]);
//		else
//			printf("[Used %d] acc %d, group %d\n", i, sp_group->rank_to_dpu_acc[i], sp_group->rank_to_ds_group_id[i]);
}



// self = self + value
bool element_wise_add(float *self,
                      const float *value,
                      int64_t self_shape_len,
                      const int64_t *self_shape){
	int64_t len = 1;
	for (int i = 0; i < self_shape_len; i++)
		len *= self_shape[i];
	for (int i = 0; i < len; i++)
		self[i] += value[i];
	return true;
}


#endif