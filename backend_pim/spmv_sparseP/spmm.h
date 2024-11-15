#ifndef _SPMM_H_
#define _SPMM_H_

#include "support/common.h"
#include "support/matrix.h"
#include "support/timer.h"
#include "dpu.h"



struct triplet {
		uint32_t rank;
		uint32_t ci;
		uint32_t dpu;
};

struct dpu_devices_t {
		unsigned int *dpus_per_rank;
		unsigned int *rank_mram_offset;
		unsigned int nr_of_dpus;
		unsigned int nr_of_ranks;
		struct dpu_set_t all_ranks;
		struct dpu_set_t *ranks;
		struct triplet *dpus;
};

struct dpu_info_t_coo {
		uint32_t rows_per_dpu;
		uint32_t prev_rows_dpu;
		uint32_t prev_nnz_dpu;
		uint32_t nnz;
		uint32_t nnz_pad;
		dpu_results_t results;
};


struct coo_info{
		struct dpu_info_t_coo **dpu_info;
		dpu_arguments_t_coo **input_args;
		struct dpu_devices_t *dpu_devices;
		uint64_t *max_rows_per_dpu;
		uint64_t *max_nnz_per_dpu;
		uint64_t *max_nnz_per_tasklet;
		struct COOMatrix *A;
};

struct dense_info{
		val_dt *x;
		uint32_t nrows;
		uint32_t ncols;
};



struct coo_info_group {
		struct coo_info **sp_info;
		uint32_t total_rows;
		uint32_t total_cols;
		uint32_t h_size;
		uint32_t n_parts;
		uint32_t dense_parts;
		uint32_t *dense_ncols;
		uint32_t max_rows_per_dpu_all_groups;
		dpu_arguments_t_coo *dummy_input_args;
		uint32_t *rank_to_dpu_acc;
		uint32_t *rank_to_ds_group_id;
		uint32_t ranks_per_spmv;
};

//share same number of rows.(column partition)
struct dense_info_group{
		struct dense_info ** ds_info;
		uint32_t total_rows;
		uint32_t total_cols;
		uint32_t n_parts;
};
//void add_2D(val_dt* A, val_dt* B, uint32_t A_ncols, uint32_t B_ncols, uint32_t off_x, uint32_t off_y, uint32_t len_x, uint32_t len_y);
//inline void memadd(val_dt* dest, val_dt* src, uint32_t size);
//void memadd_2D(val_dt* dest, val_dt* src, uint32_t dest_ncols, uint32_t src_ncols, uint32_t off_x, uint32_t off_y, uint32_t len_x, uint32_t len_y);
//void memcpy_2D(val_dt* dest, val_dt* src, uint32_t dest_ncols, uint32_t src_ncols, uint32_t off_x, uint32_t off_y, uint32_t len_x, uint32_t len_y);
void *pim_realloc(void *__ptr, size_t old_size, size_t new_size);
void print_results(struct Timer *timer);

void spmm_host_coo(val_dt* y, struct COOMatrix *A, val_dt* x, uint32_t ncols);
void add_2D(val_dt* A, val_dt* B, uint32_t A_ncols, uint32_t B_ncols, uint32_t off_x, uint32_t off_y, uint32_t len_x, uint32_t len_y);
void prepare_pim_coo(struct coo_info *sp_info, uint32_t dense_size, uint32_t sp_group_index, uint32_t ds_group_index, uint32_t nr_of_dense_groups, uint32_t ranks_per_spmv);
void spmv_pim_coo(val_dt* y, struct coo_info_group *sp_group, struct dense_info_group *ds_group);
void copy_sparse_coo(struct coo_info_group *sp_group, uint32_t ds_parts);



#endif