#include "ops.hpp"
#include "utils.hpp"
#include "torch/torch.h"
#include "iostream"


class dpu_instance{
public:
		struct dpu_devices_t *dpu_devices = nullptr;
		uint32_t nr_of_dpus;

		void print_info(){
#if BLNC_ROW
			puts("BLNC = BLNC_ROW");
#elif BLNC_NNZ
			puts("BLNC = BLNC_NNZ");
#elif BLNC_NNZ_RGRN
			puts("BLNC = BLNC_NNZ_RGRN");
#else
			puts("BLNC = None");
#endif

#if SYNC
			puts("SYNC = True");
#else
			puts("SYNC = False");
#endif

#if BLNC_TSKLT_ROW
			puts("BLNC_TSKLT = BLNC_TSKLT_ROW");
#elif BLNC_TSKLT_NNZ
			puts("BLNC_TSKLT = BLNC_TSKLT_NNZ");
#elif BLNC_TSKLT_NNZ_RGRN
			puts("BLNC_TSKLT = BLNC_TSKLT_NNZ_RGRN");
#else
			puts("BLNC_TSKLT = None");
#endif

#if CG_LOCK
			puts("LOCK = CG_LOCK");
#elif LOCKFREEV2
			puts("LOCK = LOCKFREEV2");
#else
			puts("LOCK = None");
#endif

#if BLOCK_MERGE
			puts("MERGE = BLOCK");
#elif ROW_MERGE
			puts("MERGE = ROW");
#else
			puts("MERGE = None");
#endif

#ifdef PIM_SEQREAD_CACHE_SIZE
			printf("PIM_SEQREAD_CACHE_SIZE=%d\n", PIM_SEQREAD_CACHE_SIZE);
#endif

#if INT8
			puts("val_dt = INT8");
#elif INT16
			puts("val_dt = INT16");
#elif INT32
			puts("val_dt = INT32");
#elif INT64
			puts("val_dt = INT64");
#elif FLT32
			puts("val_dt = FLT32");
#elif DBL64
			puts("val_dt = DBL64");
#else
			puts("val_dt = INT32");
#endif
		}

		uint32_t get_nr_dpu(int64_t nr_ranks){

			struct dpu_set_t dpus, rank;
			uint32_t nr_dpu_per_rank[64], nr_dpus = 0, curr_rank = 0;
			memset(nr_dpu_per_rank, 0, sizeof(uint32_t) * 64);

			DPU_ASSERT(dpu_alloc(nr_ranks * 64, nullptr, &(dpus)));
			DPU_RANK_FOREACH (dpus, rank, curr_rank) {
				dpu_get_nr_dpus(rank, &nr_dpu_per_rank[curr_rank]);
			}
			DPU_ASSERT(dpu_free(dpus));

			for(int i = 0; i < nr_ranks; i++)
				nr_dpus += nr_dpu_per_rank[i];
			return nr_dpus;

		}

		void init(int64_t nr_dpus = 0, int64_t nr_ranks = 0){
//			printf("%d, %d\n", nr_dpus, nr_ranks);
			if (nr_dpus == 0){
				assert(nr_ranks > 0);
				nr_dpus = get_nr_dpu(nr_ranks);
			}
			else
				assert(nr_ranks == 0);

			dpu_devices = (struct dpu_devices_t *) malloc(sizeof(struct dpu_devices_t));;
			printf("Loading kernel from: %s\n", SPMM_KERNEL_BINARY);
			printf("DPU allocation start: %d dpus\n", nr_dpus);
			fflush(stdout);
			DPU_ASSERT(dpu_alloc(nr_dpus, nullptr, &(dpu_devices->all_ranks)));
			DPU_ASSERT(dpu_load(dpu_devices->all_ranks, SPMM_KERNEL_BINARY, NULL));
			DPU_ASSERT(dpu_get_nr_dpus(dpu_devices->all_ranks, &nr_of_dpus));
			printf("DPU allocation finished\n");
			fflush(stdout);

			dpu_devices->dpus_per_rank = (unsigned int *) malloc(NB_RANKS_MAX * sizeof(unsigned int));
			dpu_devices->rank_mram_offset = (unsigned int *) malloc(NB_RANKS_MAX * sizeof(unsigned int));
			dpu_devices->ranks = (struct dpu_set_t *) malloc(NB_RANKS_MAX * sizeof(struct dpu_set_t));
			dpu_devices->nr_of_dpus = 0;
			dpu_devices->nr_of_ranks = 0;
			unsigned int curr_rank;
			struct dpu_set_t rank;
			DPU_RANK_FOREACH (dpu_devices->all_ranks, rank, curr_rank) {
				dpu_devices->ranks[curr_rank] = rank;
				DPU_ASSERT(dpu_get_nr_dpus(rank, &dpu_devices->dpus_per_rank[curr_rank]));
				dpu_devices->rank_mram_offset[curr_rank] = dpu_devices->nr_of_dpus;
				dpu_devices->nr_of_dpus += dpu_devices->dpus_per_rank[curr_rank];
				dpu_devices->nr_of_ranks++;
//				printf("%u \n", dpu_devices->dpus_per_rank[curr_rank]);
			}
			printf("%u DPUs are allocated in %u ranks\n", dpu_devices->nr_of_dpus, dpu_devices->nr_of_ranks);
			printf("Allocated %d TASKLET(s) per DPU\n", NR_TASKLETS);
			print_info();
			fflush(stdout);
			assert(dpu_devices->nr_of_dpus == nr_of_dpus);

			// Find rank, ci, dpu ids for each dpu
			struct dpu_set_t dpu;
			unsigned int curr_dpu = 0;
			dpu_devices->dpus = (struct triplet *) malloc(dpu_devices->nr_of_dpus * sizeof(struct triplet));
			DPU_FOREACH (dpu_devices->all_ranks, dpu, curr_dpu) {
				struct dpu_t *temp_dpu = dpu_from_set(dpu);
				dpu_devices->dpus[curr_dpu].rank = dpu_get_rank_id(dpu_get_rank(temp_dpu));
				dpu_devices->dpus[curr_dpu].ci = dpu_get_slice_id(temp_dpu);
				dpu_devices->dpus[curr_dpu].dpu = dpu_get_member_id(temp_dpu);
			}
		}

		void free_dpu_devices() const{
			free(dpu_devices->dpus_per_rank);
			free(dpu_devices->rank_mram_offset);
			free(dpu_devices->ranks);
			free(dpu_devices->dpus);
			DPU_ASSERT(dpu_free(dpu_devices->all_ranks));
			free(dpu_devices);
		}

}dpus;

std::vector<int64_t> dpu_init_ranks(int64_t nr_ranks){
	dpus.init(0, nr_ranks);
	auto dpus_per_rank = std::vector<int64_t>();
	unsigned int curr_rank;
	struct dpu_set_t rank;
	DPU_RANK_FOREACH (dpus.dpu_devices->all_ranks, rank, curr_rank) {
		dpus_per_rank.push_back(dpus.dpu_devices->dpus_per_rank[curr_rank]);
	}
	return dpus_per_rank;
}

std::vector<int64_t> dpu_init_dpus(int64_t nr_dpus){
	dpus.init(nr_dpus, 0);
	auto dpus_per_rank = std::vector<int64_t>();
	unsigned int curr_rank;
	struct dpu_set_t rank;
	DPU_RANK_FOREACH (dpus.dpu_devices->all_ranks, rank, curr_rank) {
		dpus_per_rank.push_back(dpus.dpu_devices->dpus_per_rank[curr_rank]);
	}
	return dpus_per_rank;
}

void dpu_release(){
	dpus.free_dpu_devices();
}


//
//
//
//void free_sp_group(struct csr_info_group* sp_group){
//	for (uint32_t i = 0; i < sp_group->n_parts; i++){
//		struct csr_info *sp_info = sp_group->sp_info[i];
//		freeCSRMatrix(sp_info->A);
//		for (uint32_t j = 0; j < sp_group->n_parts; j++){
//			free(sp_info->dpu_info[j]);
//			free(sp_info->input_args[j]);
//		}
//		free(sp_info->max_rows_per_dpu);
//		free(sp_info->max_nnz_ind_per_dpu);
//		free(sp_info->max_nnz_val_per_dpu);
//		free(sp_info->max_rows_per_tasklet);
//		free(sp_info);
//	}
//	free(sp_group);
//	return;
//}

void free_ds_group(struct dense_info_group* ds_group){
	for (uint32_t i = 0; i < ds_group->n_parts; i++){
		struct dense_info *info = ds_group->ds_info[i];
		delete info;
	}
	delete ds_group;
	return;
}

//
//void spmm_free_group(int64_t sp_group_ptr){
//	csr_info_group* sp_group = (csr_info_group*)sp_group_ptr;
//	free_sp_group(sp_group);
//}


int64_t spmm_csr_to_device_group(std::vector<torch::Tensor> row_indices,
                                 std::vector<torch::Tensor> col_indices,
                                 std::vector<torch::Tensor> values,
                                 std::vector<int64_t> nrows,
                                 std::vector<int64_t> ncols,
                                 std::vector<torch::Tensor> dense_cols,
                                 int64_t h_size
																 ) {
	puts("spmm_csr_to_device_group");
	uint32_t n_parts = ncols.size();
	assert(n_parts == nrows.size());
	csr_info_group* sp_group = new csr_info_group;
	sp_group->n_parts = n_parts;
	sp_group->h_size = h_size;
	sp_group->total_cols = 0;
	sp_group->total_rows = nrows[0];
	sp_group->sp_info = new csr_info*[n_parts];
//	sp_group->dense_parts = dense_cols.size();
	sp_group->dense_ncols = new uint32_t*[sp_group->n_parts];
	uint32_t max_h_size = 0;
	uint32_t lcm_ncols = 8 / byte_dt;
	for (int i = 0; i < n_parts; i++){
		auto info = sp_group->sp_info[i] = new csr_info;
		sp_group->dense_ncols[i] = (uint32_t*)dense_cols[i].data_ptr<int32_t>();
		CSRMatrix *A = info->A = new CSRMatrix;
		info->dpu_devices = dpus.dpu_devices;
		info->dense_parts = dpus.dpu_devices->dpus_per_rank[i];
		A->nrows = nrows[i];
		A->ncols = ncols[i];
		A->nnz = values[i].size(0);
		A->rowptr = (uint32_t*)row_indices[i].data_ptr<int32_t>();
		A->colind = (uint32_t*)col_indices[i].data_ptr<int32_t>();
		A->values = values[i].data_ptr<val_dt>();
		A->rowptr_size = row_indices[i].size(0);
		A->colind_size = col_indices[i].size(0);
		A->values_size = values[i].size(0);
		sp_group->total_cols += A->ncols;
	}
	puts("prepare_pim start");
	spmm_csr_sparse_to_device_group(sp_group, dpus.dpu_devices);
	int64_t sp_group_ptr =  reinterpret_cast<std::uintptr_t>(sp_group);
	puts("prepare_pim finish");
	return sp_group_ptr;
}




torch::Tensor spmm_csr_run_group(int64_t sp_group_ptr,
                                 std::vector<torch::Tensor> B_parts){
	csr_info_group* sp_group = (csr_info_group*)sp_group_ptr;
	puts("start spmm_csr_run_group");
	assert(sp_group->dense_parts == B_parts.size());
	dense_info_group **ds_group = new dense_info_group*[sp_group->n_parts];
	uint32_t B_off = 0;
	
	for (int i = 0; i < sp_group->n_parts; i++) {
		ds_group[i] = new dense_info_group;
		ds_group[i]->n_parts = dpus.dpu_devices->dpus_per_rank[i];
		ds_group[i]->total_cols = 0;
		ds_group[i]->total_rows = sp_group->sp_info[i]->A->ncols;
		ds_group[i]->ds_info = new dense_info*[ds_group[i]->n_parts];
		ds_group[i]->ncols_per_dpu_pad = B_parts[B_off].size(1);
		assert(ds_group[i]->ncols_per_dpu_pad == sp_group->ncols_per_dpu_pad);
		// printf("sp_%d: %d %d\n", i, ds_group[i]->ncols_per_dpu_pad, sp_group->sp_info[i]->input_args[0].dense_size_pad);

		for (int j = 0; j < ds_group[i]->n_parts; j++){
			auto info = ds_group[i]->ds_info[j] = new dense_info;
			info->nrows = B_parts[B_off + j].size(0);
			info->ncols = sp_group->dense_ncols[i][j]; //B_parts[B_off + j].size(1);
			info->x = B_parts[B_off + j].data_ptr<val_dt>();				
			// printf("(%d,%d) ", info->nrows * ds_group[i]->ncols_per_dpu_pad - 1, info->x[info->nrows * ds_group[i]->ncols_per_dpu_pad - 1]);
			ds_group[i]->total_cols += info->ncols;
		}
		// puts("\n------");
		B_off += ds_group[i]->n_parts;
		assert(sp_group->h_size == ds_group[i]->total_cols);
	}
	// for (int i = 0; i < sp_group->n_parts; i++)
	// 	ds_group[i]->ncols_per_dpu_pad = sp_group->ncols_per_dpu_pad;

	puts("ds_group construction finished");
	int64_t res_shape[2] = {sp_group->total_rows, sp_group->h_size};
	auto options = B_parts[0].dtype();
	torch::Tensor res = torch::zeros(c10::ArrayRef<int64_t>(res_shape, 2), options);
	val_dt *resptr= res.data_ptr<val_dt>();

	puts("spmm_pim_csr start");
	Timer timer;
	start(&timer, 0, 0);
	spmm_pim_csr(resptr, sp_group, ds_group);
	stop(&timer, 0);
	// spmm_host_csr_group(resptr, sp_group, ds_group);
	// for (int i = 0; i < sp_group->n_parts; i++)
	// 	free_ds_group(ds_group[i]);
	// delete[] ds_group;
	puts("del ds_group finished");
	printf("[DATA]backend_c_time(ms): ");
	print_time(&timer, 0, 1);
	return res;
}




TORCH_LIBRARY (pim_ops, m){
	std::cerr << "reg : pim_ops" << std::endl;
	m.def("dpu_init_ranks", &dpu_init_ranks);
	m.def("dpu_init_dpus", &dpu_init_dpus);
	m.def("dpu_release", &dpu_release);

//	m.def("spmm_free_group", &spmm_free_group);
	m.def("spmm_csr_to_device_group", &spmm_csr_to_device_group);
	m.def("spmm_csr_run_group", &spmm_csr_run_group);
//	m.def("spmm_coo_to_device_group", &spmm_coo_to_device_group);
//	m.def("spmm_coo_run_group", &spmm_coo_run_group);

	m.def("read_matrix_rowptr", &read_matrix_rowptr);
	m.def("read_matrix_colind", &read_matrix_colind);
	m.def("read_matrix_values", &read_matrix_values);
	m.def("read_matrix_nrows", &read_matrix_nrows);
	m.def("read_matrix_ncols", &read_matrix_ncols);
}