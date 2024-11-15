/*
 * Christina Giannoula
 * cgiannoula
 */
#ifndef _COMMON_H_
#define _COMMON_H_

#define XSTR(x) STR(x)
#define STR(x) #x

/* Size of input and output buffers */
#define BUFFER_SIZE (32 << 20)

#define PERF 1 // Use perfcounters?
#define PRINT 0
#define LOG 0

#define NB_RANKS_MAX 64
//#define INT32 1
//#define BLNC_ROW 0
//#define BLNC_NNZ 1
//#define BLNC_TSKLT_ROW 0
//#define BLNC_TSKLT_NNZ 1
//#define SYNC 1

//For COO only
// #define BLNC_TSKLT_NNZ_RGRN 0
// #define BLNC_TSKLT_NNZ 1
// #define BLNC_NNZ_RGRN 0
// #define BLNC_NNZ 1

#define LOCKFREE 0
// #define CG_LOCK 0
// #define LOCKFREEV2 1
//#define ROW_MERGE 1
//#define BLOCK_MERGE 0

// Define datatype for matrix elements
#if INT8
typedef int8_t val_dt;
#define byte_dt 1
#elif INT16
typedef int16_t val_dt;
#define byte_dt 2
#elif INT32
typedef int32_t val_dt;
#define byte_dt 4
#elif INT64
typedef int64_t val_dt;
#define byte_dt 8
#elif FLT32
typedef float val_dt;
#define byte_dt 4
#elif DBL64
typedef double val_dt;
#define byte_dt 8
#else
typedef int32_t val_dt;
#define byte_dt 4
#endif

/* Structures used by both the host and the dpu to communicate information */
typedef struct {
		uint32_t nrows;
		uint32_t max_rows;
		uint32_t max_nnz_ind;
		uint32_t max_rows_tasklet;
		uint32_t tcols;
		uint32_t dense_size;
		uint32_t dense_size_pad;
		uint32_t nnz_pad;
		uint32_t start_row[NR_TASKLETS];
		uint32_t rows_per_tasklet[NR_TASKLETS];
} dpu_arguments_t_csr;

//typedef struct {
//		uint32_t nrows;
//		uint32_t nnzs;
//		uint32_t max_rows;
//		uint32_t max_nnzs;
//		uint32_t tstart_row;
//		uint32_t tcols;
//		uint32_t dense_size;
//		uint32_t dummy;
//		uint32_t start_nnz[NR_TASKLETS];
//		uint32_t nnz_per_tasklet[NR_TASKLETS];
//} dpu_arguments_t_coo;

typedef struct {
		//uint32_t dummy;
		uint64_t cycles;
} dpu_results_t;

/**
 * @brief nnz in COO matrix format
 */
struct elem_t {
		uint32_t rowind;
		uint32_t colind;
		val_dt val;
};

#endif