#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <stdio.h>
#include "common.h"

/**
 * @brief COO matrix format 
 */
struct COOMatrixMerged {
    uint32_t nrows;
    uint32_t ncols;
    uint32_t nnz;
    uint32_t *rows;
    struct elem_t *nnzs;
    uint32_t nnz_size;
};

/**
 * @brief COO matrix format 
 */
struct COOMatrix {
    uint32_t nrows;
    uint32_t ncols;
    uint32_t nnz;
    uint32_t *rows;
    uint32_t *rowind;
    uint32_t *colind;
    val_dt *val;
    uint32_t nnz_size;
};

#endif
