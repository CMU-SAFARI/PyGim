
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "torch/torch.h"
#include "iostream"
extern "C" {
#include "support/common.h"
#include "support/matrix.h"
}

/**
 * @brief read matrix from input fileName in COO format
 * @param filename to read matrix
 */
struct COOMatrix *readCOOMatrix(const char* fileName) {

	struct COOMatrix *cooMtx;
	cooMtx = (struct COOMatrix *) malloc(sizeof(struct COOMatrix));
	// Initialize fields
	FILE* fp = fopen(fileName, "r");
	uint32_t rowindx, colindx;
	uint32_t val;
	char *line;
	char *token;
	line = (char *) malloc(1000 * sizeof(char));
	int done = false;
	int i = 0;

	while(fgets(line, 1000, fp) != NULL){
		token = strtok(line, " ");

		if(token[0] == '%'){
			;
		} else if (done == false) {
			cooMtx->nrows = atoi(token);
			token = strtok(NULL, " ");
			cooMtx->ncols = atoi(token);
			token = strtok(NULL, " ");
			cooMtx->nnz = atoi(token);
			printf("[INFO]: Read matrix %s: %u rows, %u columns, %u nonzeros\n", fileName, cooMtx->nrows, cooMtx->ncols, cooMtx->nnz);
			//assert(cooMtx->ncols == cooMtx->nrows && "non-square matrix!");
			if((cooMtx->nrows % (8 / 4)) != 0) { // Padding
				cooMtx->nrows += ((8 / 4) - (cooMtx->nrows % (8 / 4)));
			}
			if((cooMtx->ncols % (8 / 4)) != 0) {
				cooMtx->ncols += ((8 / 4) - (cooMtx->ncols % (8 / 4)));
			}
			cooMtx->rowind = (uint32_t *) calloc(cooMtx->nnz, sizeof(uint32_t));
			cooMtx->colind = (uint32_t *) calloc(cooMtx->nnz, sizeof(uint32_t));
			cooMtx->val = (val_dt *) calloc(cooMtx->nnz, sizeof(val_dt));
			done = true;
		} else {
			rowindx = atoi(token);
			token = strtok(NULL, " ");
			colindx = atoi(token);
			token = strtok(NULL, " ");
			val = (uint32_t) (1);

			cooMtx->rowind[i] = rowindx - 1; // File format indexes begin at 1
			cooMtx->colind[i] = colindx - 1; // File format indexes begin at 1
			cooMtx->val[i] = val;
			i++;
		}
	}

	free(line);
	fclose(fp);
	return cooMtx;

}

/**
 * @brief deallocate matrix in COO format
 * @param matrix in COO format
 */
void freeCOOMatrix(struct COOMatrix *cooMtx) {
	free(cooMtx->rowind);
	free(cooMtx->colind);
	free(cooMtx->val);
}

/**
 * @brief convert matrix from COO to CSR format
 * @param matrix in COO format
 */
struct CSRMatrix *coo2csr(struct COOMatrix *cooMtx) {

	struct CSRMatrix *csrMtx;
	csrMtx = (struct CSRMatrix *) malloc(sizeof(struct CSRMatrix));

	csrMtx->nrows = (int64_t) cooMtx->nrows;
	csrMtx->ncols = (int64_t) cooMtx->ncols;
	csrMtx->nnz = cooMtx->nnz;
	csrMtx->rowptr = (uint32_t *) calloc((csrMtx->nrows + 2), sizeof(int32_t));
	csrMtx->colind = (uint32_t *) calloc((csrMtx->nnz + 1), sizeof(int32_t));
	csrMtx->values = (val_dt *) calloc((csrMtx->nnz + 8), sizeof(val_dt ));

	//memset(csrMtx->rowptr, 0, (csrMtx->nrows + 1) * sizeof(uint32_t));
	for(unsigned int i = 0; i < cooMtx->nnz; ++i) {
		int32_t rowIndx = cooMtx->rowind[i];
		csrMtx->rowptr[rowIndx]++;
	}

	uint32_t sumBeforeNextRow = 0;
	for(unsigned int rowIndx = 0; rowIndx < csrMtx->nrows; ++rowIndx) {
		int32_t sumBeforeRow = sumBeforeNextRow;
		sumBeforeNextRow += csrMtx->rowptr[rowIndx];
		csrMtx->rowptr[rowIndx] = sumBeforeRow;
	}
	csrMtx->rowptr[csrMtx->nrows] = sumBeforeNextRow;

	for(unsigned int i = 0; i < cooMtx->nnz; ++i) {
		int32_t rowIndx = cooMtx->rowind[i];
		int32_t nnzIndx = csrMtx->rowptr[rowIndx]++;
		csrMtx->colind[nnzIndx] = cooMtx->colind[i];
		csrMtx->values[nnzIndx] = cooMtx->val[i];
	}

	for(unsigned int rowIndx = csrMtx->nrows - 1; rowIndx > 0; --rowIndx) {
		csrMtx->rowptr[rowIndx] = csrMtx->rowptr[rowIndx - 1];
	}
	csrMtx->rowptr[0] = 0;

	return csrMtx;

}

/**
 * @brief deallocate matrix in CSR format
 * @param matrix in CSR format
 */
void freeCSRMatrix(struct CSRMatrix *csrMtx) {
	free(csrMtx->rowptr);
	free(csrMtx->colind);
	free(csrMtx->values);
}

torch::Tensor read_matrix_rowptr(std::string filename) {
	struct COOMatrix *B = readCOOMatrix(filename.c_str());
	struct CSRMatrix *A = coo2csr(B);
	auto options = torch::TensorOptions().dtype(torch::kInt32);
	torch::Tensor tharray = torch::from_blob(A->rowptr, {A->nrows+1}, options);
	return tharray;
}

torch::Tensor read_matrix_colind(std::string filename) {
	struct COOMatrix *B = readCOOMatrix(filename.c_str());
	struct CSRMatrix *A = coo2csr(B);
	auto options = torch::TensorOptions().dtype(torch::kInt32);
	torch::Tensor tharray = torch::from_blob(A->colind, {A->nnz}, options);
	return tharray;
}

torch::Tensor read_matrix_values(std::string filename) {
	struct COOMatrix *B = readCOOMatrix(filename.c_str());
	struct CSRMatrix *A = coo2csr(B);
	auto options = torch::TensorOptions().dtype(torch::kInt32);
	torch::Tensor tharray = torch::from_blob(A->values, {A->nnz}, options);
	return tharray;
}

int64_t read_matrix_nrows(std::string filename) {
	struct COOMatrix *B = readCOOMatrix(filename.c_str());
	struct CSRMatrix *A = coo2csr(B);
	return A->nrows;
}

int64_t read_matrix_ncols(std::string filename) {
	struct COOMatrix *B = readCOOMatrix(filename.c_str());
	struct CSRMatrix *A = coo2csr(B);
	return A->ncols;
}
