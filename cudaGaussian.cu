#include "stdio.h"
#include "stdlib.h"
#include "cublas.h"
#include "cusparse.h"

/* Sparse Matrix in COO form */
typedef struct sparse {
	float* data;
	long* row;
	long* col;
	long nnz; // Number of non-zero terms
} COO;

// ---------------------------------- Helper Functions 

/* Print the whole matrix */
void printM(float* A, long size) {

	long i;
	for (i = 0; i < size; i++) {
		printf("%f\n", A[i]);
	}
}

// Max 512 threads per block 
void setKernelProperty(int &threads, int &blocks, long size) {
	
	if (size <= 512) {
		threads = size;
		blocks = 1;
	} else {
		threads = 512;
		if ((size % 512) != 0) {
			blocks = size/512 + 1;
		} else {
			blocks = size/512;
		}
	}
}

// --------------------------------- CUSPARSE Helper Functions 

/* Rebuild a matrix to sparse COO form */
COO sparseBuilder(float* A, long size) {
	
	COO coo;
	long i; 
	long j = 0;
	
	/* Count the number of none-zero terms */
	for (i = 0; i < size*(size+1); i++) { // Include Vector B
		if (A[i] != 0) {
			coo.nnz++;
		}
	}
	
	/* Allocate size */
	coo.data = (float*)malloc(coo.nnz*sizeof(float));	
	coo.col = (long*)malloc(coo.nnz*sizeof(long));
	coo.row = (long*)malloc(coo.nnz*sizeof(long));
	
	/* Fill in data */
	for (i = 0; i < size*(size+1); i++) {
		if (A[i] != 0.0) {
			coo.data[j] = A[i];
			coo.col[j] = i/size;
			coo.row[j] = i%size;
			j++;
		}
	}

	return coo;
}

/* Create and Copy a COO sparse matrix into GPU space */
COO sparseGPUSetup(COO A_h) {
	
	COO A_d;
	A_d.nnz = A_h.nnz;
	
	cudaMalloc((void**)&A_d.data, A_d.nnz*sizeof(long));
	cudaMalloc((void**)&A_d.col, A_d.nnz*sizeof(long));
	cudaMalloc((void**)&A_d.row, A_d.nnz*sizeof(long));
	
	cudaMemcpy(A_d.data, A_h.data, A_d.nnz*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(A_d.col, A_h.col, A_d.nnz*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(A_d.row, A_h.row, A_d.nnz*sizeof(float), cudaMemcpyHostToDevice);
		
	return A_d;	
}

/* Clear melloc/cudaMelloc resources */
void sparseClean(COO A_h, COO A_d) {
	
	free(A_h.data); free(A_h.col); free(A_h.row);
	cudaFree(A_d.data); cudaFree(A_d.col); cudaFree(A_d.row);
}

// --------------------------------- CUDA Helper Functions 


// Divide one row with a scalor 
__global__ void pivot(float *pivot, float scaler, long size) {
	
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	if (i < size) {}
		pivot[i] /= scaler;
}


// Row sclar and multiply and add to another row 
__global__ void mulAdd(float *pivot, float *row, float scaler, long size) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	
	if (i < size) {}
		row [i] -= pivot[i] * scaler; 
}

/**********************************************************************/
/**********************************************************************/

/*
//For test perpuse
int main() {

	long unknowns = 3;

	float *h_A = (float*)malloc(unknowns*(unknowns+1)*sizeof(float)); // Add one for vector b

	h_A[0] = 2; h_A[1] = -3; h_A[2] = -2;
	h_A[3] = 1; h_A[4] = -1; h_A[5] = 1;
	h_A[6] = -1; h_A[7] = 2; h_A[8] = 2;
	h_A[9] = 8; h_A[10] = -11; h_A[11] = -3;
	
	printM(h_A, unknowns*(unknowns+1));
	
	printf("**********\n");
	
	float *h_X = (float*)malloc(unknowns*sizeof(float));
	
	navie_gaussian_cuda(h_A, h_X, unknowns);
	
	free(h_A);
	free(h_X);
}
*/

/**********************************************************************/
/**********************************************************************/

/* General Gaussian elimination */
void navie_gaussian_cublas(float* A, float* x, long size) {

	printf("Performing CUDA Forward Substitution\n");

	float scaler; // Diagonal term
	
	float *h_pivot = (float*)malloc((size+1)*sizeof(float)); // Current row
	float *d_pivot;
	
	float *h_row = (float*)malloc((size+1)*sizeof(float)); // Row below
	float *d_row;
	
	long step; // Get compute size, save time only compute upper triangle terms
	
	cublasStatus status; // Error handle cublasStatus
	cublasInit(); // Call before using cublas
	
	long col, subCol, subRow;
	
	for (col = 0; col < size; col++) { // Go through each column in A
	
		printf("Column: %ld...\n", col);
	
		step = (size+1)-col; 
		
		/* Step 1: Setup pivoting row with leading element 1 */
		scaler = A[col+col*size]; // Get the current diagonal term in A 
		
		// Fill pivot row host data
		realloc(h_pivot, step*sizeof(float)); 
		for (subCol = 0; subCol < step; subCol++) 
			h_pivot[subCol] = A[col+(subCol+col)*size]; // h start at index 0, A need mapping
		
		// Call cublasSscal()
		cublasAlloc(step, sizeof(float), (void**)&d_pivot);
		cublasSetVector(step, sizeof(float), h_pivot, 1, d_pivot, 1);
		cublasSscal(step, 1/scaler, d_pivot, 1);
		cublasGetVector(step, sizeof(float), d_pivot, 1, h_pivot, 1);
			
		// Copy back
		for (subCol = 0; subCol < step; subCol++) 
			A[col+(subCol+col)*size] = h_pivot[subCol];
			
		
		/* Step 2: Set all the element with leading element 0 */
		for (subRow = col+1; subRow < size; subRow++) {
		
			scaler = A[subRow+col*size]; // Same column but row below
			
			// Fill another row host data
			realloc(h_row, step*sizeof(float));
			for (subCol = 0; subCol < step; subCol++) 
				h_row[subCol] = A[subRow+(subCol+col)*size];
				
			// Call cublasSaxpy()
			cublasAlloc(step, sizeof(float), (void**)&d_row);
			cublasSetVector(step, sizeof(float), h_row, 1, d_row, 1);
			cublasSaxpy(step, (-1)*scaler, d_pivot, 1, d_row, 1);
			cublasGetVector(step, sizeof(float), d_row, 1, h_row, 1);
			
			// Copy back
			for (subCol = 0; subCol < step; subCol++) 
				A[subRow+(subCol+col)*size] = h_row[subCol];
				
		}
		
		cublasFree(d_pivot);
		cublasFree(d_row);
	}
	cublasShutdown();

	printf("Performing CPU Backward Substitution\n");
	x[size-1] = A[(size-1) + size*size]; // Compute the most bottom element
   	
   	long i, j;
   	
   	for (i = size-1; i >= 0; i--)	{
    	float sum = 0;
      	for (j = i+1; j < size; j++)
            sum = sum + A[i+size*j] * x[j];
      	x[i] = (A[i+size*size] - sum);
   	}    
   	
   	free(h_pivot);
   	free(h_row);
}


/* Partial Pivoting Gaussian elimination */
void partialPivot_gaussian_cublas(float* A, float* x, long size) {

	printf("Performing CUDA Forward Substitution\n");

	float scaler; // Diagonal term
	
	float *h_pivot = (float*)malloc((size+1)*sizeof(float)); // Current row
	float *d_pivot;
	
	float *h_row = (float*)malloc((size+1)*sizeof(float)); // Row below
	float *d_row;
	
	long step; // Get compute size, save time only compute upper triangle terms
	
	cublasStatus status; // Error handle cublasStatus
	cublasInit(); // Call before using cublas
	
	long col, subCol, subRow;
	
	for (col = 0; col < size; col++) { // Go through each column in A
	
		printf("Column: %ld...\n", col);
	
		step = (size+1)-col; 
		
		/* Step 1: Find the row with max element in this column and swap */
		float max = A[col+col*size];
		long maxIndex = col;
		
		for (subRow = col; subRow < size; subRow++) {	
			if (max < A[subRow+col*size]) {
				max = A[subRow+col*size];
				maxIndex = subRow;
			}
		}
		
		// Fill Data
		float *d_max, *h_max;
		float *d_current, *h_current;
		float *d_temp;
		
		h_max = (float*)malloc((size+1)*sizeof(float));
		h_current = (float*)malloc((size+1)*sizeof(float));
		
		for (subCol = 0; subCol < size+1; subCol++) {
			h_max[subCol] = A[maxIndex+subCol*size];
			h_current[subCol] = A[col+subCol*size];
		}
		
		cublasAlloc(size+1, sizeof(float), (void**)&d_max);
		cublasAlloc(size+1, sizeof(float), (void**)&d_current);
		cublasAlloc(size+1, sizeof(float), (void**)&d_temp);
		
		cublasSetVector(size+1, sizeof(float), h_max, 1, d_max, 1);
		cublasSetVector(size+1, sizeof(float), h_current, 1, d_current, 1);
		
		// Swap row, maxIndex with col
		cublasScopy(size+1, d_current, 1, d_temp, 1);
		cublasScopy(size+1, d_max, 1, d_current, 1);
		cublasScopy(size+1, d_temp, 1, d_max, 1);
		
		for (subCol = 0; subCol < size+1; subCol++) {
			A[maxIndex+subCol*size] = h_max[subCol];
			A[col+subCol*size] = h_current[subCol];
		}
		
		cublasFree(d_max); cublasFree(d_current); cublasFree(d_temp);
		free(h_max); free(h_current);
		
		/* Step 2: Setup pivoting row with leading element 1 */
		scaler = A[col+col*size]; // Get the current diagonal term in A 
		
		// Fill pivot row host data
		realloc(h_pivot, step*sizeof(float)); 
		for (subCol = 0; subCol < step; subCol++) 
			h_pivot[subCol] = A[col+(subCol+col)*size]; // h start at index 0, A need mapping
		
		// Call cublasSscal()
		cublasAlloc(step, sizeof(float), (void**)&d_pivot);
		cublasSetVector(step, sizeof(float), h_pivot, 1, d_pivot, 1);
		cublasSscal(step, 1/scaler, d_pivot, 1);
		cublasGetVector(step, sizeof(float), d_pivot, 1, h_pivot, 1);
			
		// Copy back
		for (subCol = 0; subCol < step; subCol++) 
			A[col+(subCol+col)*size] = h_pivot[subCol];
			
		
		/* Step 3: Set all the element with leading element 0 */
		for (subRow = col+1; subRow < size; subRow++) {
		
			scaler = A[subRow+col*size]; // Same column but row below
			
			// Fill another row host data
			realloc(h_row, step*sizeof(float));
			for (subCol = 0; subCol < step; subCol++) 
				h_row[subCol] = A[subRow+(subCol+col)*size];
				
			// Call cublasSaxpy()
			cublasAlloc(step, sizeof(float), (void**)&d_row);
			cublasSetVector(step, sizeof(float), h_row, 1, d_row, 1);
			cublasSaxpy(step, (-1)*scaler, d_pivot, 1, d_row, 1);
			cublasGetVector(step, sizeof(float), d_row, 1, h_row, 1);
			
			// Copy back
			for (subCol = 0; subCol < step; subCol++) 
				A[subRow+(subCol+col)*size] = h_row[subCol];
				
		}
		
		cublasFree(d_pivot);
		cublasFree(d_row);
	}
	cublasShutdown();

	printf("Performing CPU Backward Substitution\n");
	x[size-1] = A[(size-1) + size*size]; // Compute the most bottom element
   	
   	long i, j;
   	
   	for (i = size-1; i >= 0; i--)	{
    	float sum = 0;
      	for (j = i+1; j < size; j++)
            sum = sum + A[i+size*j] * x[j];
      	x[i] = (A[i+size*size] - sum);
   	} 
   	
   	free(h_pivot);
   	free(h_row);
}


/* Gaussian elimination max column size allocate */
void navie_gaussian_cublas_max(float* A, float* x, long size) {

	printf("Performing CUDA Forward Substitution\n");

	cublasStatus status; // Error handle cublasStatus
	cublasInit(); // Call before using cublas

	float scaler; // Diagonal term
	
	float *h_pivot = (float*)malloc((size+1)*sizeof(float)); // Current row
	float *d_pivot;
	cublasAlloc(size+1, sizeof(float), (void**)&d_pivot); // Allocate cublas memory
	
	float *h_row = (float*)malloc((size+1)*sizeof(float)); // Row below
	float *d_row;
	cublasAlloc(size+1, sizeof(float), (void**)&d_row); // Allocate cublas memory
	
	long col, subCol, subRow;
	
	for (col = 0; col < size; col++) { // Go through each column in A
	
		printf("Column: %ld...\n", col);
		
		/* Step 1: Setup pivoting row with leading element 1 */
		scaler = A[col+col*size]; // Get the current diagonal term in A 
		
		// Fill pivot row host data 
		for (subCol = 0; subCol < size+1; subCol++) 
			h_pivot[subCol] = A[col+subCol*size];
		
		// Call cublasSscal()
		cublasSetVector(size+1, sizeof(float), h_pivot, 1, d_pivot, 1);
		cublasSscal(size+1, 1/scaler, d_pivot, 1);
		cublasGetVector(size+1, sizeof(float), d_pivot, 1, h_pivot, 1);
			
		// Copy back
		for (subCol = 0; subCol < size+1; subCol++) 
			A[col+subCol*size] = h_pivot[subCol];
			
		
		/* Step 2: Set all the element with leading element 0 */
		for (subRow = col+1; subRow < size; subRow++) {
		
			scaler = A[subRow+col*size]; // Same column but row below
			
			// Fill another row host data
			for (subCol = 0; subCol < size+1; subCol++) 
				h_row[subCol] = A[subRow+subCol*size];
				
			// Call cublasSaxpy()
			cublasSetVector(size+1, sizeof(float), h_row, 1, d_row, 1);
			cublasSaxpy(size+1, (-1)*scaler, d_pivot, 1, d_row, 1);
			cublasGetVector(size+1, sizeof(float), d_row, 1, h_row, 1);
			
			// Copy back
			for (subCol = 0; subCol < size+1; subCol++) 
				A[subRow+subCol*size] = h_row[subCol];
				
		}
	}
	
	cublasFree(d_pivot);
	cublasFree(d_row);
	cublasShutdown();

	printf("Performing CPU Backward Substitution\n");
	x[size-1] = A[(size-1) + size*size]; // Compute the most bottom element
   	
   	long i, j;
   	
   	for (i = size-1; i >= 0; i--)	{
    	float sum = 0;
      	for (j = i+1; j < size; j++)
            sum = sum + A[i+size*j] * x[j];
      	x[i] = (A[i+size*size] - sum);
   	}    
   	
   	free(h_pivot);
   	free(h_row);
}


/* Cusparse Solver */
void gaussian_cusparse(float* A, float* x, long size) {

	COO A_h = sparseBuilder(A, size); // Convert matrix to sparse coo form
	 
	/* Setup cuspare variables */
	cusparseStatus_t status;
	cusparseHandle_t handle = 0;
	cusparseMatDescr_t ASparse = 0; // Matrix Descriptor
	
	/* Copy COO A matrix to GPU memory space */
	COO A_d = sparseGPUSetup(A_h);
		
	/* Initialize cusparse and matrix descriptor */
	cusparseCreate(&handle); 
	cusparseCreateMatDescr(&ASparse); 
	cusparseSetMatType(ASparse, CUSPARSE_MATRIX_TYPE_GENERAL); // Type: General
	cusparseSetMatIndexBase(ASparse, CUSPARSE_INDEX_BASE_ZERO); // Zero based index
	
	/* Gaussian Elimination */
	
	
	
 	/* Free resources */
	sparseClean(A_h, A_d);
	cusparseDestroy(handle);
}


/* Pure cuda implementation */ 
void navie_gaussian_cuda_max(float* A, float* x, long size) {
	
	int threads = 0, blocks = 0;
	
	setKernelProperty(threads, blocks, size+1);
	
	printf("Performing CUDA Forward Substitution\n");
	
	/* Allocate memory for max row size */
	float *h_pivot = (float*)malloc((size+1)*sizeof(float)); // Current row
	float *h_row = (float*)malloc((size+1)*sizeof(float)); // Current row
	
	float* d_pivot; 
	float* d_row;
	cudaMalloc((void**)&d_pivot, (size+1)*sizeof(float));
	cudaMalloc((void**)&d_row, (size+1)*sizeof(float));
	
	/* Gaussian Elimination */
	float scaler; // Diagonal term
	long col, subCol, subRow;
	for (col = 0; col < size; col++) { // Go through each column in A
	
		printf("Column: %ld...\n", col);
		
		/* Step 1: Setup pivoting row with leading element 1 */
		scaler = A[col+col*size]; // Get the current diagonal term in A 
		
		// Fill pivot row host data 
		for (subCol = 0; subCol < size+1; subCol++) 
			h_pivot[subCol] = A[col+subCol*size];
		
		// Launch pivot kernel 
		cudaMemcpy(d_pivot, h_pivot, (size+1)*sizeof(float), cudaMemcpyHostToDevice);
		pivot<<<blocks, threads>>>(d_pivot, scaler, size+1);
		cudaMemcpy(h_pivot, d_pivot, (size+1)*sizeof(float), cudaMemcpyDeviceToHost);
		
		// Copy back
		for (subCol = 0; subCol < size+1; subCol++) 
			A[col+subCol*size] = h_pivot[subCol];
			
		
		/* Step 2: Set all the element with leading element 0 */
		for (subRow = col+1; subRow < size; subRow++) {
		
			scaler = A[subRow+col*size]; // Same column but row below
			
			// Fill another row host data
			for (subCol = 0; subCol < size+1; subCol++) 
				h_row[subCol] = A[subRow+subCol*size];
				
			// Launch multiply-addition kernel
			cudaMemcpy(d_row, h_row, (size+1)*sizeof(float), cudaMemcpyHostToDevice);
			mulAdd<<<blocks, threads>>>(d_pivot, d_row, scaler, size+1);
			cudaMemcpy(h_row, d_row, (size+1)*sizeof(float), cudaMemcpyDeviceToHost);
				
			// Copy back
			for (subCol = 0; subCol < size+1; subCol++) 
				A[subRow+subCol*size] = h_row[subCol];
		}
	}
	
	/* Clean memory */
	free(h_pivot); free(h_row);
	cudaFree(d_pivot); cudaFree(d_row);
	
	printf("Performing CPU Backward Substitution\n");
	x[size-1] = A[(size-1) + size*size]; // Compute the most bottom element
   	
   	long i, j;
   	
   	for (i = size-1; i >= 0; i--)	{
    	float sum = 0;
      	for (j = i+1; j < size; j++)
            sum = sum + A[i+size*j] * x[j];
      	x[i] = (A[i+size*size] - sum);
   	}    
}





