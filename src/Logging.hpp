#ifndef ERRORS_H_
#define ERRORS_H_

#include <iostream>
#include <nvjpeg.h>
#include <cudnn.h>
#include <cublasLt.h>

const char *cudaGetErrorName(nvjpegStatus_t error);
const char *cudaGetErrorName(cudnnStatus_t error);
const char *cudaGetErrorName(cublasStatus_t error);


template <typename T>
void check(T result, char const *const func, const char *const file,
					 int const line) {
	if (result) {
			std::cout << "CUDA error at " << file <<":"<<line<< " code = " << static_cast<unsigned int>(result) 
				<< " " << cudaGetErrorName(result) <<" "<<func << std::endl;    
		// DEVICE_RESET    
		// exit(EXIT_FAILURE);
	}
}

#ifdef __DRIVER_TYPES_H__
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
#endif

#endif // !ERRORS_H_
