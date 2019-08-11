#pragma once

#include <iostream>
#include <string>
#include <nvjpeg.h>
#include <cudnn.h>
#include <cublas.h>

const char *cudaGetErrorName(nvjpegStatus_t error);
const char *cudaGetErrorName(cudnnStatus_t error);
const char *cudaGetErrorName(cublasStatus_t error);

class DLFSError : public std::runtime_error
{
public:
	DLFSError(const std::string &msg) : std::runtime_error(msg)
	{
		m_errorMsg = msg;
	}
	const char *what()
	{
		return m_errorMsg.c_str();
	}

private:
	std::string m_errorMsg;
};

template <typename T>
void checkWithException(T result, const char *func, const char *file,
						int const line)
{
	if (result)
	{
		std::string errorMsg =
			"CUDA error at " + std::string(file) + ":" + std::to_string(line) + " code = " +
			std::to_string(static_cast<unsigned int>(result)) +
			" " + cudaGetErrorName(result) + " " + std::string(func);
		throw DLFSError(errorMsg);
	}
}

#ifdef __DRIVER_TYPES_H__
#define checkCudaErrors(val) checkWithException((val), #val, __FILE__, __LINE__)
#endif
