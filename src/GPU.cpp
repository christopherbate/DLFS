#include <cuda_runtime_api.h>

#include "Logging.hpp"
#include "GPU.hpp"

using namespace std;
using namespace DLFS;

GPU::GPU(/* args */)
{
    m_props.deviceNum = 0;
    cudaDeviceProp props;
    checkCudaErrors(cudaGetDeviceProperties(&props, m_props.deviceNum));

    printf("Using GPU %d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n",
           m_props.deviceNum, props.name, props.multiProcessorCount,
           props.maxThreadsPerMultiProcessor, props.major, props.minor,
           props.ECCEnabled ? "on" : "off");

    m_cudnnHandles.resize(1);
    checkCudaErrors(cudnnCreate(&m_cudnnHandles[0]));
}

GPU::~GPU()
{
    for (auto &handle : m_cudnnHandles)
    {
        checkCudaErrors(cudnnDestroy(handle));
        handle = NULL;
    }
}

GPU DLFS::GPUContext;