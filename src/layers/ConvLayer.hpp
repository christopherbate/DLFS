#ifndef  CONV_LAYER_H_
#define CONV_LAYER_H_

#include <cudnn.h>
#include "Layer.hpp"
#include "../tensor/Tensor.hpp"

namespace DLFS{

struct FilterDims
{
    int height;
    int width;
    int inputFeatures;
    int outputFeatures;
    int stride;

    FilterDims()
    {
    }

    int Length(){
        return height*width*inputFeatures*outputFeatures;
    }

    FilterDims(int in, int out, int w, int h)
    {
        height = h;
        width = w;
        inputFeatures = in;
        outputFeatures = out;
        stride = 1;
    }
};

template <typename T>
class ConvLayer : public Layer<T>
{
public:
    ConvLayer(cudnnHandle_t &handle, int filterSize, int stride = 1);
    virtual ~ConvLayer();

    void SetFilerDim(FilterDims &filterDims);

    void AllocateWeightBuffers();
    void AllocateOutputBuffers();

    void FindBestAlgorithm();
    size_t GetAlgWorkspaceNeeded(cudnnConvolutionFwdAlgo_t alg);    

    static const char *GetConvAlgorithmString(cudnnConvolutionFwdAlgo_t alg);    

    void Forward() override;

protected:
    FilterDims m_filterDims;
    cudnnFilterDescriptor_t m_filterTd;
    cudnnConvolutionDescriptor_t m_convDesc;
    cudnnConvolutionFwdAlgo_t m_convAlg;
    size_t m_workspaceSize;
    uint8_t *m_workspaceBuffer;
    uint8_t *m_filterBuffer;

public: /* Debug Functions*/
    void PrintAllWorkspaces();

    // size_t GetMemoryRequirements() override
    // {
    //     return sizeof(T)*(this->m_outputDims.Length()+m_filterDims.Length());
    // }

    // void Print()
    // {
    //     std::cout << "Conv Layer:" <<"\n" << "Mem Req: " << GetMemoryRequirements() /1e6 <<" MB" << std::endl;
    // }
};

}

#endif // ! CONV_LAYER_H