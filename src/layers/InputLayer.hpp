#ifndef INPUT_LAYER_H_
#define INPUT_LAYER_H_

#include <cudnn.h>

#include "Layer.hpp"

namespace DLFS{

template <typename T>
class InputLayer : public Layer<T>
{
public:
    InputLayer(cudnnHandle_t &handle, Layer<T> *prevLayer);
    ~InputLayer();

    void SetInputDim(TensorDims &inputDims) override;

    void AllocateBuffers() override;

protected:
};

}

#endif // !INPUT_LAYER_H_