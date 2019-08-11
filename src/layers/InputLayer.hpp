#ifndef INPUT_LAYER_H_
#define INPUT_LAYER_H_

#include <cudnn.h>

#include "Layer.hpp"

namespace DLFS{

template <typename T>
class InputLayer : public Layer<T>
{
public:
    InputLayer(cudnnHandle_t &handle);
    ~InputLayer();

    void AllocateOutputBuffers() override;
    void AllocateWeightBuffers(){}

protected:
};

}

#endif // !INPUT_LAYER_H_