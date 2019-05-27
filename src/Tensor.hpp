#ifndef TENSOR_H_
#define TENSOR_H_

#include <iostream>
#include <vector>

namespace DLFS{
    template <typename T>
using CBuffer = std::vector<T>;

template <typename T>
using WCBuffer = std::vector<CBuffer<T>>;

template <typename T>
using HWCBuffer = std::vector<WCBuffer<T>>;

template <typename T>
using NHWCBuffer = std::vector<HWCBuffer<T>>;

template <typename T>
using Matrix2D = std::vector<std::vector<T>>;

template <typename T>
using Vector1D = std::vector<T>;

struct TensorDims
{
    int batch;
    int height;
    int width;
    int channels;

    TensorDims()
    {
    }

    TensorDims(int b, int h, int w, int c)
    {
        batch = b;
        channels = c;
        width = w;
        height = h;
    }      

    int Length(){
        return batch*height*width*channels;
    }
};

std::ostream &operator<<(std::ostream &out, const TensorDims &d);

}

#endif