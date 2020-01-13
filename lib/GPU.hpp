#pragma once

#include <vector>
#include <cudnn.h>

namespace DLFS
{

class GPU
{
public:
    struct Properties
    {
        int deviceNum = 0;
    };

private:
    Properties m_props;
    std::vector<cudnnHandle_t> m_cudnnHandles;

public:
    GPU();
    ~GPU();

    inline cudnnHandle_t GetCUDNNHandle(){
        return m_cudnnHandles[0];
    }
};

extern GPU GPUContext;

}
