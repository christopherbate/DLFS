#pragma once

#include <cstdint>

namespace DLFS
{

enum OpType : uint32_t
{
    Identity = 0,
    MatrixMultiply = 1
};

}