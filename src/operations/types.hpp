#pragma once
#include <array>

namespace DLFS
{
typedef std::array<int, 2> Pad2d;
typedef std::array<int, 2> Stride2d;

enum CustomOpDataType
{
    Float,
    Double,
    Half
};
} // namespace DLFS