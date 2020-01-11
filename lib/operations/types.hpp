#pragma once
#include <array>

namespace DLFS {
using Pad2D = std::array<int, 2>;
using Stride2D = std::array<int, 2>;

// Width x Height
using Filter2D = std::array<int, 2>;

constexpr Pad2D Pad0 = Pad2D{0, 0};
constexpr Pad2D Pad1 = Pad2D{1, 1};
constexpr Pad2D Pad2 = Pad2D{2, 2};
constexpr Stride2D Stride1 = Stride2D{1, 1};
constexpr Stride2D Stride2 = Stride2D{2, 2};

enum CustomOpDataType { Float, Double, Half };
} // namespace DLFS