#include "Tensor.hpp"

namespace DLFS{

/**
 * TensorDim operators
 */
std::ostream &operator<<(std::ostream &out, const TensorDims &d)
{
    out << "(BHWC) " << "(" << d.batch << ", " << d.height << ", " << d.width << ", " << d.channels << ")" << " Size: " << (d.batch*d.width*d.height*d.width) << " units";
    return out;
}

}