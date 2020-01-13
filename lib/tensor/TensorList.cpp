#include "TensorList.hpp"
#include <vector>

using namespace DLFS;
using namespace std;

template <typename T>
TensorList<T>::TensorList(int listSize) : m_listSize(listSize)
{
    m_tensors.resize(m_listSize);
}

template <typename T>
TensorList<T>::~TensorList()
{
}

template <typename T>
void TensorList<T>::SetTensorShape(unsigned int idx, TensorShape &shape)
{
    m_tensors[idx].SetShape(shape);
}

template <typename T>
Tensor<T> &TensorList<T>::GetMutableTensor(unsigned int idx)
{
    return m_tensors[idx];
}

template <typename T>
const Tensor<T> &TensorList<T>::GetTensor(unsigned int idx)
{
    return m_tensors[idx];
}

template <typename T>
TensorShape TensorList<T>::FindMaxDims()
{
    TensorShape maxDims = {1, 1, 1, 1};
    for (auto &tensor : m_tensors)
    {
        maxDims[1] = std::max(maxDims[1], tensor.GetShape()[1]);
        maxDims[2] = std::max(maxDims[2], tensor.GetShape()[2]);
        maxDims[3] = std::max(maxDims[3], tensor.GetShape()[3]);
    }
    return maxDims;
}

template <typename T>
std::vector<Tensor<T>> &TensorList<T>::GetMutableIterable()
{
    return m_tensors;
}

TensorShape TensorShapeList::FindMaxDims()
{
    TensorShape maxDims = {0, 0, 0, 0};
    for (auto &shape : m_shapeList)
    {
        maxDims[0] = std::max(maxDims[0], shape[0]);
        maxDims[1] = std::max(maxDims[1], shape[1]);
        maxDims[2] = std::max(maxDims[2], shape[2]);
        maxDims[3] = std::max(maxDims[3], shape[3]);
    }
    return maxDims;
}