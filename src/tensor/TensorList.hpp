#ifndef TENSOR_LIST_H_
#define TENSOR_LIST_H_

#include "Tensor.hpp"

#include <cassert>
#include <vector>

namespace DLFS
{

template <typename T>
class TensorList
{
public:
    TensorList(int listSize = 1);
    ~TensorList();

    void SetTensorShape(unsigned int idx, TensorShape &shape);

    Tensor<T> &GetMutableTensor(unsigned int idx);
    const Tensor<T> &GetTensor(unsigned int idx);

    TensorShape FindMaxDims();

    inline unsigned int Length()
    {
        return m_tensors.size();
    }

    std::vector<Tensor<T>> &GetMutableIterable();

private:
    int m_listSize;
    std::vector<Tensor<T>> m_tensors;
    Tensor<T> m_combinedTensor;
};

class TensorShapeList
{
public:
    TensorShapeList(int size = 0)
    {
        m_shapeList.resize(size);
    }
    ~TensorShapeList() {}

    void AddShape(const TensorShape &shape)
    {
        m_shapeList.push_back(shape);
    }

    void SetShape(const unsigned int idx, const TensorShape &shape)
    {
        assert(idx < m_shapeList.size());
        m_shapeList[idx] = shape;
    }

    TensorShape FindMaxDims();

    inline unsigned int Length()
    {
        return m_shapeList.size();
    }

private:
    std::vector<TensorShape> m_shapeList;
};

} // namespace DLFS

#endif