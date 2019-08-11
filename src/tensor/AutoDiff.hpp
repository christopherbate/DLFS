#pragma once

#include <string>
#include <vector>
#include <memory>

#include "operations/OpTypes.hpp"
#include "tensor/Tensor.hpp"

namespace DLFS
{

class TrackableOp
{
public:
    TrackableOp();

private:
    std::string m_name;
    OpType m_opType;
};

class AutoDiffContext
{
public:
    AutoDiffContext();
    ~AutoDiffContext();

    void AddOp(std::shared_ptr<TrackableOp> op)
    {
        m_opTrace.push_back(op);
    }

    void Reset()
    {
        m_opTrace.clear();
        m_tensorTrace.clear();
    }

    inline unsigned int GetOpTraceSize()
    {
        return m_opTrace.size();
    }

    template <typename T>
    std::shared_ptr<Tensor<T>> CreateTensor()
    {
        TensorPtr<T> p = std::make_shared<Tensor<T>>();
        p->SetId(m_tensorTrace.size());
        m_tensorTrace.emplace_back(p);
        return p;
    }

private:
    std::vector<std::shared_ptr<TrackableOp>> m_opTrace;
    std::vector<std::shared_ptr<TensorBase>> m_tensorTrace;    
};

extern AutoDiffContext ADContext;

} // namespace DLFS