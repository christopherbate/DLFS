#pragma once

#include <string>
#include <vector>
#include <memory>

#include "tensor/Tensor.hpp"

namespace DLFS
{

class TrackableOp
{
public:
    TrackableOp();

    virtual void ExecuteForward() = 0;
    virtual TensorBasePtr GetOutputTensor() = 0;
    virtual void ExecuteBackward() = 0;

    inline std::string GetName()
    {
        return m_name;
    }

    inline void SetName(const std::string &name)
    {
        m_name = name;
    }

    inline void SetId(uint32_t id)
    {
        m_id = id;
    }

    inline uint32_t GetId()
    {
        return m_id;
    }    

private:
    std::string m_name;
    uint32_t m_id;
};

class AutoDiffContext
{
public:
    AutoDiffContext();
    ~AutoDiffContext();

    void AddOp(std::shared_ptr<TrackableOp> op)
    {
        op->SetId(m_opTrace.size());
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

    void CalcGradient(TensorBasePtr scalarTensor,
                      std::vector<TensorBasePtr> parameters)
    {
        std::cout << "Calc gradient of f'n with output name : " << scalarTensor->GetName() << std::endl;
        std::cout << "With respect to parameters with names : " << std::endl;
        for (auto p : parameters)
        {
            std::cout << p->GetName() << ":" << p->GetId() << std::endl;
        }

        std::cout << "Operations: " << std::endl;
        for (auto op : m_opTrace)
        {
            std::cout << op->GetName() << ":" << op->GetId() << std::endl;
            if (op->GetOutputTensor() == scalarTensor)
            {
                std::cout << "This op has the scalarTensor as output" << std::endl;
                // op->ExecuteBackward(scalarTensor);
            }
        }
    }

private:
    std::vector<std::shared_ptr<TrackableOp>> m_opTrace;
    std::vector<std::shared_ptr<TensorBase>> m_tensorTrace;
};

extern AutoDiffContext ADContext;

} // namespace DLFS