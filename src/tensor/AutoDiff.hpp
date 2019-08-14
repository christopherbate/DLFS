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

    template <typename T>
    std::shared_ptr<Tensor<T>> CreateTensor(TensorShape shape, const std::string &name,
                                            T constValueFill, bool grad = true)
    {
        TensorPtr<T> p = std::make_shared<Tensor<T>>();
        p->SetGradFlag(grad);
        p->SetShape(shape);
        p->SetName(name);
        p->AllocateIfNecessary();
        p->FillConstant(constValueFill);
        p->SetId(m_tensorTrace.size());

        m_tensorTrace.emplace_back(p);
        return p;
    }

    template <typename T>
    std::shared_ptr<Tensor<T>> CreateFilter(int inChannel, int outChannel, int filterSize,
                                            const std::string &name,
                                            T constValueFill, bool grad = true)
    {
        TensorPtr<T> p = std::make_shared<Tensor<T>>();
        p->SetGradFlag(grad);
        p->SetFilterShape(inChannel, outChannel, filterSize, filterSize);
        p->SetName(name);
        p->AllocateIfNecessary();
        p->FillConstant(constValueFill);
        p->SetId(m_tensorTrace.size());

        m_tensorTrace.emplace_back(p);
        return p;
    }

    void CalcGradient(TensorBasePtr scalarTensor,
                      std::vector<TensorBasePtr> parameters)
    {
        LOG.INFO() << "Calc gradient of f'n with output name : " << scalarTensor->GetName();
        LOG.INFO() << "With respect to parameters with names : ";
        for (auto p : parameters)
        {
            LOG.INFO() << p->GetName() << ":" << p->GetId();
        }

        // Initialize the backward operation. This operation sets up the
        // gradient tensor at the top of the chain.
        scalarTensor->InitGradChain();

        // Cycle through the operations in reverse order.
        LOG.INFO() << "Conducting backward pass:";
        for (auto opIter = m_opTrace.rbegin(); opIter != m_opTrace.rend(); opIter++)
        {
            auto op = *opIter;
            LOG.INFO() << op->GetName() << ":" << op->GetId();

            // Skip this op if it's output hasn't seen a backward pass.
            // this means that this op is somehow disconnected or upstream
            // from scalarTensor
            if (op->GetOutputTensor()->GetBackwardPasses() < 1)
            {
                LOG.INFO() << "Skipping this op.";
                continue;
            }

            op->ExecuteBackward();
        }
    }

private:
    std::vector<std::shared_ptr<TrackableOp>> m_opTrace;
    std::vector<std::shared_ptr<TensorBase>> m_tensorTrace;
};

extern AutoDiffContext ADContext;

} // namespace DLFS