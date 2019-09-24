#pragma once

#include <string>

#include "types.hpp"
#include "../tensor/Tensor.hpp"

namespace DLFS
{


class BaseOperation
{
public:
    BaseOperation() {}

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

} // namespace DLFS
