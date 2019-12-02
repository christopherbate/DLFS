#pragma once

#include <string>

#include "../tensor/TensorBase.hpp"
#include "types.hpp"

namespace DLFS {

class BaseOperation : public std::enable_shared_from_this<BaseOperation> {
  public:
    BaseOperation() {}

    // virtual void ExecuteForward() = 0;
    virtual TensorBasePtr GetOutputTensor() = 0;
    virtual void ExecuteBackward() = 0;

    std::string GetName() { return m_name; }

    void SetName(const std::string &name) { m_name = name; }

    std::shared_ptr<BaseOperation> GetShared() { return this->shared_from_this(); }

  private:
    std::string m_name;
};

using BaseOpPtr = std::shared_ptr<BaseOperation>;

} // namespace DLFS
