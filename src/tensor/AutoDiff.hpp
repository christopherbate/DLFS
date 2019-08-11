#pragma once

#include <string>
#include <vector>
#include <memory>

#include "operations/OpTypes.hpp"

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
    }
    
private:
    std::vector<std::shared_ptr<TrackableOp>> m_opTrace;
};

extern AutoDiffContext ADContext;

} // namespace DLFS