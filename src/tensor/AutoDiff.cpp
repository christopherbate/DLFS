#include "tensor/Tensor.hpp"

#include <sstream>
#include <string>

using namespace DLFS;
using namespace std;

std::string AutoDiffContext::Print() {
    ostringstream ss;

    ss << "-----------Auto Diff Information----------------\n";    
    ss << "------------Operations---------------------------\n";

    for (auto &op : m_opTrace) {
        ss << op->GetName() << " - " << "\n";
    }

    return ss.str();
}

AutoDiffContext DLFS::ADContext;