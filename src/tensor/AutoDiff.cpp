#include "AutoDiff.hpp"

#include <sstream>
#include <string>

using namespace DLFS;
using namespace std;

AutoDiffContext::AutoDiffContext() {}

AutoDiffContext::~AutoDiffContext() {}

std::string AutoDiffContext::Print() {
    ostringstream ss;

    ss << "-----------Auto Diff Information----------------\n";
    ss << "-----------Tensors------------------------------\n";

    for (auto &t : m_tensorTrace) {
        ss << t->GetName() << " - " << t->PrintShape() <<
            " - BW Passes: " << t->GetBackwardPasses() << "\n";
    }

    ss << "------------Operations---------------------------\n";

    for (auto &op : m_opTrace) {
        ss << op->GetName() << " - " << "\n";
    }

    return ss.str();
}

AutoDiffContext DLFS::ADContext;