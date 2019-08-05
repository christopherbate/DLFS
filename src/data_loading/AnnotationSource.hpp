#ifndef ANN_SRC_H_
#define ANN_SRC_H_

#include <string>

#include "annotations_generated.h"

namespace DLFS
{
class AnnotationSource
{
public:
    void init(const std::string &path);

    const Example* GetExample(unsigned int idx);

private:
    Dataset *m_dataset;
    unsigned int m_numExamples;
};
} // namespace DLFS

#endif