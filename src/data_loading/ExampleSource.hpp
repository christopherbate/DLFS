#ifndef ANN_SRC_H_
#define ANN_SRC_H_

#include <string>

#include "dataset_generated.h"

namespace DLFS
{
class ExampleSource
{
public:
    ExampleSource();
    ~ExampleSource();

    void init(const std::string &path);

    const Example* GetExample(unsigned int idx);

    inline unsigned int GetNumExamples(){
        return m_numExamples;
    }

private:
    Dataset *m_dataset;
    unsigned int m_numExamples;
    char *m_buffer;
};
} // namespace DLFS

#endif