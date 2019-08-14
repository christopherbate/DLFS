#ifndef ANN_SRC_H_
#define ANN_SRC_H_

#include <string>

#include "dataset_generated.h"

namespace DLFS
{
/**
 * Wraps loading a flat buffer-serialized
 * dataset (annotations) file. Currently only supported implementation is 
 * to load this file from the local disk, but later we could 
 * integrate this with the "LocalSource/AWSSource, etc." to 
 * load this and the images at the same time.
 */
class ExampleSource
{
public:
    ExampleSource();
    ~ExampleSource();

    /**
     * Loads the flat buffer from the 
     * specified path.
     */
    void Init(const std::string &path);

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