#ifndef DATA_SOURCE_H_
#define DATA_SOURCE_H_

#include <string>
#include <vector>

namespace DLFS
{
class DataSource
{
public:
    virtual void GetBlob(const std::string &blobPath,
                         std::vector<uint8_t> &destBuffer) = 0;
};
} // namespace DLFS

#endif