/**
 * LocalSource - implements DataSource interface for 
 * loading files from local storage
 */
#ifndef LOCAL_SOURCE_H_
#define LOCAL_SOURCE_H_

#include <string>
#include <vector>

#include "./DataSource.hpp"

namespace DLFS
{
class LocalSource : public DataSource
{
public:
    LocalSource(const std::string &base_dir = "");

    void GetBlob(const std::string &blobPath,
                 std::vector<uint8_t> &destBuffer) override;

private:
};
} // namespace DLFS

#endif