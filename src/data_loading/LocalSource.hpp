#ifndef LOCAL_SOURCE_H_
#define LOCAL_SOURCE_H_

#include <string>
#include <vector>

#include "./DataSource.hpp"

namespace DLFS{
class LocalSource : public DataSource
{
public:
    LocalSource( const std::string &base_dir);
    std::vector<uint8_t> get_blob(const std::string &blob_path);

    void getBlob(const std::string &blobPath, std::vector<uint8_t> &destBuffer);
private:
    std::string m_baseDir;
};
}

#endif