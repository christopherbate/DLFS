#ifndef LOCAL_SOURCE_H_
#define LOCAL_SOURCE_H_

#include <string>
#include <vector>

#include "./DataSource.hpp"

namespace DLFS{
class LocalSource : public DataSource
{
public:
    std::vector<uint8_t> get_blob(const std::string &blob_path);
};
}

#endif