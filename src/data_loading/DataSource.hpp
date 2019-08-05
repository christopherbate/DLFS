#ifndef DATA_SOURCE_H_
#define DATA_SOURCE_H_

#include <string>
#include <vector>

namespace DLFS{
class DataSource
{
public:
    virtual std::vector<uint8_t> get_blob( const std::string &blob_path ) = 0;
};
}

#endif