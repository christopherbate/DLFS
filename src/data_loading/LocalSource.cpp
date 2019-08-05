#include "LocalSource.hpp"
#include <fstream>
using namespace DLFS;

std::vector<uint8_t> get_blob(const std::string &blob_path)
{
    unsigned int file_size = 0;

    std::ifstream infile(blob_path, std::ifstream::binary);
    infile.seekg(0, std::ios::end);
    file_size = infile.tellg();
    infile.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(file_size);
    infile.read(reinterpret_cast<char *>(buffer.data()),
                buffer.size());
}