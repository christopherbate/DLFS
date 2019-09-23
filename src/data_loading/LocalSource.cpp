#include "LocalSource.hpp"
#include <fstream>
#include <iostream>
using namespace DLFS;

LocalSource::LocalSource(const std::string &base_dir)
{
    m_baseDir = base_dir;
}

/**
 * Reads in binary file and fills destBuffer
 */
void LocalSource::GetBlob(const std::string &blobPath, std::vector<uint8_t> &destBuffer)
{
    unsigned int file_size = 0;

    std::string full_path = m_baseDir + blobPath;

    std::ifstream infile(full_path, std::ifstream::binary);
    if (!infile.is_open())
    {
        throw std::runtime_error("Failed to open file");
    }
    infile.seekg(0, std::ios::end);
    file_size = infile.tellg();
    infile.seekg(0, std::ios::beg);

    destBuffer.resize(file_size);
    infile.read(reinterpret_cast<char *>(destBuffer.data()),
                destBuffer.size());
}

/**
 * SetDirectory
 */
void LocalSource::SetDirectory(const std::string &path)
{
    m_baseDir = path;
}