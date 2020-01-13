#pragma once

#include <string>
#include <vector>

namespace DLFS {
class DataSource {
  public:
    virtual void GetBlob(const std::string &blobPath,
                         std::vector<uint8_t> &destBuffer) = 0;

    void SetPath(const std::string &path) { m_path = path; }

    const std::string &GetPath() { return m_path; }

  private:
    std::string m_path;
};
} // namespace DLFS
