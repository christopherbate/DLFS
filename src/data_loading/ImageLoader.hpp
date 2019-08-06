#ifndef IMAGE_LDR_H_
#define IMAGE_LDR_H_

#include <string>
#include <vector>
#include <fstream>
#include <cstdint>
#include <nvjpeg.h>

#include "../tensor/Tensor.hpp"

namespace DLFS
{
struct ImageInfo
{
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    int channels;
    nvjpegImage_t nvImage;
    nvjpegChromaSubsampling_t subsampling;
};

class ImageLoader
{
public:
    ImageLoader(unsigned int batchSize, unsigned int maxHostThread = 4);
    ~ImageLoader();

    std::vector<ImageInfo> DecodeJPEG(const std::vector<std::vector<uint8_t>> &buffer);

    void WriteBatchBMP();

    static const char *SamplingTypeString(nvjpegChromaSubsampling_t type);

    void GetJPEGInfo(const std::vector<std::vector<uint8_t>> &dataBuffers, std::vector<ImageInfo> &imgInfoBufs);

    void AllocateBuffers(ImageInfo &imgInfo, Tensor &tensor);

private:
    nvjpegHandle_t m_jpegHandle;
    nvjpegJpegState_t m_jpegState;
    nvjpegOutputFormat_t m_outputFormat;
    cudaStream_t m_stream;

    std::vector<nvjpegImage_t> m_jpegs;
    std::vector<size_t> m_imgLengths;

    int m_maxHostThread;
    int m_batchSize;
    int m_dev;
    int m_warmpup;
    bool m_pipelined;
    bool m_batched;
};

} // namespace DLFS

#endif // !IMAGE_H