#ifndef IMAGE_LDR_H_
#define IMAGE_LDR_H_

#include <string>
#include <vector>
#include <fstream>
#include <cstdint>
#include <nvjpeg.h>
#include <thread>

#include "../tensor/TensorList.hpp"

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
    ImageLoader(unsigned int batchSize, unsigned int maxHostThread = 4, bool padToMax = true);
    ~ImageLoader();

    std::vector<ImageInfo> DecodeJPEG(const std::vector<std::vector<uint8_t>> &buffer,
                                      Tensor &imgBatchTensor);

    void WriteBatchBMP();

    static const char *SamplingTypeString(nvjpegChromaSubsampling_t type);

    void GetJPEGInfo(const std::vector<std::vector<uint8_t>> &dataBuffers, std::vector<ImageInfo> &imgInfoBufs);

    void AllocateBuffers(std::vector<ImageInfo> &imgInfo, Tensor &tenso);

private:
    nvjpegHandle_t m_jpegHandle;
    nvjpegJpegState_t m_jpegState;
    nvjpegOutputFormat_t m_outputFormat;
    cudaStream_t m_stream;

    int m_batchSize;
    int m_maxHostThread;
    bool m_padToMax;

    int m_dev;
    int m_warmpup;
    bool m_pipelined;
    bool m_batched;    
};

int writeBMPi(const char *filename, const unsigned char *d_RGB, int pitch,
              int width, int height);

} // namespace DLFS

#endif // !IMAGE_H