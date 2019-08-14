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

/**
 * Handles decoding jpegs into 3d arrays
 * Uses nvidia's cuda jpeg library to do this on 
 * the GPU.
 * 
 * This class does not do file loading, for that see
 * the DataSource class tree.
 * 
 * Currently it automatically zero-pads the entire batch to the size of the largest 
 * image. 
 * 
 * TODO: Periodic padding.
 * 
 * Example usage
 */
class ImageLoader
{
public:
    /**
     * batchSize - how many images to decode at one time.
     * maxHostThread - how many cpu threads to spawn during decoding.
     * padToMax - whether to zero-pad all images out the largest image size.
     */
    ImageLoader();
    ~ImageLoader();

    /**
     * This accepts a vector of buffers containing the binary data of the images.
     * It also a single Tensor (uint8_t type). 
     * The images are batch decoded and placed into imgBatchTensor.
     */
    std::vector<ImageInfo> DecodeJPEG(const std::vector<std::vector<uint8_t>> &buffer,
                                      Tensor<uint8_t> &imgBatchTensor,
                                      unsigned int maxHostThread = 4);

    static const char *SamplingTypeString(nvjpegChromaSubsampling_t type);

    void GetJPEGInfo(const std::vector<std::vector<uint8_t>> &dataBuffers, std::vector<ImageInfo> &imgInfoBufs);

    void AllocateBuffers(std::vector<ImageInfo> &imgInfo, Tensor<uint8_t> &tensor);

private:
    nvjpegHandle_t m_jpegHandle;
    nvjpegJpegState_t m_jpegState;
    nvjpegOutputFormat_t m_outputFormat;
    cudaStream_t m_stream;    
};

int writeBMPi(const char *filename, const unsigned char *d_RGB, int pitch,
              int width, int height);

} // namespace DLFS

#endif // !IMAGE_H