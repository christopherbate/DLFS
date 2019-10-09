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
    ImageLoader();
    ~ImageLoader();

    /**
     * buffer - batch of images read in binary form from .jpg files
     * imgBatchTensor - tensor (not necessarily allocated yet) where the batch will be stored.
     * maxHostThread - number of max CPU threads
     * 
     * returns:
     *  vector of ImageInfo
     */
    std::vector<ImageInfo> BatchDecodeJPEG(const std::vector<std::vector<uint8_t>> &buffer,
                                           TensorPtr<uint8_t> imgBatchTensor,
                                           unsigned int maxHostThread = 4);

    /**
     * Decodes a single jpeg. 
     * For batched images, see above is much more efficient.
     * 
     * tensor - does not need to be allocated.
     */
    void DecodeJPEG(const std::vector<uint8_t> &imgData,
                    TensorPtr<uint8_t> tensor);

    static const char *SamplingTypeString(nvjpegChromaSubsampling_t type);

    void GetJPEGInfo(const std::vector<std::vector<uint8_t>> &dataBuffers, std::vector<ImageInfo> &imgInfoBufs);

    void AllocateBuffers(std::vector<ImageInfo> &imgInfo, TensorPtr<uint8_t> tensor);

private:
    nvjpegHandle_t m_jpegHandle;
    nvjpegJpegState_t m_jpegState;
    nvjpegOutputFormat_t m_outputFormat;
    cudaStream_t m_stream;
};

int writeBMPi(const char *filename, const unsigned char *d_RGB, int pitch,
              int width, int height);

/**
 * Saves a tensor as a BMP 
 * Must have 3 channels (RGB)
 */
void WriteImageTensorBMP(const char *filename, TensorPtr<uint8_t> imageTensor);

/**
 * Saves a tensor as a PNG
 * 
 * Automatically performs greyscale conversion 
 * if single-channel
 */
void WriteImageTensorPNG(const std::string &filename, TensorPtr<uint8_t> imageTensor);

} // namespace DLFS

#endif // !IMAGE_H