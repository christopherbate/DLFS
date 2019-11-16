#ifndef IMAGE_LDR_H_
#define IMAGE_LDR_H_

#include <cstdint>
#include <fstream>
#include <nvjpeg.h>
#include <string>
#include <thread>
#include <vector>

#include "../tensor/TensorList.hpp"

namespace DLFS {
struct ImageInfo {
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
 * Currently it automatically zero-pads the entire batch to the size of the
 * largest image.
 *
 * TODO: Periodic padding.
 *
 * Example usage
 */
class ImageLoader {
  public:
    ImageLoader();
    ~ImageLoader();

    /**
     * buffer - batch of images read in binary form from .jpg files
     * imgBatchTensor - tensor (not necessarily allocated yet) where the batch
     * will be stored. maxHostThread - number of max CPU threads
     *
     * returns:
     *  vector of ImageInfo
     */
    std::vector<ImageInfo>
    BatchDecodeJPEG(const std::vector<std::vector<uint8_t>> &buffer,
                    TensorPtr<uint8_t> &imgBatchTensor,
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

    void GetJPEGInfo(const std::vector<std::vector<uint8_t>> &dataBuffers,
                     std::vector<ImageInfo> &imgInfoBufs);

    void AllocateBuffers(std::vector<ImageInfo> &imgInfo,
                         TensorPtr<uint8_t> tensor);

    /**
     * Encoder a single jpeg
     */
    template <typename T>
    void EncodeJPEG(TensorPtr<uint8_t> imgTensor, const std::string &filename) {
        std::vector<uint8_t> hostBuffer;
        imgTensor->CopyBufferToHost(hostBuffer);
        auto shape = imgTensor->GetShape();

        nvjpegImage_t src;
        src.channel[0] = (uint8_t *)hostBuffer.data();
        src.pitch[0] = shape[2]*3;

        nvjpegEncodeImage(m_jpegHandle, m_encoderState, m_encoderParams, &src,
                          NVJPEG_INPUT_RGB, shape[2], shape[1],
                          m_encoderStream);

        size_t length;
        nvjpegEncodeRetrieveBitstream(m_jpegHandle, m_encoderState, NULL,
                                      &length, 0);
        hostBuffer.resize(length);
        nvjpegEncodeRetrieveBitstream(m_jpegHandle, m_encoderState,
                                      hostBuffer.data(), &length, 0);

        cudaStreamSynchronize(m_encoderStream);

        try {
            std::ofstream outfile(filename, std::ios::binary | std::ios::out);
            outfile.write((char *)hostBuffer.data(), length);
            outfile.close();
        } catch (std::exception &e) {
            LOG.ERROR() << "Failed to write jpeg output " << e.what();
        }
    }

  private:
    nvjpegHandle_t m_jpegHandle;
    nvjpegJpegState_t m_jpegState;
    nvjpegOutputFormat_t m_outputFormat;
    nvjpegEncoderParams_t m_encoderParams;
    cudaStream_t m_stream;
    nvjpegEncoderState_t m_encoderState;
    cudaStream_t m_encoderStream;
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
void WriteImageTensorPNG(const std::string &filename,
                         TensorPtr<uint8_t> imageTensor);

} // namespace DLFS

#endif // !IMAGE_H