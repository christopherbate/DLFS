#include <cstring>
#include <errno.h>

#include "../Logging.hpp"
#include "../tensor/Tensor.hpp"
#include "ImageLoader.hpp"

#include <cassert>
#include <cuda_runtime.h>

#include <cassert>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

using namespace DLFS;
using namespace std;

/**
 * These are the allocators for nvjpeg backend.
 */
int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }
int dev_free(void *p) { return (int)cudaFree(p); }

ImageLoader::ImageLoader() {
    m_outputFormat = NVJPEG_OUTPUT_RGBI;

    nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
    checkCudaErrors(
        nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &m_jpegHandle));

    checkCudaErrors(nvjpegJpegStateCreate(m_jpegHandle, &m_jpegState));
    checkCudaErrors(
        cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));
    checkCudaErrors(
        cudaStreamCreateWithFlags(&m_encoderStream, cudaStreamNonBlocking));
    checkCudaErrors(nvjpegEncoderParamsCreate(m_jpegHandle, &m_encoderParams,
                                              m_encoderStream));
    checkCudaErrors(
        nvjpegEncoderParamsSetQuality(m_encoderParams, 90, m_encoderStream));
    checkCudaErrors(nvjpegEncoderStateCreate(m_jpegHandle, &m_encoderState,
                                             m_encoderStream));
    checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(
        m_encoderParams, NVJPEG_CSS_444, m_encoderStream));
}

ImageLoader::~ImageLoader() {        
    checkCudaErrors(cudaStreamDestroy(m_stream));
    checkCudaErrors(cudaStreamDestroy(m_encoderStream));
    checkCudaErrors(nvjpegJpegStateDestroy(m_jpegState));
    checkCudaErrors(nvjpegEncoderParamsDestroy(m_encoderParams));
    checkCudaErrors(nvjpegEncoderStateDestroy(m_encoderState));
    checkCudaErrors(nvjpegDestroy(m_jpegHandle));
    
}

/**
 * Inspects the JPEG header and returns channel information
 */
void ImageLoader::GetJPEGInfo(const vector<std::vector<uint8_t>> &dataBuffers,
                              std::vector<ImageInfo> &imgInfoBufs) {
    assert(dataBuffers.size() == imgInfoBufs.size());
    for (unsigned int i = 0; i < dataBuffers.size(); i++) {
        checkCudaErrors(nvjpegGetImageInfo(
            m_jpegHandle, (unsigned char *)dataBuffers[i].data(),
            dataBuffers[i].size(), &imgInfoBufs[i].channels,
            &imgInfoBufs[i].subsampling, imgInfoBufs[i].widths,
            imgInfoBufs[i].heights));
    }
}

/**
 * Takes the information retrieved from GetJPEGInfo, allocates necessary device
 * buffers and returns.
 *
 * Mostly follows NVIDIA sample / docs
 *
 * Currently, we have fixed our output format to be RGBi for output to a NHWC
 * tensor, which is used for tensor cores.
 */
void ImageLoader::AllocateBuffers(std::vector<ImageInfo> &imgInfos,
                                  TensorPtr<uint8_t> tensor) {
    unsigned int idx = 0;

    TensorShapeList shapeList(imgInfos.size());

    for (auto &imgInfo : imgInfos) {
        TensorShape tensorShape = {1, imgInfo.heights[0], imgInfo.widths[0], 3};
        shapeList.SetShape(idx, tensorShape);
        idx++;
    }

    TensorShape maxDims = shapeList.FindMaxDims();
    tensor->SetShape({(int)imgInfos.size(), maxDims[1], maxDims[2], 3});
    tensor->AllocateIfNecessary();

    idx = 0;
    for (auto devPtr : tensor->GetIterablePointersOverBatch()) {
        imgInfos[idx].channels = 1;
        imgInfos[idx].nvImage.pitch[0] = 3 * tensor->GetShape()[2];
        imgInfos[idx].nvImage.channel[0] = devPtr;
        idx++;
    }
}

/**
 * Batch decodes images from vector "buffers" into the uint8 tensor
 * "imgBatchTensor"
 *
 * Args:
 * 	Array of buffers filled with jpeg data.
 *
 * Outputs:
 * 	Pointer to CUDA memory structs
 */
std::vector<ImageInfo>
ImageLoader::BatchDecodeJPEG(const vector<vector<uint8_t>> &buffers,
                             TensorPtr<uint8_t> &imgBatchTensor,
                             unsigned int maxHostThread) {
    std::vector<size_t> imgLengths;
    std::vector<ImageInfo> imgInfoBufs(buffers.size());
    std::vector<const unsigned char *> imgBuffers;
    std::vector<nvjpegImage_t> nvjpegBuffer;
    const unsigned int batchSize = buffers.size();

    for (auto &buf : buffers) {
        imgLengths.push_back(buf.size());
        imgBuffers.push_back(buf.data());
    }

    GetJPEGInfo(buffers, imgInfoBufs);

    AllocateBuffers(imgInfoBufs, imgBatchTensor);

    for (auto &img : imgInfoBufs) {
        nvjpegBuffer.push_back(img.nvImage);
    }

    checkCudaErrors(nvjpegDecodeBatchedInitialize(
        m_jpegHandle, m_jpegState, batchSize, maxHostThread, m_outputFormat));
    checkCudaErrors(nvjpegDecodeBatched(m_jpegHandle, m_jpegState,
                                        imgBuffers.data(), imgLengths.data(),
                                        nvjpegBuffer.data(), m_stream));
    checkCudaErrors(cudaStreamSynchronize(m_stream));

    return imgInfoBufs;
}

/**
 * Decodes a single JPEG.
 *
 * Args:
 * 	buffers
 *  tensor
 *
 * Outputs:
 * 	Pointer to CUDA memory structs
 */
void ImageLoader::DecodeJPEG(const vector<uint8_t> &imgData,
                             TensorPtr<uint8_t> tensor) {
    ImageInfo imgInfo;

    checkCudaErrors(nvjpegGetImageInfo(
        m_jpegHandle, (unsigned char *)imgData.data(), imgData.size(),
        &imgInfo.channels, &imgInfo.subsampling, imgInfo.widths,
        imgInfo.heights));

    TensorShape tensorShape = {1, imgInfo.heights[0], imgInfo.widths[0], 3};
    tensor->SetShape(tensorShape);
    tensor->AllocateIfNecessary();

    /**
     * Note: this configuration is specific to OUTPUT_RGBI
     */
    imgInfo.channels = 1;
    imgInfo.nvImage.pitch[0] = 3 * tensor->GetShape()[2];
    imgInfo.nvImage.channel[0] = tensor->GetDevicePointer();

    nvjpegDecode(m_jpegHandle, m_jpegState, imgData.data(), imgData.size(),
                 m_outputFormat, &imgInfo.nvImage, m_stream);

    checkCudaErrors(cudaStreamSynchronize(m_stream));
}

/**
 * Returns a string for a given nvJpeg Chroma Subsampling type
 */
const char *ImageLoader::SamplingTypeString(nvjpegChromaSubsampling_t type) {
    switch (type) {
    case NVJPEG_CSS_444:
        return "YUV 4:4:4 chroma subsampling";
        break;
    case NVJPEG_CSS_440:
        return "YUV 4:4:0 chroma subsampling";
        break;
    case NVJPEG_CSS_422:
        return "YUV 4:2:2 chroma subsampling";
        break;
    case NVJPEG_CSS_420:
        return "YUV 4:2:0 chroma subsampling";
        break;
    case NVJPEG_CSS_411:
        return "YUV 4:1:1 chroma subsampling";
        break;
    case NVJPEG_CSS_410:
        return "YUV 4:1:0 chroma subsampling";
        break;
    case NVJPEG_CSS_GRAY:
        return "Grayscale JPEG ";
        break;
    case NVJPEG_CSS_UNKNOWN:
        return "Unknown chroma subsampling";
    }
    return "Unknown chroma subsampling";
}
