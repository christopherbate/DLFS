#include <errno.h>
#include <cstring>

#include "ImageLoader.hpp"
#include "../Logging.hpp"
#include "../tensor/Tensor.hpp"

#include <cuda_runtime.h>

using namespace DLFS;
using namespace std;

ImageLoader::ImageLoader(unsigned int batchSize, unsigned int maxHostThread) : m_batchSize(batchSize),
																			   m_maxHostThread(maxHostThread)
{
	m_outputFormat = NVJPEG_OUTPUT_RGB;

	checkCudaErrors(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, NULL, &m_jpegHandle));
	checkCudaErrors(nvjpegJpegStateCreate(m_jpegHandle, &m_jpegState));
	checkCudaErrors(nvjpegDecodeBatchedInitialize(m_jpegHandle, m_jpegState,
												  m_batchSize, m_maxHostThread, m_outputFormat));

	m_jpegs.resize(m_batchSize);
	m_imgLengths.resize(m_batchSize);

	// Create cuda Stream
	checkCudaErrors(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));
}

ImageLoader::~ImageLoader()
{
	// release cuda buffers
	for (int i = 0; i < m_jpegs.size(); i++)
	{
		for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++)
		{
			if (m_jpegs[i].channel[c])
			{
				cudaFree(m_jpegs[i].channel[c]);
			}
		}
	}

	checkCudaErrors(nvjpegJpegStateDestroy(m_jpegState));
	checkCudaErrors(nvjpegDestroy(m_jpegHandle));

	checkCudaErrors(cudaStreamDestroy(m_stream));
}

/**
 * Inspects the JPEG header and returns channel information
 */
void ImageLoader::GetJPEGInfo(const vector<std::vector<uint8_t>> &dataBuffers, std::vector<ImageInfo> &imgInfoBufs)
{
	for (unsigned int i = 0; i < dataBuffers.size(); i++)
	{
		nvjpegGetImageInfo(m_jpegHandle, (unsigned char *)dataBuffers[i].data(), dataBuffers[i].size(),
						   &imgInfoBufs[i].channels, &imgInfoBufs[i].subsampling,
						   imgInfoBufs[i].widths, imgInfoBufs[i].heights);
	}
}

/**
 * Takes the information retrieved from GetJPEGInfo, allocates necessary device buffers
 * and returns.
 * 
 * Mostly follows NVIDIA sample / docs
 * 
 * Currently, we have fixed our output format to be RGB, but extra logic is included 
 * in here from the sample in case that changes.
 */
void ImageLoader::AllocateBuffers(ImageInfo &imgInfo, Tensor &tensor)
{
	int mul = 1;

	// in the case of interleaved RGB output, write only to single channel, but
	// 3 samples at once
	if (m_outputFormat == NVJPEG_OUTPUT_RGBI ||
		m_outputFormat == NVJPEG_OUTPUT_BGRI)
	{
		imgInfo.channels = 1;
		mul = 3;
	}
	else if (m_outputFormat == NVJPEG_OUTPUT_RGB ||
			 m_outputFormat == NVJPEG_OUTPUT_BGR)
	{
		imgInfo.channels = 3;
		imgInfo.widths[1] = imgInfo.widths[2] = imgInfo.widths[0];
		imgInfo.heights[1] = imgInfo.heights[2] = imgInfo.heights[0];
	}
	else
	{
		throw std::runtime_error(string("Unknown output format " + to_string(m_outputFormat)).c_str());
	}

	// For RGB, all pitchs should be width[0], all channels
	// should be of size pitch[0]*height
	// For RGBi, pitch is width[0]*3 (hence mul factor)
	std::array<int, 4> tensorShape;
	for (int c = 0; c < imgInfo.channels; c++)
	{
		int pitch = mul * imgInfo.widths[c];
		int sz = pitch * imgInfo.heights[c];
		imgInfo.nvImage.pitch[c] = pitch;
		// imgInfo.nvImage.channel[c] =
		cudaMalloc(&imgInfo.nvImage.channel[c], sz);

		//TODO: How to get this into tensor?
		tensorShape[c] = sz;
	}

	// We are explicitly only allocating 3 channel tenosrs.
	tensorShape = {1, imgInfo.heights[0], imgInfo.widths[0], 3};
	tensor.SetShape(tensorShape);
	tensor.AllocateIfNecessary();
}

/**
 * Args: 
 * 	Array of buffers filled with jpeg data.
 * 
 * Outputs:
 * 	Pointer to CUDA memory structs
 */
std::vector<ImageInfo> ImageLoader::DecodeJPEG(const vector<vector<uint8_t>> &buffers)
{
	std::vector<size_t> imgLengths;
	std::vector<ImageInfo> imgInfoBufs(buffers.size());

	for (auto &&buf : buffers)
	{
		imgLengths.push_back(buf.size());
	}

	GetJPEGInfo(buffers, imgInfoBufs);

	for (auto &imgInfo : imgInfoBufs)
	{
		Tensor tensor;
		AllocateBuffers(imgInfo, tensor);
	}

	checkCudaErrors(nvjpegDecodeBatched(m_jpegHandle, m_jpegState,
										reinterpret_cast<const unsigned char *const *>(buffers.data()),
										imgLengths.data(), m_jpegs.data(), m_stream));
	checkCudaErrors(cudaStreamSynchronize(m_stream));

	return imgInfoBufs;
}

const char *ImageLoader::SamplingTypeString(nvjpegChromaSubsampling_t type)
{
	switch (type)
	{
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
