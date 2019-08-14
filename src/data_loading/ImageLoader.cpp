#include <errno.h>
#include <cstring>

#include "ImageLoader.hpp"
#include "../Logging.hpp"
#include "../tensor/Tensor.hpp"

#include <cuda_runtime.h>
#include <cassert>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

using namespace DLFS;
using namespace std;

ImageLoader::ImageLoader()
{
	m_outputFormat = NVJPEG_OUTPUT_RGBI;
	checkCudaErrors(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, NULL, &m_jpegHandle));
	checkCudaErrors(nvjpegJpegStateCreate(m_jpegHandle, &m_jpegState));
	checkCudaErrors(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));
}

ImageLoader::~ImageLoader()
{
	checkCudaErrors(nvjpegJpegStateDestroy(m_jpegState));
	checkCudaErrors(nvjpegDestroy(m_jpegHandle));
	checkCudaErrors(cudaStreamDestroy(m_stream));
}

/**
 * Inspects the JPEG header and returns channel information
 */
void ImageLoader::GetJPEGInfo(const vector<std::vector<uint8_t>> &dataBuffers,
							  std::vector<ImageInfo> &imgInfoBufs)
{
	assert(dataBuffers.size() == imgInfoBufs.size());
	for (unsigned int i = 0; i < dataBuffers.size(); i++)
	{
		checkCudaErrors(nvjpegGetImageInfo(m_jpegHandle, (unsigned char *)dataBuffers[i].data(), dataBuffers[i].size(),
										   &imgInfoBufs[i].channels, &imgInfoBufs[i].subsampling,
										   imgInfoBufs[i].widths, imgInfoBufs[i].heights));
	}
}

/**
 * Takes the information retrieved from GetJPEGInfo, allocates necessary device buffers
 * and returns.
 * 
 * Mostly follows NVIDIA sample / docs
 * 
 * Currently, we have fixed our output format to be RGBi for output to a NHWC tensor, 
 * which is used for tensor cores.
 */
void ImageLoader::AllocateBuffers(std::vector<ImageInfo> &imgInfos, Tensor<uint8_t> &tensor)
{
	unsigned int idx = 0;

	TensorShapeList shapeList(imgInfos.size());

	for (auto &imgInfo : imgInfos)
	{
		TensorShape tensorShape = {1, imgInfo.heights[0], imgInfo.widths[0], 3};
		shapeList.SetShape(idx, tensorShape);
		idx++;
	}

	TensorShape maxDims = shapeList.FindMaxDims();
	tensor.SetShape({(int)imgInfos.size(), maxDims[1], maxDims[2], 3});
	tensor.AllocateIfNecessary();

	idx = 0;
	for (auto devPtr : tensor.GetIterablePointersOverBatch())
	{
		imgInfos[idx].channels = 1;
		imgInfos[idx].nvImage.pitch[0] = 3 * tensor.GetShape()[2];
		imgInfos[idx].nvImage.channel[0] = devPtr;
		idx++;
	}
}

/**
 * Batch decodes images from vector "buffers" into the uint8 tensor "imgBatchTensor"
 * 
 * Args: 
 * 	Array of buffers filled with jpeg data.
 * 
 * Outputs:
 * 	Pointer to CUDA memory structs
 */
std::vector<ImageInfo> ImageLoader::DecodeJPEG(const vector<vector<uint8_t>> &buffers,
											   Tensor<uint8_t> &imgBatchTensor,
											   unsigned int maxHostThread)
{
	std::vector<size_t> imgLengths;
	std::vector<ImageInfo> imgInfoBufs(buffers.size());
	std::vector<const unsigned char *> imgBuffers;
	std::vector<nvjpegImage_t> nvjpegBuffer;
	const unsigned int batchSize = buffers.size();

	for (auto &buf : buffers)
	{
		imgLengths.push_back(buf.size());
		imgBuffers.push_back(buf.data());
	}

	GetJPEGInfo(buffers, imgInfoBufs);

	AllocateBuffers(imgInfoBufs, imgBatchTensor);

	for (auto &img : imgInfoBufs)
	{
		nvjpegBuffer.push_back(img.nvImage);
	}

	checkCudaErrors(nvjpegDecodeBatchedInitialize(m_jpegHandle, m_jpegState,
												  batchSize,
												  maxHostThread,
												  m_outputFormat));
	checkCudaErrors(nvjpegDecodeBatched(m_jpegHandle, m_jpegState,
										imgBuffers.data(),
										imgLengths.data(),
										nvjpegBuffer.data(),
										m_stream));
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

int DLFS::writeBMPi(const char *filename, const unsigned char *d_RGB, int pitch,
					int width, int height)
{
	unsigned int headers[13];
	FILE *outfile;
	int extrabytes;
	int paddedsize;
	int x;
	int y;
	int n;
	int red, green, blue;

	std::vector<unsigned char> vchanRGB(height * width * 3);
	unsigned char *chanRGB = vchanRGB.data();
	checkCudaErrors(cudaMemcpy2D(chanRGB, (size_t)width * 3, d_RGB, (size_t)pitch,
								 width * 3, height, cudaMemcpyDeviceToHost));

	extrabytes =
		4 - ((width * 3) % 4); // How many bytes of padding to add to each
	// horizontal line - the size of which must
	// be a multiple of 4 bytes.
	if (extrabytes == 4)
		extrabytes = 0;

	paddedsize = ((width * 3) + extrabytes) * height;

	// Headers...
	// Note that the "BM" identifier in bytes 0 and 1 is NOT included in these
	// "headers".
	headers[0] = paddedsize + 54; // bfSize (whole file size)
	headers[1] = 0;				  // bfReserved (both)
	headers[2] = 54;			  // bfOffbits
	headers[3] = 40;			  // biSize
	headers[4] = width;			  // biWidth
	headers[5] = height;		  // biHeight

	// Would have biPlanes and biBitCount in position 6, but they're shorts.
	// It's easier to write them out separately (see below) than pretend
	// they're a single int, especially with endian issues...

	headers[7] = 0;			 // biCompression
	headers[8] = paddedsize; // biSizeImage
	headers[9] = 0;			 // biXPelsPerMeter
	headers[10] = 0;		 // biYPelsPerMeter
	headers[11] = 0;		 // biClrUsed
	headers[12] = 0;		 // biClrImportant

	if (!(outfile = fopen(filename, "wb")))
	{
		std::cerr << "Cannot open file: " << filename << std::endl;
		return 1;
	}

	//
	// Headers begin...
	// When printing ints and shorts, we write out 1 character at a time to avoid
	// endian issues.
	//

	fprintf(outfile, "BM");

	for (n = 0; n <= 5; n++)
	{
		fprintf(outfile, "%c", headers[n] & 0x000000FF);
		fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
		fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
		fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
	}

	// These next 4 characters are for the biPlanes and biBitCount fields.

	fprintf(outfile, "%c", 1);
	fprintf(outfile, "%c", 0);
	fprintf(outfile, "%c", 24);
	fprintf(outfile, "%c", 0);

	for (n = 7; n <= 12; n++)
	{
		fprintf(outfile, "%c", headers[n] & 0x000000FF);
		fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
		fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
		fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
	}

	//
	// Headers done, now write the data...
	//
	for (y = height - 1; y >= 0;
		 y--) // BMP image format is written from bottom to top...
	{
		for (x = 0; x <= width - 1; x++)
		{
			red = chanRGB[(y * width + x) * 3];
			green = chanRGB[(y * width + x) * 3 + 1];
			blue = chanRGB[(y * width + x) * 3 + 2];

			if (red > 255)
				red = 255;
			if (red < 0)
				red = 0;
			if (green > 255)
				green = 255;
			if (green < 0)
				green = 0;
			if (blue > 255)
				blue = 255;
			if (blue < 0)
				blue = 0;
			// Also, it's written in (b,g,r) format...

			fprintf(outfile, "%c", blue);
			fprintf(outfile, "%c", green);
			fprintf(outfile, "%c", red);
		}
		if (extrabytes) // See above - BMP lines must be of lengths divisible by 4.
		{
			for (n = 1; n <= extrabytes; n++)
			{
				fprintf(outfile, "%c", 0);
			}
		}
	}

	fclose(outfile);
	return 0;
}