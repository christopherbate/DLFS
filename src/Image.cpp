#include <errno.h>
#include <cstring>

#include "Image.hpp"
#include "Logging.hpp"

#include <cuda_runtime.h>

using namespace DLFS;
using namespace std;

int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }
int dev_free(void *p) { return (int)cudaFree(p); }

Image::Image(nvjpegImage_t &jpegDest) : jpeg(jpegDest)
{
	width = height = imgSize = 0;

	for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++)
	{
		jpeg.channel[c] = NULL;
		jpeg.pitch[c] = 0;
		jpegSize.pitch[c] = c;
	}
}

Image::~Image()
{
}

void Image::Read(const std::string &filename)
{
	// Read raw binary data
	filepath = filename;
	std::ifstream infile(filename, std::ifstream::in | std::ifstream::binary);

	if (!infile.is_open())
	{
		cerr << "Cannot open file " << filename << strerror(errno) << std::endl;
		return;
	}

	imgSize = Image::FileSize(infile);

	buffer.resize(imgSize);

	infile.read((char *)buffer.data(), imgSize);

	if (infile.tellg() != imgSize)
	{
		cerr << "Short read on file " << filename << strerror(errno) << std::endl;
		return;
	}
}

unsigned int Image::FileSize(std::ifstream &infile)
{
	unsigned int size = 0;
	infile.seekg(0, ios::end);
	size = infile.tellg();
	infile.seekg(0, ios::beg);

	return size;
}

void Image::PrepareBuffers(nvjpegHandle_t &jpegHandle, nvjpegOutputFormat_t outputFormat)
{
	int channels;
	int widths[NVJPEG_MAX_COMPONENT];
	int heights[NVJPEG_MAX_COMPONENT];
	nvjpegChromaSubsampling_t subsampling;

	checkCudaErrors(nvjpegGetImageInfo(jpegHandle, buffer.data(), imgSize, &channels, &subsampling, widths, heights));

	height = heights[0];
	width = widths[0];

	cout << "Processing " << filepath << endl;
	for (int i = 0; i < NVJPEG_MAX_COMPONENT; i++)
	{
		cout << "Ch " << i << ": " << widths[i] << " x " << heights[i] << endl;
	}
	cout << ImageLoader::SamplingTypeString(subsampling) << endl;

	int mul = 1;

	// in the case of interleaved RGB output, write only to single channel, but 3 samples at once
	// in the case of rgb create 3 buffers with sizes of original image
	if (outputFormat == NVJPEG_OUTPUT_RGBI || outputFormat == NVJPEG_OUTPUT_BGRI)
	{
		channels = 1;
		mul = 3;
	}
	else if (outputFormat == NVJPEG_OUTPUT_RGB ||
			 outputFormat == NVJPEG_OUTPUT_BGR)
	{
		channels = 3;
		widths[1] = widths[2] = widths[0];
		heights[1] = heights[2] = heights[0];
	}

	// realloc output buffer if required
	for (int c = 0; c < channels; c++)
	{
		int aw = mul * widths[c];
		int ah = heights[c];
		int sz = aw * ah;
		jpeg.pitch[c] = aw;
		if (sz > jpegSize.pitch[c])
		{
			if (jpeg.channel[c])
			{
				checkCudaErrors(cudaFree(jpeg.channel[c]));
			}
			checkCudaErrors(cudaMalloc(&jpeg.channel[c], sz));
			jpegSize.pitch[c] = sz;
		}
	}
}

ImageLoader::ImageLoader(FileNames fileNames)
{
	m_batchSize = fileNames.size();
	m_maxHostThread = 1;

	m_outputFormat = NVJPEG_OUTPUT_RGBI;
	nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};

	checkCudaErrors(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &m_jpegHandle));
	checkCudaErrors(nvjpegJpegStateCreate(m_jpegHandle, &m_jpegState));
	checkCudaErrors(nvjpegDecodeBatchedInitialize(m_jpegHandle, m_jpegState,
												  m_batchSize, m_maxHostThread, m_outputFormat));

	cout << "Loading " << fileNames.size() << " images with batch size " << m_batchSize << std::endl;

	m_jpegs.resize(fileNames.size());
	m_imgLengths.resize(m_batchSize);

	// Create cuda Stream
	checkCudaErrors(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));

	unsigned int count = 0;
	for (auto iter = fileNames.begin(); iter != fileNames.end(); iter++)
	{
		m_images.push_back(Image(m_jpegs[count]));
		m_images[count].Read(fileNames[count]);
		m_images[count].PrepareBuffers(m_jpegHandle, m_outputFormat);

		count++;
	}

	// Decode
	checkCudaErrors(cudaStreamSynchronize(m_stream));
	nvjpegStatus_t err;

	std::vector<const unsigned char *> raw_inputs;
	for (int i = 0; i < m_batchSize; i++)
	{
		raw_inputs.push_back((const unsigned char *)m_images[i].buffer.data());
		m_imgLengths[i] = m_images[i].imgSize;
	}

	checkCudaErrors(nvjpegDecodeBatched(m_jpegHandle, m_jpegState, raw_inputs.data(), m_imgLengths.data(), m_jpegs.data(), m_stream));
	checkCudaErrors(cudaStreamSynchronize(m_stream));
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

void ImageLoader::WriteBatchBMP()
{
	// Writeback (for testing)
	for (int i = 0; i < m_batchSize; i++)
	{
		// This will be used to rename the output file.
		size_t position = m_images[i].filepath.rfind(".");
		string sFileName = (std::string::npos == position) ? m_images[i].filepath
														   : m_images[i].filepath.substr(0, position);
		std::string fname(sFileName + ".bmp");

		int err;
		if (m_outputFormat == NVJPEG_OUTPUT_RGB || m_outputFormat == NVJPEG_OUTPUT_BGR)
		{
			err = Image::WriteBMP(fname.c_str(), m_jpegs[i].channel[0], m_jpegs[i].pitch[0],
								  m_jpegs[i].channel[1], m_jpegs[i].pitch[1], m_jpegs[i].channel[2],
								  m_jpegs[i].pitch[2], m_images[i].width, m_images[i].height);
		}
		else if (m_outputFormat == NVJPEG_OUTPUT_RGBI ||
				 m_outputFormat == NVJPEG_OUTPUT_BGRI)
		{
			// Write BMP from interleaved data
			err = Image::WriteBMPi(fname.c_str(), m_jpegs[i].channel[0], m_jpegs[i].pitch[0], m_images[i].width, m_images[i].height);
		}

		if (err)
		{
			std::cout << "Cannot write output file: " << fname << std::endl;
			return;
		}

		std::cout << "Done writing decoded image to file: " << fname << std::endl;
	}
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

// write bmp, input - RGB, device
int Image::WriteBMP(const char *filename, const unsigned char *d_chanR, int pitchR,
					const unsigned char *d_chanG, int pitchG,
					const unsigned char *d_chanB, int pitchB, int width, int height)
{
	unsigned int headers[13];
	FILE *outfile;
	int extrabytes;
	int paddedsize;
	int x;
	int y;
	int n;
	int red, green, blue;

	std::vector<unsigned char> vchanR(height * width);
	std::vector<unsigned char> vchanG(height * width);
	std::vector<unsigned char> vchanB(height * width);
	unsigned char *chanR = vchanR.data();
	unsigned char *chanG = vchanG.data();
	unsigned char *chanB = vchanB.data();
	checkCudaErrors(cudaMemcpy2D(chanR, (size_t)width, d_chanR, (size_t)pitchR,
								 width, height, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy2D(chanG, (size_t)width, d_chanG, (size_t)pitchR,
								 width, height, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy2D(chanB, (size_t)width, d_chanB, (size_t)pitchR,
								 width, height, cudaMemcpyDeviceToHost));

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
			red = chanR[y * width + x];
			green = chanG[y * width + x];
			blue = chanB[y * width + x];

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

// write bmp, input - RGB, device
int Image::WriteBMPi(const char *filename, const unsigned char *d_RGB, int pitch,
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