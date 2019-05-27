#ifndef IMAGE_H_
#define IMAGE_H_

#include <string>
#include <vector>
#include <fstream>
#include <cstdint>

#include <nvjpeg.h>

namespace DLFS
{

class Image
{
public:
    typedef std::vector<unsigned char> Buffer;

    void LoadJPEG();
    Buffer buffer;
    uint32_t width;
    uint32_t height;
    nvjpegImage_t &jpeg;
    nvjpegImage_t jpegSize;
    size_t imgSize;
    std::string filepath;
    
    Image( nvjpegImage_t &jpegDest );
    ~Image();

    void Read( const std::string &filename );

    void PrepareBuffers( nvjpegHandle_t &jpegHandle, nvjpegOutputFormat_t outputFormat );

    static int WriteBMPi(const char *filename, const unsigned char *d_RGB, int pitch,
              int width, int height);

    static int WriteBMP(const char *filename, const unsigned char *d_chanR, int pitchR,
             const unsigned char *d_chanG, int pitchG,
             const unsigned char *d_chanB, int pitchB, int width, int height);

    static unsigned int FileSize( std::ifstream &infile );
};

class ImageLoader
{   
public:
    typedef std::vector<std::string> FileNames;
    typedef std::vector<std::vector<char>> FileDataBuffer;

private:
    nvjpegHandle_t m_jpegHandle;
    nvjpegJpegState_t m_jpegState;
    nvjpegOutputFormat_t m_outputFormat;
    cudaStream_t m_stream;

    std::vector<nvjpegImage_t> m_jpegs;
    std::vector<size_t> m_imgLengths;        
    std::vector<Image> m_images;

    int m_maxHostThread;
    int m_batchSize;
    
public:
    ImageLoader( FileNames fileNames );
    ~ImageLoader();

    void ReadBatch();

    void WriteBatchBMP();

    static const char *SamplingTypeString( nvjpegChromaSubsampling_t type);
};


} // namespace DLFS

#endif // !IMAGE_H