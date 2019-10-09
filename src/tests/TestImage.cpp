#include "QuickTestCPP.h"
#include "UnitTest.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/AutoDiff.hpp"
#include "data_loading/ImageLoader.hpp"

#include <fstream>
#include <vector>

using namespace DLFS;
using namespace std;

/**
* Tests filtering an image with a pre-specified filter
* using the convolution operation.
**/
void TestImage()
{
    TestRunner::GetRunner()->AddTest(
        "Image Utilities",
        "Can load single image",
        []() {
            vector<uint8_t> buffer;

            ifstream infile("./data/img1.jpg", ifstream::binary);
            if (!infile.is_open())
            {
                throw std::runtime_error("Failed to open binary file.");
            }

            infile.seekg(0, std::ios::end);
            size_t file_size = infile.tellg();
            infile.seekg(0, std::ios::beg);

            buffer.resize(file_size);
            infile.read(reinterpret_cast<char *>(buffer.data()),
                        buffer.size());

            ImageLoader imgLoader;

            TensorPtr<uint8_t> tensor = ADContext.CreateTensor<uint8_t>();

            imgLoader.DecodeJPEG(buffer, tensor);
        });

    TestRunner::GetRunner()->AddTest(
        "Image Utilities",
        "Can load JPEG and save as BMP",
        []() {
            vector<uint8_t> buffer;

            ifstream infile("./data/img1.jpg", ifstream::binary);
            if (!infile.is_open())
            {
                throw std::runtime_error("Failed to open binary file.");
            }

            infile.seekg(0, std::ios::end);
            size_t file_size = infile.tellg();
            infile.seekg(0, std::ios::beg);

            buffer.resize(file_size);
            infile.read(reinterpret_cast<char *>(buffer.data()),
                        buffer.size());

            ImageLoader imgLoader;

            TensorPtr<uint8_t> tensor = ADContext.CreateTensor<uint8_t>();

            imgLoader.DecodeJPEG(buffer, tensor);

            WriteImageTensorBMP("./data/img1_encode.bmp", tensor);
        });

    TestRunner::GetRunner()->AddTest(
        "Image Utilities",
        "Can load JPEG and save as color (RGB) PNG",
        []() {
            vector<uint8_t> buffer;

            ifstream infile("./data/img1.jpg", ifstream::binary);
            if (!infile.is_open())
            {
                throw std::runtime_error("Failed to open binary file.");
            }

            infile.seekg(0, std::ios::end);
            size_t file_size = infile.tellg();
            infile.seekg(0, std::ios::beg);

            buffer.resize(file_size);
            infile.read(reinterpret_cast<char *>(buffer.data()),
                        buffer.size());

            ImageLoader imgLoader;

            TensorPtr<uint8_t> tensor = ADContext.CreateTensor<uint8_t>();

            imgLoader.DecodeJPEG(buffer, tensor);

            WriteImageTensorPNG("./data/img1_encode.png", tensor);
        });

    TestRunner::GetRunner()->AddTest(
        "Image Utilities",
        "Can load JPEG and save as greyscale (8bit) PNG",
        []() {
            vector<uint8_t> buffer;

            ifstream infile("./data/img1.jpg", ifstream::binary);
            if (!infile.is_open())
            {
                throw std::runtime_error("Failed to open binary file.");
            }

            infile.seekg(0, std::ios::end);
            size_t file_size = infile.tellg();
            infile.seekg(0, std::ios::beg);

            buffer.resize(file_size);
            infile.read(reinterpret_cast<char *>(buffer.data()),
                        buffer.size());

            ImageLoader imgLoader;
            TensorPtr<uint8_t> tensor = ADContext.CreateTensor<uint8_t>();

            // Load image and convert to greyscale.
            imgLoader.DecodeJPEG(buffer, tensor);

            auto tensorFloat = tensor->Cast<float>();
            auto filter = ADContext.CreateFilter<float>(3, 1, 1, "rgb2grey_filter", 0.33333f, false);
            Pad2d padding = {0, 0};
            tensorFloat = tensorFloat->Convolve(filter, padding);

            auto greyImage = tensorFloat->Cast<uint8_t>();

            WriteImageTensorPNG("./data/img1_encode_grey.png", greyImage);
        });

    /**
     * Tests generating a simple image pyramid.
     **/
    TestRunner::GetRunner()->AddTest(
        "Image Utilities",
        "Filter image with 3x3 lowpass filter.",
        []() {
            vector<uint8_t> buffer;

            ifstream infile("./data/img1.jpg", ifstream::binary);
            if (!infile.is_open())
            {
                throw std::runtime_error("Failed to open binary file.");
            }

            infile.seekg(0, std::ios::end);
            size_t file_size = infile.tellg();
            infile.seekg(0, std::ios::beg);

            buffer.resize(file_size);
            infile.read(reinterpret_cast<char *>(buffer.data()),
                        buffer.size());

            ImageLoader imgLoader;
            TensorPtr<uint8_t> tensor = ADContext.CreateTensor<uint8_t>();

            // Load image and convert to greyscale.
            imgLoader.DecodeJPEG(buffer, tensor);
            auto tensorFloat = tensor->Cast<float>();
            auto filter = ADContext.CreateFilter<float>(3, 1, 1, "rgb2grey_filter", 0.33333f, false);
            Pad2d padding = {0, 0};
            tensorFloat = tensorFloat->Convolve(filter, padding);

            // Low pass filter - loop
            padding = {1, 1};
            Stride2d stride = {2, 2};
            filter = ADContext.CreateFilter<float>(1, 1, 3, "3x3LowPass", (1.0f / 9.0f), false);

            for (int i = 0; i < 5; i++)
            {            
                tensorFloat = tensorFloat->Convolve(filter, padding, stride);
                auto greyImage = tensorFloat->Cast<uint8_t>();
                const string filename = "./data/lowpass_test/img1_lowpass" + to_string(i+1) + ".png";
                WriteImageTensorPNG(filename.c_str(), greyImage);
            }
        });
}