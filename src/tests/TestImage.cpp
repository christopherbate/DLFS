#include "QuickTestCPP.h"
#include "UnitTest.hpp"
#include "tensor/Tensor.hpp"

#include "data_loading/ImageLoader.hpp"

#include <fstream>
#include <vector>

using namespace DLFS;
using namespace std;

/**
 * Tests filtering an image with a pre-specified filter
 * using the convolution operation.
 **/
void TestImage() {
    TestRunner::GetRunner()->AddTest(
        "Image Utilities", "Can load single image", []() {
            vector<uint8_t> buffer;

            ifstream infile("./data/img1.jpg", ifstream::binary);
            if (!infile.is_open()) {
                throw std::runtime_error("Failed to open binary file.");
            }

            infile.seekg(0, std::ios::end);
            size_t file_size = infile.tellg();
            infile.seekg(0, std::ios::beg);

            buffer.resize(file_size);
            infile.read(reinterpret_cast<char *>(buffer.data()), buffer.size());

            ImageLoader imgLoader;

            TensorPtr<uint8_t> tensor = CreateTensor<uint8_t>();

            imgLoader.DecodeJPEG(buffer, tensor);
        });

    TestRunner::GetRunner()->AddTest(
        "Image Utilities", "Can load, decode JPEG and re-encode JPEG", []() {
            vector<uint8_t> buffer;

            ifstream infile("./data/img1.jpg", ifstream::binary);
            if (!infile.is_open()) {
                throw std::runtime_error("Failed to open binary file.");
            }

            infile.seekg(0, std::ios::end);
            size_t file_size = infile.tellg();
            infile.seekg(0, std::ios::beg);

            buffer.resize(file_size);
            infile.read(reinterpret_cast<char *>(buffer.data()), buffer.size());

            ImageLoader imgLoader;

            TensorPtr<uint8_t> tensor = CreateTensor<uint8_t>();

            imgLoader.DecodeJPEG(buffer, tensor);
            LOG.DEBUG() << "Decoded into tensor of shape "
                        << tensor->PrintShape();
            // LOG.DEBUG() << tensor->PrintTensor();
            imgLoader.EncodeJPEG(tensor, "./data/img1.reencodeTest.jpg");
        });

    /**
     * Tests generating a simple image pyramid.
     **/
    TestRunner::GetRunner()->AddTest(
        "Image Utilities", "Filter image with 3x3 lowpass filter.", []() {
            vector<uint8_t> buffer;

            ifstream infile("./data/img1.jpg", ifstream::binary);
            if (!infile.is_open()) {
                throw std::runtime_error("Failed to open binary file.");
            }

            infile.seekg(0, std::ios::end);
            size_t file_size = infile.tellg();
            infile.seekg(0, std::ios::beg);

            buffer.resize(file_size);
            infile.read(reinterpret_cast<char *>(buffer.data()), buffer.size());

            ImageLoader imgLoader;
            TensorPtr<uint8_t> tensor = CreateTensor<uint8_t>();

            // Load image and convert to greyscale.
            imgLoader.DecodeJPEG(buffer, tensor);
            auto tensorFloat = tensor->Cast<float>();
            auto filter = CreateFilter<float>(3, 1, 1, 1, "rgb2grey_filter",
                                              0.33333f, false);
            tensorFloat = tensorFloat->Convolve(filter, Stride1, Pad0);

            // Low pass filter - loop
            // padding = {1, 1};
            // Stride2D stride = {2, 2};
            filter = CreateFilter<float>(1, 1, 3, 3, "3x3LowPass",
                                         (1.0f / 9.0f), false);
        });
}