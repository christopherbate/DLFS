#include "QuickTestCPP.h"
#include "jpec.h"
#include "lib/data_loading/ImageLoader.hpp"
#include "lib/data_loading/LocalSource.hpp"
#include "lib/operations/Convolution.hpp"
#include "lib/tensor/Tensor.hpp"
#include "lib/utils/Timer.hpp"
#include <fstream>
#include <string>
#include <vector>

using namespace DLFS;
using namespace std;

/**
 * Tests filtering an image with a pre-specified filter
 * using the convolution operation.
 **/
void TestImage() {
    TestRunner::GetRunner()->AddTest(
        "ImageLoader", "Can load single image", []() {
            vector<uint8_t> buffer;
            std::string filename = "tests/data/img1.jpg";
            ifstream infile(filename.c_str(), ifstream::binary);
            if (!infile.is_open()) {
                throw std::runtime_error("Failed to open binary file " +
                                         filename);
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
        "ImageLoader", "Can load, decode JPEG and re-encode JPEG", []() {
            vector<uint8_t> buffer;

            ifstream infile("tests/data/img1.jpg", ifstream::binary);
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
            imgLoader.EncodeJPEG(tensor, "tests/data/img1.reencodeTest.jpg");
        });

    /**
     * Tests generating a simple image pyramid.
     **/
    TestRunner::GetRunner()->AddTest(
        "ImageLoader", "Filter image to create greyscale img.", []() {
            ADContext.Reset();

            vector<uint8_t> buffer;

            ifstream infile("tests/data/img1.jpg", ifstream::binary);
            if (!infile.is_open()) {
                throw std::runtime_error("Failed to open binary file.");
            }

            infile.seekg(0, std::ios::end);
            size_t file_size = infile.tellg();
            infile.seekg(0, std::ios::beg);

            buffer.resize(file_size);
            infile.read(reinterpret_cast<char *>(buffer.data()), buffer.size());

            ImageLoader imgLoader;
            auto tensor = CreateTensor<uint8_t>();

            // Load image and convert to greyscale.
            imgLoader.DecodeJPEG(buffer, tensor);
            auto tensorFloat = tensor->Cast<float>();
            auto filter = CreateFilter<float>(3, 1, 1, 1, "rgb2grey_filter",
                                              0.33333f, false);
            auto result =
                Convolve(tensorFloat, filter, Stride2D{1, 1}, Pad2D{0, 0});

            auto res_int = result->Cast<uint8_t>();

            buffer.clear();
            res_int->CopyBufferToHost(buffer);
            // imgLoader.EncodeJPEG(res_int,
            //  "tests/data/img1.reencode_rgb2grey.jpg");
            jpec_enc_t *e = jpec_enc_new(buffer.data(), res_int->GetShape()[2],
                                         res_int->GetShape()[1]);
            int len;
            const uint8_t *jpeg = jpec_enc_run(e, &len);

            ofstream outfile("tests/data/img1.reencode_grey.jpg",
                             ifstream::binary);
            outfile.write(reinterpret_cast<const char *>(jpeg), len);
            outfile.close();

            jpec_enc_del(e);

            ADContext.Reset();
        });

    TestRunner::GetRunner()->AddTest(
        "ImageLoader", "Can load jpegs (2-batched)", []() {
            ImageLoader imgLoader;
            LocalSource localSrc("tests/data/");
            TensorPtr<uint8_t> imgBatchTensor =
                std::make_shared<Tensor<uint8_t>>("ImageBatchTensor");

            vector<uint8_t> img1;
            vector<uint8_t> img2;

            localSrc.GetBlob("img1.jpg", img1);
            localSrc.GetBlob("img2.jpg", img2);

            std::vector<std::vector<uint8_t>> data = {img1, img2};

            double avgTime = 0.0;
            Timer timer;
            timer.tick();
            for (auto i = 0; i < 2; i++) {
                imgLoader.BatchDecodeJPEG(data, imgBatchTensor, 8);
                avgTime += timer.tick();
            }

            avgTime = avgTime / 2.0;
            LOG.INFO() << "Average image decode time (2 imgs): " << avgTime
                       << " msec";
        });
}

int main() {
    LOG.SetMinLevel(Debug);

    TestImage();

    TestRunner::GetRunner()->Run();
    TestRunner::GetRunner()->PrintSummary();

    return TestRunner::GetRunner()->GetRetCode();
}