#include "UnitTest.hpp"
#include "QuickTestCPP.h"
#include "../data_loading/ExampleSource.hpp"
#include "../data_loading/DataLoader.hpp"
#include "../data_loading/ImageLoader.hpp"
#include "../data_loading/LocalSource.hpp"
#include "../utils/Timer.hpp"

#include <cuda_runtime.h>

#include <string>
#include <iostream>

using namespace std;
using namespace DLFS;

void TestDataLoader()
{

    TestRunner::GetRunner()->AddTest(
        "ExampleSource",
        "Can load test-case serialized dataset.",
        []() {
            ExampleSource annSrc;
            annSrc.init("./test.ann.db");
        });

    TestRunner::GetRunner()->AddTest(
        "ExampleSource",
        "Benchmark label loading.",
        []() {
            ExampleSource annSrc;
            annSrc.init("./test.ann.db");
            float avgTime = 0.0;
            Timer timer;
            timer.tick();

            QuickTest::Equal(annSrc.GetNumExamples(), (unsigned int)5000);

            for (unsigned int i = 0; i < 5000; i++)
            {
                const Example *ex = annSrc.GetExample(i);
                auto file_name = ex->file_name()->str();
                avgTime += timer.tick();
            }
            avgTime = avgTime / 5000.0;

            cout << "Average example load time: " << avgTime << endl;
        });

    TestRunner::GetRunner()->AddTest(
        "ImageLoader",
        "Can load jpegs.",
        []() {
            ImageLoader imgLoader(2, 4);
            LocalSource localSrc("/models/data/coco/val2017/");
            Tensor<uint8_t> imgBatchTensor;

            auto img1 = localSrc.get_blob("000000000139.jpg");
            auto img2 = localSrc.get_blob("000000000285.jpg");

            std::vector<std::vector<uint8_t>> data = {img1, img2};

            float avgTime = 0.0;
            Timer timer;
            timer.tick();
            for (auto i = 0; i < 1; i++)
            {
                imgLoader.DecodeJPEG(data, imgBatchTensor);
                avgTime += timer.tick();
            }

            auto count = 0;
            for (auto devPtr : imgBatchTensor.GetIterablePointersOverBatch())
            {
                auto tensorShape = imgBatchTensor.GetShape();
                string fileName = "./data/test_" + to_string(count) + ".bmp";
                writeBMPi(fileName.c_str(), devPtr,
                          3 * tensorShape[2], tensorShape[2], tensorShape[1]);
                count++;
            }       
            avgTime = avgTime / 2.0;
            cout << "Average image decode time (2 imgs): " << avgTime
                 << " msec" << endl;
        });

    TestRunner::GetRunner()->AddTest(
        "DataLoader",
        "Can load test-case serialized dataset.",
        []() {
            DataLoader dataLoader("./test.ann.db", "/models/data/coco/val2017/", 5);
            for (auto i = 0; i < 10; i++)
            {
                dataLoader.GetNextBatch();
                dataLoader.Summary();
            }
        });

    TestRunner::GetRunner()->AddTest(
        "DataLoader",
        "Can load images from LocalSource and Decode.",
        []() {
            DataLoader dataLoader("./test.ann.db", "/models/data/coco/val2017/", 10);
            dataLoader.Summary();
        });
}