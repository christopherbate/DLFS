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
#include <vector>

using namespace std;
using namespace DLFS;

void TestDataLoader()
{

    TestRunner::GetRunner()->AddTest(
        "ExampleSource",
        "Can load test-case serialized dataset.",
        []() {
            ExampleSource annSrc;
            annSrc.Init("./test.ann.db");
        });

    TestRunner::GetRunner()->AddTest(
        "ExampleSource",
        "Benchmark label loading.",
        []() {
            ExampleSource annSrc;
            annSrc.Init("./test.ann.db");
            double avgTime = 0.0;
            Timer timer;
            timer.tick_us();

            QuickTest::Equal(annSrc.GetNumExamples(), (unsigned int)5000);

            for (unsigned int i = 0; i < 1000; i++)
            {
                const Example *ex = annSrc.GetExample(i);
                auto file_name = ex->file_name()->str();
                avgTime += timer.tick_us();
            }
            avgTime = avgTime*1e-3;

            cout << "Average example load time: " << avgTime << "uSec " << endl;
        });

    TestRunner::GetRunner()->AddTest(
        "ImageLoader",
        "Can load jpegs (2-batched)",
        []() {
            ImageLoader imgLoader;
            LocalSource localSrc("/models/data/coco/val2017/");
            Tensor<uint8_t> imgBatchTensor;
            
            auto img1 = localSrc.get_blob("000000000139.jpg");
            auto img2 = localSrc.get_blob("000000000285.jpg");

            std::vector<std::vector<uint8_t>> data = {img1, img2};

            double avgTime = 0.0;
            Timer timer;
            timer.tick();
            for (auto i = 0; i < 2; i++)
            {
                imgLoader.DecodeJPEG(data, imgBatchTensor, 8);
                avgTime += timer.tick();
            }

            avgTime = avgTime / 2.0;
            cout << "Average image decode time (2 imgs): " << avgTime
                 << " msec" << endl;

            auto count = 0;
            avgTime = 0.0;
            timer.tick();
            for (auto devPtr : imgBatchTensor.GetIterablePointersOverBatch())
            {
                auto tensorShape = imgBatchTensor.GetShape();
                string fileName = "./data/JpegLoad_non_batch_test_" + to_string(count) + ".bmp";
                writeBMPi(fileName.c_str(), devPtr,
                          3 * tensorShape[2], tensorShape[2], tensorShape[1]);
                avgTime += timer.tick();
                count++;
            }       
            avgTime = avgTime / 2.0;
            cout << "Average image write time (2 imgs): " << avgTime
                 << " msec" << endl;
        });

     TestRunner::GetRunner()->AddTest(
        "ImageLoader",
        "Can load jpegs (8-batched)",
        []() {
            // 8 images per batch, 8 host threads.
            ImageLoader imgLoader;
            LocalSource localSrc("./data/");
            Tensor<uint8_t> imgBatchTensor;

            vector<vector<uint8_t>> data;
            
            for(unsigned int i = 1; i < 8; i++)
            {
                string filename = "img"+to_string(i)+".jpg";
                data.push_back(localSrc.get_blob(filename));
            }

            double avgTime = 0.0;
            Timer timer;
            timer.tick();
            for (auto i = 0; i < 2; i++)
            {                
                imgLoader.DecodeJPEG(data, imgBatchTensor, 8);
                avgTime += timer.tick();
            }

            avgTime = avgTime / 2.0;
            cout << "Average image decode time (8 imgs): " << avgTime
                 << " msec" << endl;

            auto count = 0;
            avgTime = 0.0;
            timer.tick();
            for (auto devPtr : imgBatchTensor.GetIterablePointersOverBatch())
            {
                auto tensorShape = imgBatchTensor.GetShape();
                string fileName = "./data/JpegLoad_8batched_" + to_string(count) + ".bmp";
                writeBMPi(fileName.c_str(), devPtr,
                          3 * tensorShape[2], tensorShape[2], tensorShape[1]);
                avgTime += timer.tick();
                count++;
            }       
            avgTime = avgTime / 8.0;
            cout << "Average image write time (per img): " << avgTime
                 << " msec" << endl;
        });

    TestRunner::GetRunner()->AddTest(
        "DataLoader",
        "Can load test-case serialized dataset.",
        []() {
            DataLoader dataLoader("./test.ann.db", "/models/data/coco/val2017/");
            dataLoader.SetBatchSize(5);

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
            DataLoader dataLoader("./test.ann.db", "/models/data/coco/val2017/");
            dataLoader.SetBatchSize(10);
            
            dataLoader.Summary();
        });
}