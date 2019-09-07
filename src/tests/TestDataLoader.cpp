#include "UnitTest.hpp"
#include "QuickTestCPP.h"
#include "../data_loading/ExampleSource.hpp"
#include "../data_loading/DataLoader.hpp"
#include "../data_loading/ImageLoader.hpp"
#include "../data_loading/LocalSource.hpp"
#include "../utils/Timer.hpp"
#include "Logging.hpp"

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
        "Can load serialized dataset.",
        []() {
            ExampleSource annSrc;
            annSrc.Init("coco.val.ann.db");
        });

    TestRunner::GetRunner()->AddTest(
        "ExampleSource",
        "Benchmark label loading.",
        []() {
            ExampleSource annSrc;
            annSrc.Init("./coco.val.ann.db");
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
            LocalSource localSrc("/home/chris/datasets/coco/val2017/");
            TensorPtr<uint8_t> imgBatchTensor = std::make_shared<Tensor<uint8_t>>("ImageBatchTensor");
            
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
            LOG.INFO() << "Average image decode time (2 imgs): " << avgTime
                 << " msec";

            auto count = 0;
            avgTime = 0.0;
            timer.tick();
            for (auto devPtr : imgBatchTensor->GetIterablePointersOverBatch())
            {
                auto tensorShape = imgBatchTensor->GetShape();
                string fileName = "./data/JpegLoad_non_batch_test_" + to_string(count) + ".bmp";
                writeBMPi(fileName.c_str(), devPtr,
                          3 * tensorShape[2], tensorShape[2], tensorShape[1]);
                avgTime += timer.tick();
                count++;
            }       
            avgTime = avgTime / 2.0;
            LOG.INFO() << "Average image write time (2 imgs): " << avgTime
                 << " msec";
        });    

    TestRunner::GetRunner()->AddTest(
        "DataLoader",
        "Can intatiate data loader and retrieve batches.",
        []() {
            DataLoader dataLoader("./coco.val.ann.db", "/home/chris/datasets/coco/val2017/");
            dataLoader.SetBatchSize(5);

            for (auto i = 0; i < 10; i++)
            {
                dataLoader.GetNextBatch();
                dataLoader.Summary();
            }            
        });    
}