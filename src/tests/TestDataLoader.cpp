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

            QuickTest::Equal(annSrc.GetNumExamples(), (unsigned int)4952);

            for (unsigned int i = 0; i < 1000; i++)
            {
                const Example *ex = annSrc.GetExample(i);
                auto file_name = ex->file_name()->str();
                avgTime += timer.tick_us();
            }
            avgTime = avgTime * 1e-3;

            cout << "Average example load time: " << avgTime << "uSec " << endl;
        });

    TestRunner::GetRunner()->AddTest(
        "ImageLoader",
        "Can load jpegs (2-batched)",
        []() {
            ImageLoader imgLoader;
            LocalSource localSrc("/home/chris/datasets/coco/val2017/");
            TensorPtr<uint8_t> imgBatchTensor = std::make_shared<Tensor<uint8_t>>("ImageBatchTensor");

            vector<uint8_t> img1;
            vector<uint8_t> img2;

            localSrc.GetBlob("000000000139.jpg", img1);
            localSrc.GetBlob("000000000285.jpg", img2);

            std::vector<std::vector<uint8_t>> data = {img1, img2};

            double avgTime = 0.0;
            Timer timer;
            timer.tick();
            for (auto i = 0; i < 2; i++)
            {
                imgLoader.BatchDecodeJPEG(data, imgBatchTensor, 8);
                avgTime += timer.tick();
            }

            avgTime = avgTime / 2.0;
            LOG.INFO() << "Average image decode time (2 imgs): " << avgTime
                       << " msec";           
        });

    TestRunner::GetRunner()->AddTest(
        "DataLoader",
        "Can intatiate data loader and retrieve batches (COCO VAL).",
        []() {
            DataLoader dataLoader;
            dataLoader.LoadDataset("./coco.val.ann.db");
            dataLoader.SetDataSourcePath("/home/chris/datasets/coco/val2017/");
            dataLoader.EnableJpegDecoder();
            dataLoader.SetBatchSize(5);

            for (auto i = 0; i < 10; i++)
            {
                dataLoader.RunOnce();
                dataLoader.Summary();
            }
        });

    TestRunner::GetRunner()->AddTest(
        "DataLoader",
        "Can intatiate data loader and retrieve batches with image data in array",
        []() {
            DataLoader dataLoader("./mnist.train.db");
            dataLoader.SetBatchSize(5);

            for (auto i = 0; i < 10; i++)
            {
                dataLoader.RunOnce();
                dataLoader.Summary();
            }
        });
}