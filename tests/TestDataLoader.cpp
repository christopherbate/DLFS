#include "QuickTestCPP.h"
#include "lib/Logging.hpp"
#include "lib/data_loading/DataLoader.hpp"
#include "lib/data_loading/ExampleSource.hpp"
#include "lib/data_loading/ImageLoader.hpp"
#include "lib/data_loading/LocalSource.hpp"
#include "lib/utils/Timer.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace DLFS;

void TestDataLoader() {
    TestRunner::GetRunner()->AddTest(
        "DataLoader",
        "Can intatiate data loader and retrieve batches (COCO VAL).", []() {
            DataLoader dataLoader;
            dataLoader.LoadDataset("tests/data/coco.val.ann.db");
            dataLoader.SetDataSourcePath("tests/data/");
            dataLoader.EnableJpegDecoder();
            dataLoader.SetBatchSize(2);

            for (auto i = 0; i < 2; i++) {
                dataLoader.RunOnce();
                dataLoader.Summary();
            }
        });

    TestRunner::GetRunner()->AddTest("DataLoader",
                                     "Can intatiate data loader and retrieve "
                                     "batches with image data in array",
                                     []() {
                                         DataLoader dataLoader(
                                             "tests/data/mnist.train.db");
                                         dataLoader.SetBatchSize(5);

                                         for (auto i = 0; i < 10; i++) {
                                             dataLoader.RunOnce();
                                             dataLoader.Summary();
                                         }
                                     });
}

int main() {
    LOG.SetMinLevel(Info);

    TestDataLoader();

    TestRunner::GetRunner()->Run();
    TestRunner::GetRunner()->PrintSummary();

    return TestRunner::GetRunner()->GetRetCode();
}