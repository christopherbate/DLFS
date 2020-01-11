#include "QuickTestCPP.h"
#include "lib/Logging.hpp"
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

int main() {
    LOG.SetMinLevel(Info);

    TestRunner::GetRunner()->AddTest(
        "ExampleSource", "Can load serialized dataset.", []() {
            ExampleSource annSrc;
            annSrc.Init("tests/data/coco.val.ann.db");
        });

    TestRunner::GetRunner()->AddTest(
        "ExampleSource", "Benchmark label loading.", []() {
            ExampleSource annSrc;
            annSrc.Init("tests/data/coco.val.ann.db");
            double avgTime = 0.0;
            Timer timer;
            timer.tick_us();

            QuickTest::Equal(annSrc.GetNumExamples(), (unsigned int)4952);

            for (unsigned int i = 0; i < 1000; i++) {
                const Example *ex = annSrc.GetExample(i);
                auto file_name = ex->file_name()->str();
                avgTime += timer.tick_us();
            }
            avgTime = avgTime * 1e-3;

            cout << "Average example load time: " << avgTime << "uSec " << endl;
        });

    TestRunner::GetRunner()->Run();
    TestRunner::GetRunner()->PrintSummary();

    return TestRunner::GetRunner()->GetRetCode();
}