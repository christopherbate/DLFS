#include "QuickTestCPP.h"
#include "../data_loading/AnnotationSource.hpp"
#include "../utils/Timer.hpp"

#include <iostream>
#include <string>

using namespace DLFS;
using namespace std;

int main()
{
    TestRunner::GetRunner()->AddTest(
        "AnnotationSource",
        "Can load test-case serialized dataset.",
        []() {
            AnnotationSource annSrc;
            annSrc.init("./test.ann.db");
            return 1;
        });

    TestRunner::GetRunner()->AddTest(
        "AnnotationSource",
        "Benchmark label loading.",
        []() {
            AnnotationSource annSrc;
            annSrc.init("./test.ann.db");
            float avgTime = 0.0;
            Timer timer;
            timer.tick();
            for (unsigned int i = 0; i < 5000; i++)
            {
                const Example *ex = annSrc.GetExample(i);                
                auto file_name = ex->file_name()->str();                
                avgTime += timer.tick();
            }
            avgTime = avgTime/5000.0;

            cout << "Average example load time: " << avgTime << endl;

            return 1;
        });

    TestRunner::GetRunner()->Run();
    TestRunner::GetRunner()->PrintSummary();

    return TestRunner::GetRunner()->GetRetCode();
}