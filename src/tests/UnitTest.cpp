#include "QuickTestCPP.h"
#include "../data_loading/ExampleSource.hpp"
#include "../data_loading/DataLoader.hpp"
#include "../data_loading/ImageLoader.hpp"
#include "../data_loading/LocalSource.hpp"
#include "../utils/Timer.hpp"

#include <iostream>
#include <string>

using namespace DLFS;
using namespace std;

int main()
{
    TestRunner::GetRunner()->AddTest(
        "ExampleSource",
        "Can load test-case serialized dataset.",
        []() {
            ExampleSource annSrc;
            annSrc.init("./test.ann.db");
            return 1;
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
            if(annSrc.GetNumExamples() != 5000){
                return 0;
            }
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

    TestRunner::GetRunner()->AddTest(
        "DataLoader",
        "Can load test-case serialized dataset.",
        []() {
            DataLoader dataLoader("./test.ann.db","/models/data/coco/val2017/");
            float avgTime = 0.0;
            Timer timer;
            timer.tick();
            for(auto i =0; i <2; i++){
                dataLoader.GetBatch();
                avgTime += timer.tick();                
            }            
            avgTime = avgTime/2.0;
            cout << "Average example load time: " << avgTime << endl;
            return 1;
        });

    TestRunner::GetRunner()->AddTest(
        "ImageLoader",
        "Can load jpegs.",
        []() {
            ImageLoader imgLoader(1, 4);
            LocalSource localSrc("/models/data/coco/val2017/");

            auto img1 = localSrc.get_blob("000000000139.jpg");
            auto img2 = localSrc.get_blob("000000000285.jpg");

            std::vector<std::vector<uint8_t>> data = {img1, img2};

            float avgTime = 0.0;
            Timer timer;
            timer.tick();
            for(auto i =0; i < 1; i++){
                imgLoader.DecodeJPEG(data);
                avgTime += timer.tick();                
            }            
            avgTime = avgTime/2.0;
            cout << "Average image load time: " << avgTime << endl;
            return 1;
        });

    TestRunner::GetRunner()->Run();
    TestRunner::GetRunner()->PrintSummary();

    return TestRunner::GetRunner()->GetRetCode();
}