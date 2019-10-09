#include "../data_loading/DataLoader.hpp"
#include "../data_loading/ExampleSource.hpp"
#include "../data_loading/ImageLoader.hpp"
#include "../data_loading/LocalSource.hpp"
#include "../utils/Timer.hpp"
#include "QuickTestCPP.h"
#include "UnitTest.hpp"
#include "tensor/AutoDiff.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace DLFS;

/**
 * Trains an MNIST digit classifier.
 */
void TestMNIST() {
    TestRunner::GetRunner()->AddTest(
        "MnistTest", "Can load and train on MNIST", []() {
            DataLoader dataLoader("/home/chris/datasets/mnist/train.db");
            dataLoader.SetBatchSize(5);

            ADContext.Reset();

            TensorPtr<float> f1 = ADContext.CreateFilter<float>(
                1, 16, 3, "conv1_filter", 0.1, true);

            TensorPtr<float> f2 = ADContext.CreateFilter<float>(
                16, 32, 3, "conv2_filter", 0.1, true);

            TensorPtr<float> f3 = ADContext.CreateFilter<float>(
                32, 64, 3, "conv3_filter", 0.1, true);

            TensorPtr<float> f4 = ADContext.CreateFilter<float>(
                64, 128, 3, "conv4_filter", 0.1, true);

            TensorPtr<float> f5 = ADContext.CreateFilter<float>(
                128, 128, 4, "conv5_filter", 0.1, true);

            // Final filter for outputting to 10 classes
            TensorPtr<float> out_filter = ADContext.CreateFilter<float>(
                128, 10, 1, "out_filter", 0.1, true);

            for (auto i = 0; i < 1; i++) {
                dataLoader.RunOnce();
                dataLoader.Summary();

                ObjDetExampleBatch ex_batch = dataLoader.DequeBatch();

				// As a test, try to save this image.
				auto image = std::get<0>(ex_batch);					
				WriteImageTensorPNG("./data/mnist_load_test.png", image);

                TensorPtr<float> imageBatch =
                    std::get<0>(ex_batch)->Cast<float>();								

                // First
                TensorPtr<float> features =
                    imageBatch->Convolve(f1, {1, 1}, {1, 1});

                // Second convolution - size 14x14
                auto features2 = features->Convolve(f2, {1, 1}, {2, 2});

                // Third convolution - size 7x7
                auto features3 = features2->Convolve(f3, {1, 1}, {2, 2});

                LOG.INFO() << features3->PrintShape();

                // Fourth convolution - size 4x4
                auto features4 = features3->Convolve(f4, {1, 1}, {2, 2});

                LOG.INFO() << features4->PrintShape();

                // Fifth convolution - size 2x2
                auto features5 = features4->Convolve(f5, {0, 0}, {1, 1});

                LOG.INFO() << features5->PrintShape();

                auto out = features5->Convolve(out_filter, {0, 0}, {1, 1});

                // Calculate loss function
                auto cat_ids = std::get<2>(ex_batch);

                auto loss = out->SigmoidCELoss(cat_ids);

                LOG.INFO() << "Loss Shape: " << loss->PrintShape();
            }
        });
}