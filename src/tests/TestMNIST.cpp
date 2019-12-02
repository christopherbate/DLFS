#include "../data_loading/DataLoader.hpp"
#include "../data_loading/ExampleSource.hpp"
#include "../data_loading/ImageLoader.hpp"
#include "../data_loading/LocalSource.hpp"
#include "../utils/Timer.hpp"
#include "QuickTestCPP.h"
#include "UnitTest.hpp"

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
            Timer batchTimer;

            DataLoader dataLoader("./mnist.train.db");
            dataLoader.SetBatchSize(5);

            ADContext.Reset();

            TensorPtr<float> f1 =
                CreateFilter<float>(1, 4, 3, 3, "conv1_filter", 0.01, false);

            LOG.INFO() << "Filter size: " << f1->GetLinearSize();

            /*size 3*3*1*4*/
            vector<float> f1_init = {/* first filter */
                                     0.0, 1.0 / (2 * 255.0f), 0.0, 0.0, 0.0,
                                     0.0, 0.0, -1.0 / (2 * 255.0f), 0.0,
                                     /* second filter */
                                     0.0, 0.0, 0.0, 1.0 / (2 * 255.0f), 0.0,
                                     -1.0 / (2 * 255.0f), 0.0, 0.0, 0.0,
                                     /* third filter */
                                     0.0, 0.0, 0.0, -1.0 / (2 * 255.0f), 0.0,
                                     1.0 / (2 * 255.0f), 0.0, 0.0, 0.0,
                                     /* fourth filter */
                                     0.0, -1.0 / (2 * 255.0f), 0.0, 0.0, 0.0,
                                     0.0, 0.0, 1.0 / (2 * 255.0f), 0.0};

            /*size 3*3*1*4*/
            vector<float> f2_init = {
                /* first filter row 1 */
                0.0, 0.0, 0.0, 0.0, 1.0 / (2.0), 1.0 / (2), 1.0 / (2),
                1.0 / (2), 0.0, 0.0, 0.0, 0.0,

                /*row 2*/
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

                /* row 3 */
                0.0, 0.0, 0.0, 0.0, -1.0 / (2.0f), -1.0 / (2.0f), -1.0 / (2.0f),
                -1.0 / (2.0f), 0.0, 0.0, 0.0, 0.0,

                /* second filter */
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                /*row 2*/
                -1.0 / (2 * 1.0), -1.0 / (2 * 1.0f), -1.0 / (2 * 1.0f),
                -1.0 / (2 * 1.0f), 0.0, 0.0, 0.0, 0.0, 1.0 / (2), 1.0 / (2),
                1.0 / (2), 1.0 / (2),
                /* row 3 */
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

                /* third filter  row 1*/
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                /*row 2*/
                1.0 / (2), 1.0 / (2), 1.0 / (2), 1.0 / (2), 0.0, 0.0, 0.0, 0.0,
                -1.0 / (2 * 1.0), -1.0 / (2 * 1.0f), -1.0 / (2 * 1.0f),
                -1.0 / (2 * 1.0f),
                /* row 3 */
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

                /* fourth filter */
                0.0, 0.0, 0.0, 0.0, 1.0 / (2 * 1.0f), 1.0 / (2 * 1.0f),
                1.0 / (2 * 1.0f), 1.0 / (2 * 1.0f), 0.0, 0.0, 0.0, 0.0,

                /*row 2*/
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

                /* row 3 */
                0.0, 0.0, 0.0, 0.0, -1.0 / (2 * 1.0f), -1.0 / (2 * 1.0f),
                -1.0 / (2 * 1.0f), -1.0 / (2 * 1.0f), 0.0, 0.0, 0.0, 0.0};

            LOG.INFO() << "Filter size: " << f1_init.size();
            cout << endl;

            f1->CopyBufferToDevice(f1_init);

            TensorPtr<float> f2 =
                CreateFilter<float>(4, 4, 3, 3, "conv2_filter", 0.01, true);

            f2->CopyBufferToDevice(f2_init);

            TensorPtr<float> f3 =
                CreateFilter<float>(4, 4, 3, 3, "conv3_filter", 0.11, true);

            f3->CopyBufferToDevice(f2_init);

            TensorPtr<float> f4 =
                CreateFilter<float>(4, 4, 3, 3, "conv4_filter", 0.1, true);

            f4->CopyBufferToDevice(f2_init);

            TensorPtr<float> f5 =
                CreateFilter<float>(4, 4, 3, 3, "conv5_filter", 0.1, true);
            f5->CopyBufferToDevice(f2_init);

            // Final filter for outputting to 10 classes
            TensorPtr<float> out_filter =
                CreateFilter<float>(4, 10, 2, 2, "out_filter", 0.11, true);

            std::vector<float> out_init = {
                -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
                0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,
                0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,
                -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,

                -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
                0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,
                0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,
                -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,

                -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
                0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,
                0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,
                -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,

                -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
                0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,
                0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,
                -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
            };

            out_filter->CopyBufferToDevice(out_init);

            LOG.INFO() << "Filter size: " << out_filter->GetLinearSize() << " "
                       << out_init.size();
            cout << endl;

            // Create a vector of trainable parametrs.
            std::vector<TensorBasePtr> params = {f2, f3, f4, f5, out_filter};

            for (auto i = 0; i < 100; i++) {
                batchTimer.tick();

                dataLoader.RunOnce();
                dataLoader.Summary();
                ADContext.Reset();

                ObjDetExampleBatch ex_batch = dataLoader.DequeBatch();

                // As a test, try to save this image.
                auto image = std::get<0>(ex_batch);
                auto cat_ids = std::get<2>(ex_batch);

                // WriteImageTensorPNG(
                // "./data/mnist_load_test" + to_string(i) + ".png", image);

                TensorPtr<float> imageBatch = image->Cast<float>();

                // First conv - 28x28 output
                TensorPtr<float> features =
                    imageBatch->Convolve(f1, Stride1, Pad1);
                features = features->ReLU();

                // Second convolution - size 14x14
                auto features2 = features->Convolve(f2, Stride2, Pad1);
                features2 = features2->ReLU();

                // Third convolution - size 7x7
                auto features3 = features2->Convolve(f3, Stride2, Pad1);
                features3 = features3->ReLU();

                // Fourth convolution - size 4x4
                auto features4 = features3->Convolve(f4, Stride2, Pad1);
                features4 = features4->ReLU();

                // Fifth convolution - size 2x2
                auto features5 = features4->Convolve(f5, Stride1, Pad0);
                features5 = features5->ReLU();

                auto out = features5->Convolve(out_filter, Stride1, Pad0);

                auto loss = out->SigmoidCELoss(cat_ids);

                std::vector<TensorBasePtr> featuresList = {
                    features, features2, features3, features4, features5};

                // LOG.INFO() << "Features: ";
                // for (auto &t : featuresList) {
                //     LOG.INFO() << t->GetName() << "\n   "
                //                << t->PrintTensor(false, true);
                // }

                ADContext.CalcGradient(loss);

                LOG.DEBUG() << "Parameters: ";
                for (auto &t : params) {
                    LOG.DEBUG() << t->GetName();
                }

                // LOG.DEBUG() << "F2 grad: " << f2->PrintTensor(true, true);

                LOG.DEBUG() << "Logits: " << out->PrintTensor(false, false);
                LOG.INFO() << "Loss: " << loss->PrintTensor(false, false);
                LOG.INFO() << "Labels: " << cat_ids->PrintTensor(false, false);

                // LOG.INFO() << "First filter gradient: ";
                // LOG.INFO() << f1->PrintTensor(true);

                // LOG.INFO() << "First filter: ";
                // LOG.INFO() << f1->PrintTensor();

                ADContext.StepOptimizer(params);

                // LOG.INFO() << "First filter, post optimizer step: ";
                // LOG.INFO() << f1->PrintTensor();

                // LOG.INFO() << imageBatch->GetDevicePointer();

                LOG.INFO() << ADContext.Print();

                LOG.INFO() << "Batch elapsed: " << batchTimer.tick_us()
                           << " usec";
            }
        });
}