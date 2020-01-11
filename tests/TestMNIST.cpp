#include "QuickTestCPP.h"
#include "lib/data_loading/DataLoader.hpp"
#include "lib/data_loading/ExampleSource.hpp"
#include "lib/data_loading/ImageLoader.hpp"
#include "lib/data_loading/LocalSource.hpp"
#include "lib/operations/Convolution.hpp"
#include "lib/operations/SigmoidCrossEntropy.hpp"
#include "lib/utils/Timer.hpp"

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

            DataLoader dataLoader("tests/data/mnist.train.db");
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

            for (auto i = 0; i < 10; i++) {
                batchTimer.tick();

                dataLoader.RunOnce();
                dataLoader.Summary();
                ADContext.Reset();

                ObjDetExampleBatch ex_batch = dataLoader.DequeBatch();

                auto image = std::get<0>(ex_batch);
                auto cat_ids = std::get<2>(ex_batch);

                TensorPtr<float> imageBatch = image->Cast<float>();

                // First conv - 28x28 output
                TensorPtr<float> features =
                    MakeConvolve(imageBatch, f1, Stride1, Pad1);
                features = features->ReLU();

                // Second convolution - size 14x14
                auto features2 = MakeConvolve(features, f2, Stride2, Pad1);
                features2 = features2->ReLU();

                // Third convolution - size 7x7
                auto features3 = MakeConvolve(features2, f3, Stride2, Pad1);
                features3 = features3->ReLU();

                // Fourth convolution - size 4x4
                auto features4 = MakeConvolve(features3, f4, Stride2, Pad1);
                features4 = features4->ReLU();

                // Fifth convolution - size 2x2
                auto features5 = MakeConvolve(features4, f5, Stride1, Pad0);
                features5 = features5->ReLU();

                // Final convolution - size Batchx1x1x10
                auto logits =
                    MakeConvolve(features5, out_filter, Stride1, Pad0);

                // auto loss = out->SigmoidCELoss(cat_ids);
                auto loss = SigmoidCELoss(logits, cat_ids, true);

                std::vector<TensorBasePtr> featuresList = {
                    features, features2, features3, features4, features5};
                ADContext.CalcGradient(loss);

                LOG.DEBUG() << "Parameters: ";
                for (auto &t : params) {
                    LOG.DEBUG() << t->GetName();
                }

                LOG.DEBUG() << "Logits: " << logits->PrintTensor(false, false);
                LOG.INFO() << "Loss: " << loss->PrintTensor(false, false);
                LOG.INFO() << "Labels: " << cat_ids->PrintTensor(false, false);

                ADContext.StepOptimizer(params);

                LOG.INFO() << ADContext.Print();
                LOG.INFO() << "Batch elapsed: " << std::fixed
                           << batchTimer.tick_us() << " usec";
            }
        });
}

int main() {
    LOG.SetMinLevel(Info);

    TestMNIST();

    TestRunner::GetRunner()->Run();
    TestRunner::GetRunner()->PrintSummary();

    return TestRunner::GetRunner()->GetRetCode();
}