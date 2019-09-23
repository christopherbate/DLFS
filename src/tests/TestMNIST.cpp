#include "UnitTest.hpp"
#include "QuickTestCPP.h"
#include "../data_loading/ExampleSource.hpp"
#include "../data_loading/DataLoader.hpp"
#include "../data_loading/ImageLoader.hpp"
#include "../data_loading/LocalSource.hpp"
#include "../utils/Timer.hpp"
#include "tensor/AutoDiff.hpp"

#include <cuda_runtime.h>

#include <string>
#include <iostream>
#include <vector>

using namespace std;
using namespace DLFS;

/**
 * Trains an MNIST digit classifier.
 */
void TestMNIST()
{
	TestRunner::GetRunner()->AddTest(
		"MnistTest",
		"Can load and train on MNIST",
		[]() {
			DataLoader dataLoader("./mnist.train.db");			
			dataLoader.SetBatchSize(5);

			ADContext.Reset();

			TensorPtr<float> f1 =
				ADContext.CreateFilter<float>(1, 1, 3,
											  "conv1_filter",
											  0.1, true);

			TensorPtr<float> f2 =
				ADContext.CreateFilter<float>(1, 1, 3,
											  "conv2_filter",
											  0.1, true);

			TensorPtr<float> f3 =
				ADContext.CreateFilter<float>(1, 1, 3,
											  "conv3_filter",
											  0.1, true);

			TensorPtr<float> f4 =
				ADContext.CreateFilter<float>(1, 1, 3,
											  "conv4_filter",
											  0.1, true);	

			TensorPtr<float> f5 =
				ADContext.CreateFilter<float>(1, 1, 4,
											  "conv5_filter",
											  0.1, true);												  										  

			TensorPtr<float> bias =
				ADContext.CreateTensor<float>({1, 28, 28, 1},
											  "bias", 0.1, true);

			for (auto i = 0; i < 1; i++)
			{
				dataLoader.RunOnce();
				dataLoader.Summary();

				ObjDetExampleBatch ex_batch = dataLoader.DequeBatch();

				TensorPtr<float> imageBatch = std::get<0>(ex_batch)->Cast<float>();

				TensorPtr<float> features = imageBatch->Convolve(f1, {1, 1}, {1, 1}) + bias;

				auto features2 = imageBatch->Convolve(f2, {1, 1}, {2, 2});

				LOG.INFO() << features2->PrintShape();

				auto features3 = features2->Convolve(f3, {1, 1}, {2, 2});

				LOG.INFO() << features3->PrintShape();

				auto features4 = features3->Convolve(f4, {1, 1}, {2, 2});

				LOG.INFO() << features4->PrintShape();

				auto features5 = features4->Convolve(f5, {0, 0}, {1, 1});

				LOG.INFO() << features5->PrintShape();

				auto cat_ids = std::get<2>(ex_batch);

				LOG.INFO() << "Labels: " << cat_ids->PrintShape();

				// TensorPtr<float>features = imageBatch + bias;
				// vector<float> buffer1;
				// imageBatch->CopyBufferToHost(buffer1);
				// vector<float> buffer2;
				// features->CopyBufferToHost(buffer2);
				// for(unsigned int i = 0; i < buffer1.size(); i++){
				// 	QTEqual(buffer2[i] - buffer1[i], 0.1f);
				// }
			}
		});
}