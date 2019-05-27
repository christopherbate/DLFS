#include <iostream>
#include <fstream>
#include "Logging.hpp"
#include "GPU.hpp"
#include "Image.hpp"
#include "Network.hpp"

#include "layers/InputLayer.hpp"
#include "layers/ConvLayer.hpp"

using namespace std;
using namespace DLFS;

int main(int argc, char **argv)
{
    GPU gpu;

    ImageLoader::FileNames fileNames;

    for (int j = 0; j < 1; j++)
    {
        for (int i = 0; i < 8; i++)
        {
            string fileName = "./data/img" + std::to_string(i + 1) + ".jpg";
            fileNames.push_back(fileName);
        }
    }

    ImageLoader imgLoader(fileNames);

    Network<float> network;
    
    NHWCBuffer<float> testData;
    network.CreateTestTensor( testData );

    cout << "Testing input layer creation "<< endl;

    InputLayer<float> inputLayer( network.GetCUDNN(), NULL );

    TensorDims inputDim( 16, 128, 128, 128);    
    inputLayer.SetInputDim( inputDim );

    cout << "Testing conv layer creation "<< endl;
    FilterDims filterDims( 128, 256, 3, 3); /* in out w h*/
    ConvLayer<float> convLayer( network.GetCUDNN(), (InputLayer<float>*)&inputLayer);
    convLayer.SetFilerDim(filterDims);

    convLayer.PrintAllWorkspaces();

    cout << "InputLayer input dim: " << inputLayer.GetInputDims() << endl;
    cout << "InputLayer output dim: " << inputLayer.GetOutputDims() << endl;
    cout << "Conv Layer input dimensions: "  << convLayer.GetInputDims() << endl;  
    cout << "Conv Layer output dimensions: "  << convLayer.GetOutputDims() << endl;      

    convLayer.Print();

    convLayer.FindBestAlgorithm();

    // Setup for forward pass.
    inputLayer.AllocateBuffers();
    convLayer.AllocateBuffers();

    // Run forward pass.
    // Copy in image.

    convLayer.Forward();
    
    return 0;
}