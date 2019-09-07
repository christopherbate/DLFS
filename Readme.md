# Deep Learning from Scratch

A deep learning C++ library built on CUDNN. The aim of this library is to provide simple interfaces to build deep learning 
alogorithms (mostly object detection) while maximizing speed and the latest features provided by NVIDIA RTX/Turing cards.


# Prerequisites

This library is GPU only. You must have a system with an NVIDIA GPU card, preferably a recent card. 

We develop on Ubuntu 19.04 systems with Intel CPUs and RTX 2080 series cards. Recommend 
you isolate your development environment from the various other software that are trying 
to manipulate CUDA libraries and runtimes (e.g. `conda`).

Requires :
- Nvidia driver (we're on 418)
- CUDA >= 10.1 (we're developing on the bleeding edge, `cuda_10.1.243`. We install with the `run` file on 19.04)
- CuDNN 7.6.3 (recommend installing manually with the `tar` file, not the `deb`)
- NvJPEG
- flatbuffer compiler (see below)
- cmake
- COCO 2017 images and labels (see below)

## flatbufferc
When setting up a new environment, you must clone and build Google's flatbuffer compiler.

```
git clone git@github.com:google/flatbuffers.git
cd flatbuffers
mkdir build && cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ..
make -j[num cores]
sudo make install (or add the created bin to your path)
```

## COCO 

We setup all datasets in `~/datasets`.

```
mkdir -p  ~/datasets/coco && cd ~/datasets/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip
unzip annotations_trainval2017.zip
```

# Build

Build is provided by `make`. Binaries go into `bin`.


# Quickstart 

For fast usage, please look in the `examples` folder. There are examples for implementing basic object detection pipelines.


# Overview 

To train a deep learning model in a reproducable way, you need minimally the following things:

 1. Data and annotations
 2. A model
 3. Optimization/training management system that takes model and data and outputs trained weights, including the ability to optimizer accross multi-GPU and multi-host systems.
 4. Visualization/Monitoring: an external system for consuming sample outputs of the model in realtime so you can 
 monitor progress.
 5. A standard way for serializing the model and its weights as well as the predictions
 6. The ability to serialize stateful parts of the training system to restore from checkpoint if training is interrupted.
 7. The necessary functions for efficiently calculating various performance metrics (e.g. mAP) during/after training.
 8. A system for data augmentation, which can considerably improve model robustness and performance

For inference, you need the following things:

1. A way to specify models which should be served and loading those model into the inference engine
2. A method for feeding data to the models efficiently.
3. Excellent logging and monitoring for production

This repo intends to provide all of the above in a compact library for object detection models, focusing 
on fast turn-around between data collection, labeling, training, inference, back to labeling. It provides serialized formats for streaming predictions and labels to/from an system during training for effective real-time training. To do this, we leverage CuDNN for nearly all primitives within the deep learning engine, resulting in a simple, highly performant, readable and maintainable code base. 

Unlike general deep learning frameworks such as PyTorch and Tensorflow, this library is completely biased towards vision systems. There are only a small set of operations which are implemented (e.g. convolution), but they are finely tuned for speed and the latest features (e.g. mixed-precision training) and research (so we have new kidnds of convolution such as seperable convolution, etc.). There are basic primitives for bounding box operations for dense anchor-based systems as well as post-processing functions such as Non-Max Suppression. 

Futhermore, we assume all input data is in the form of images which will be turned into 3-channel tensors. There is no notion of general n-d tensor operations like in Tensorflow, PyTorch and Numpy. Nevertheless, a great deal of interesting models and research questions can be tackled within this framework, all within a simple system that can take the model into "production" environments.

The following sections detail components in detail and walkthrough the creation and training of a system trained on MS COCO.

## Data, Annotations, and Augmentation

In this system, the raw training data consists of a set of binary objects sitting in a datstore. Two forms are supported: local disk and GCP Cloud Storage, but extending to other systems is as simple as implementing the abstract `DataSource` class. Labels/annotations and image meta-data are stored in some sort of database allowing for efficient querying and maintainability, and the interface is defined by the abstract `AnnotationStore` class. Currently, `SQLiteAnnStore` is the only implementation. The `AnnotationStore` conatains image metadata to allow the data-loading functions to pull binary objects from `DataSource`.

We must load binary objects from the `DataSource` and preprocess them into a form which can be consumed by the model. How we do this depends on whether performing pre-processing up front will bottleneck our ability to feed the model. For example, we might want to pre-store all the JPEGs into some serialized format where normalization and augmentation has already taken place. We handle this as follows:

The `DataLoader` module is an abstract base class takes meta-data from the `AnnotationStore` and pulls binary objects from `DataSource` and provides the minimal set of functions that must be implemented for the data loading:
1. A method `get_batch` that returns the next batch of examples.
2. A method `length` that returns the number of total examples in the dataset.
3. A vector of operations which define pre-processing. 

There are two options for using the output of `DataLoader`: you can either feed the model 
during training by directly calling `get_batch`, or you can serialize the batch outputs so that all pre-processing 
does not need to re-occur. See the tutorial program `01_data_loading.cpp` for examples of how to do both.

## Models

Models are defined in an object-oriented manner, much like in PyTorch, and use computation primitives defined in the `src/layers` folder. The semantics closely follows PyTorch. `Model` and `Layer` are abstract base classes. You define strings of operations which are executed at run-time. All error-checking for e.g. dimension matching also occurs at runtime.

The `forward` function computes the forward output, and the backward information is maintained within the model.


Here we will define a simple dense bounding box predictor:

```
#include "layers/ConvLayer"

class SimpleBackbone : public Layer {
public:
    SimpleBackbone(){

    }

    forward(Layer *prevLayer){        
        Layer *out = prevLayer
        for(layer : m_set1){
            out = layer.forward(out)
        }
        
        out = m_adder.forward(out, prevLayer)

        for(layer : m_set2){
            out = layer.forward(out)
        }
        return out;
    }

private:
    std::vector<ConvLayer> m_set1;
    std::vector<ConvLayer> m_set2;
};

class AnchorRegression: public Layer {
public:
    AnchorRegression(){        
    }

    forward(Layer *prevLayer){
        for(layer : m_set1){
            prevLayer = layer.forward(prevLayer)
        }
    }
private:
    std::vector<ConvLayer> m_layers;
}


class DenseBoxModel : public Model {
public:
    DenseBoxModel(){

    }

    forward(Layer *prevLayer) {
        bbOut = m_backbone.forward(m_inputLayer);
        m_anchorLayer.forward(bbOut);

        return m_
    }

private:
    InputLayer m_inputLayer;
    SimpleBackbone m_backbone;
    AnchorRegression m_anchorLayer;
};

```


## Training

Training provides a single optimizer: SGD with momentum, which is the most commonly used optimizer in large frameworks for training object detection models. The trainer has built-in learning rate optimization strategies and is configured in an iteration-based scheme. So it is up to the user to calculate the number of iterations from epochs if so desired.

## Serializing the model for checkpoints or inference


## Outputing visualizations


## Predictions format


## Batch Inference


## Inference Service

## 