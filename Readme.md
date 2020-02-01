# Deep Learning from Scratch

[Overview Presentation](https://docs.google.com/presentation/d/13iccyzgeLPLY2YbbjRyn206PnmnWebZJnQ1kIaA85Uk/edit?usp=sharing)

A deep learning C++ library built on CUDNN. The aim of this library is to provide simple interfaces to build deep learning 
alogorithms primarily for computer vision in a small, flexible package.

Priorities:

1. Utilize latest capabilities offered by RTX cards (Tensor Cores)
    - TODO: GPU Profiling tests for different cards / platforms
2. Simple code base
3. Focus on mapping similar research capabilities as Detectron2 while matching/exceeding performance.

## Roadmap 

Done: 

1. Basic add/sub/mul/power operations
2. Autodiff (single threaded)
3. Data loading / batching using NvJPEG (single thread)
4. Fully working MNIST "hello world" unit tested for correct autograd calculations

In Progress:

2. Multithreaded batch loader
3. SQLite annotation data loader
3. ResNet example
4. RetinaNet example
5. Serialization

# Prerequisites

This library is GPU only. You must have a system with an NVIDIA GPU.

We develop on Ubuntu 19.04 systems with Intel CPUs and RTX 2080 series cards. Recommend 
you isolate your development environment from the various other software that are trying 
to manipulate CUDA libraries and runtimes (e.g.conda).

Requires :

* For build we use Bazel 2.0.0 [Install Link](https://docs.bazel.build/versions/master/install-ubuntu.html)
* Nvidia driver (we're on 418)
* CUDA >= 10.1 (we're developing on `cuda_10.1.243` installed with the `run` file on 19.04)
* CuDNN 7.6.3 
* NvJPEG
* flatbuffer compiler (see below)
* cmake
* COCO 2017 images and labels (see below)
* Lodepng (put in src/external/lodepng/lodepng.h)

## COCO 

We setup all datasets in `~/datasets` .

``` 
mkdir -p  ~/datasets/coco && cd ~/datasets/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip
unzip annotations_trainval2017.zip
```

# Build

Bazel is required (see above).

View build dependencies:

``` 
sudo apt update && sudo apt install graphviz xdot
./dep_graph.sh
```

To build the library:

```shell
bazel build //lib:all
```

To compile the runner client:

```
bazel build //client:main
```

# Test

For the data loading tests, you need the coco annotations in our serialized format (download link).
The other tests only require some sample images which are included. 

```
bazel test //tests:all
```

### Getting the testing data.

Tests depend on a couple files from COCO validation set. Do the following:

``` shell
COCO=[coco2017 validation directory]
cp COCO/000000397133.jpg tests/data 
000000037777.jpg tests/data
000000252219.jpg tests/data
000000087038.jpg tests/datra
```

Run:

``` shell
bazel test //tests:all
```

# Overview 

To train a deep learning model in a reproducable way, you need minimally the following things:

 1. Data and annotations
 2. A model
 3. Optimization/training management system that takes model and data and outputs trained weights, including the ability to optimizer accross multi-GPU and multi-host systems.
 4. Visualization/Monitoring: an external system for consuming sample outputs of the model in realtime so you can 
 monitor progress.
 5. A standard way for serializing the model and its weights as well as the predictions
 6. The ability to serialize stateful parts of the training system to restore from checkpoint if training is interrupted.
 7. The necessary functions for efficiently calculating various performance metrics (e.g.mAP) during/after training.
 8. A system for data augmentation, which can considerably improve model robustness and performance

For inference, you need the following things:

1. A way to specify models which should be served and loading those model into the inference engine
2. A method for feeding data to the models efficiently.
3. Excellent logging and monitoring for production

The goal is to provide all of the above in a compact library for computer vision models, focusing 
on fast turn-around between data collection, labeling, training, inference, back to labeling. It will provide serialized formats for streaming predictions and labels to/from an system during training for effective real-time training. To do this, we leverage CuDNN for nearly all primitives within the deep learning engine, resulting in a simple, highly performant, readable and maintainable code base.

Unlike general deep learning frameworks such as PyTorch and Tensorflow, this library is completely biased towards vision systems. There are only a small set of operations which are implemented (e.g.convolution), but they are finely tuned for speed and the latest features (e.g.mixed-precision training) and research. There will b e basic primitives for bounding box operations for dense anchor-based systems as well as post-processing functions such as Non-Max Suppression.

Futhermore, we assume all input data is in the form of images which will be turned into 3-channel tensors. There is no notion of general n-d tensor operations like in Tensorflow, PyTorch and Numpy. There is no support for general slicing of arrays either. Nevertheless, a great deal of interesting models and research questions can be tackled within this framework, all within a simple system that can take the model into "production" environments.

