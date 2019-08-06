HOST_COMPILER ?= g++
CUDA_PATH := /usr/local/cuda
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
CXX := $(NVCC) -g
FLAT_C := flatc

CXX_FLAGS := -Wall

.PHONY = all clean dirs flatbuffers

# CUDA_OPTS=--cudart static --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61
CUDA_OPTS=

LIBS=-lnvjpeg -lcudnn -lcublas

NON_MAIN_OBJS = src/data_loading/ImageLoader.o src/GPU.o src/Logging.o src/Network.o src/layers/InputLayer.o src/layers/ConvLayer.o \
	src/layers/Layer.o src/data_loading/LocalSource.o src/data_loading/DataLoader.o src/data_loading/ExampleSource.o src/tensor/Tensor.o \
	src/tensor/TensorList.o

OBJS = src/main.o $(NON_MAIN_OBJS)

OBJS_UTILS = src/utils/ConvertCoco.o $(NON_MAIN_OBJS)

TEST_OBJS = src/tests/UnitTest.o $(NON_MAIN_OBJS)	

all: DLFS utils tests

DLFS: $(OBJS) 
	$(NVCC) -o bin/dlfs $(LIBS) $(CUDA_OPTS) $(OBJS)

utils: $(OBJS_UTILS)
	$(NVCC) -o bin/convert_coco $(LIBS) $(CUDA_OPTS) $(OBJS_UTILS) 

tests: $(TEST_OBJS)
	$(NVCC) -o bin/test $(LIBS) $(CUDA_OPTS) $(TEST_OBJS) 

$(OBJS): | dirs
$(OBJS): | flatbuffers

src/utils/ConvertCoco.o: src/utils/ConvertCoco.cpp

src/data_loading/ImageLoader.o: src/data_loading/ImageLoader.hpp src/Logging.hpp

src/Network.o: src/Network.hpp src/Logging.hpp

src/Main.o: src/Logging.hpp src/GPU.hpp src/Network.hpp src/layers/ConvLayer.hpp src/layers/InputLayer.hpp

src/GPU.o: src/GPU.hpp src/Logging.hpp

src/Logging.o: src/Logging.hpp

src/tensor/Tensor.o: src/tensor/Tensor.hpp

src/layers/Layer.o: src/layers/Layer.hpp

src/layers/ConvLayer.o: src/layers/ConvLayer.hpp src/layers/Layer.hpp

src/layers/InputLayer.o: src/layers/InputLayer.hpp src/layers/Layer.hpp

src/data_loading/LocalSource.o: src/data_loading/LocalSource.hpp

src/data_loading/DataLoader.o: src/data_loading/DataLoader.hpp

src/data_loading/ExampleSource.o: src/data_loading/ExampleSource.hpp src/data_loading/dataset_generated.h

src/tests/UnitTest.cpp: src/data_loading/ExampleSource.hpp

dirs:
	mkdir -p ./bin/ ./.build/

flatbuffers:
	$(FLAT_C) --cpp --gen-mutable -o ./src/data_loading/ ./src/data_loading/dataset.fbs 

clean:
	rm -rf ./bin/ ./.build/ $(OBJS) $(OBJS_UTILS) $(TEST_OBJS)

