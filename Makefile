HOST_COMPILER ?= g++
CUDA_PATH := /usr/local/cuda
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
CXX := $(NVCC)

CXX_FLAGS := 

.PHONY = all clean dirs

CUDA_OPTS=--cudart static --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61

LIBS=-lnvjpeg -lcudnn -lcublasLt

OBJS = src/main.o src/Image.o src/GPU.o src/Logging.o src/Network.o src/layers/InputLayer.o src/layers/ConvLayer.o src/layers/Layer.o src/Tensor.o

all: DLFS

DLFS: $(OBJS) 
	$(NVCC) -o bin/dlfs $(LIBS) $(CUDA_OPTS) $(OBJS)

$(OBJS): | dirs

src/Image.o: src/Image.hpp src/Logging.hpp

src/Network.o: src/Network.hpp src/Logging.hpp

src/Main.o: src/Logging.hpp src/GPU.hpp src/Network.hpp src/layers/ConvLayer.hpp src/layers/InputLayer.hpp

src/GPU.o: src/GPU.hpp src/Logging.hpp

src/Logging.o: src/Logging.hpp

src/Tensor.o: src/Tensor.hpp

src/layers/Layer.o: src/layers/Layer.hpp

src/layers/ConvLayer.o: src/layers/ConvLayer.hpp src/layers/Layer.hpp

src/layers/InputLayer.o: src/layers/InputLayer.hpp src/layers/Layer.hpp

src/Image.o: src/Image.hpp src/Logging.hpp

dirs:
	mkdir -p ./bin/ ./.build/

clean:
	rm -rf ./bin/ ./.build/ $(OBJS)

