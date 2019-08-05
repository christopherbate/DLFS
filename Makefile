HOST_COMPILER ?= g++
CUDA_PATH := /usr/local/cuda
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
CXX := $(NVCC)

CXX_FLAGS := 

.PHONY = all clean dirs

# CUDA_OPTS=--cudart static --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61
CUDA_OPTS=

LIBS=-lnvjpeg -lcudnn -lcublas

OBJS = src/main.o src/Image.o src/GPU.o src/Logging.o src/Network.o src/layers/InputLayer.o src/layers/ConvLayer.o \
	src/layers/Layer.o src/Tensor.o src/data_loading/LocalSource.o src/data_loading/DataLoader.o src/data_loading/AnnotationSource.o

OBJS_UTILS = src/utils/ConvertCoco.o src/data_loading/AnnotationSource.o

TEST_OBJS = src/tests/TestAnnSrc.o src/data_loading/AnnotationSource.o

all: DLFS utils tests

DLFS: $(OBJS) 
	$(NVCC) -o bin/dlfs $(LIBS) $(CUDA_OPTS) $(OBJS)

utils: $(OBJS_UTILS)
	$(HOST_COMPILER) -o bin/convert_coco $(OBJS_UTILS)

tests: $(TEST_OBJS)
	$(HOST_COMPILER) -o bin/test $(TEST_OBJS)

$(OBJS): | dirs

src/utils/ConvertCoco.o: src/utils/ConvertCoco.cpp

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

src/data_loading/LocalSource.o: src/data_loading/LocalSource.hpp

src/data_loading/DataLoader.o: src/data_loading/DataLoader.hpp

src/data_loading/AnnotationSource.o: src/data_loading/AnnotationSource.hpp src/data_loading/annotations_generated.h

src/tests/TestAnnSrc.cpp: src/data_loading/AnnotationSource.hpp

dirs:
	mkdir -p ./bin/ ./.build/

clean:
	rm -rf ./bin/ ./.build/ $(OBJS)

