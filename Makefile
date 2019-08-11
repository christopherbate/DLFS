HOST_COMPILER ?= g++
CUDA_PATH := /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
CXX := $(NVCC) 
FLAT_C := flatc
CXX_FLAGS := --compiler-options -Wall --compiler-options -Werror --compiler-options -MMD \
			 --compiler-options -Wextra -I ./src/ --gpu-architecture=compute_75 --compiler-options -g
CUDA_OPTS=
LIBS := -lnvjpeg -lcudnn -lcublas
OBJDIR := .build

.PHONY = all clean dirs 

FLATBUFFERS := src/data_loading/dataset_generated.h

DIRS := bin $(OBJDIR) $(OBJDIR)/data_loading $(OBJDIR)/tensor \
		$(OBJDIR)/utils $(OBJDIR)/tests $(OBJDIR)/operations

# Object definitions
_NON_MAIN_OBJS = GPU.o data_loading/ImageLoader.o Logging.o \
			     data_loading/LocalSource.o data_loading/DataLoader.o \
				 data_loading/ExampleSource.o tensor/Tensor.o tensor/TensorList.o \
				 operations/Convolution.o tensor/AutoDiff.o 

NON_MAIN_OBJS = $(addprefix $(OBJDIR)/, $(_NON_MAIN_OBJS))

DLFS_OBJS = .build/main.o $(NON_MAIN_OBJS)
CONVERT_UTIL_OBJS = .build/utils/ConvertCoco.o $(NON_MAIN_OBJS)

_TEST_OBJS = UnitTest.o TestTensor.o TestAutoDiff.o TestGPU.o TestDataLoader.o
UNIT_TEST_OBJS = $(addprefix $(OBJDIR)/tests/, $(_TEST_OBJS)) $(NON_MAIN_OBJS)

ALL_OBJS = $(DLFS_OBJS) $(CONVERT_UTIL_OBJS) $(UNIT_TEST_OBJS)

EXECUTABLES := DLFS utils tests

DEPS = $(ALL_OBJS:.o=.d)

$(OBJDIR)/%.o: src/%.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

all: $(EXECUTABLES)
	
DLFS: $(DLFS_OBJS)
	$(NVCC) -o bin/dlfs $(LIBS) $(CUDA_OPTS) $(DLFS_OBJS)

utils: $(CONVERT_UTIL_OBJS)
	$(NVCC) -o bin/convert_coco $(LIBS) $(CUDA_OPTS) $(CONVERT_UTIL_OBJS)

tests: $(UNIT_TEST_OBJS)
	$(NVCC) -o bin/test $(LIBS) $(CUDA_OPTS) $(UNIT_TEST_OBJS)

$(CONVERT_UTIL_OBJS) $(UNIT_TEST_OBJS) $(DLFS_OBJS):  $(FLATBUFFERS) | $(DIRS)

$(DIRS):
	mkdir -p $(DIRS)	

./src/data_loading/dataset_generated.h: ./src/data_loading/dataset.fbs
	$(FLAT_C) --cpp --gen-mutable -o ./src/data_loading/ ./src/data_loading/dataset.fbs 

clean:
	rm -rf $(DIRS) $(FLATBUFFERS) $(DEPS)

-include $(DEPS)
