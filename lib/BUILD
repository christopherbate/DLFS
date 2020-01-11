package(default_visibility = ["//visibility:public"])

filegroup(
    name = "kernel_deps",
    srcs = [
        "GPU.hpp",
        "Logging.hpp",
        "operations/BaseOperation.hpp",
        "operations/types.hpp",
        "tensor/Tensor.hpp",
        "tensor/TensorBase.hpp",
    ],
)

filegroup(
    name = "cu_files",
    srcs = [
        "operations/PointwiseKernels.cu",
        "operations/SigmoidCEKernel.cu",
        "operations/SoftmaxCEKernel.cu"
    ]
)

genrule(
    name = "cuda_kernels",
    srcs = [
        ":kernel_deps",
        ":cu_files",
    ],
    outs = ["kernels_75.a"],
    cmd = "/usr/local/cuda/bin/nvcc --compiler-options -fPIC -std c++14 -I./lib/ --gpu-architecture=sm_75 --lib -o $@ $(locations :cu_files)",
)

cc_library(
    name = "kernels",
    srcs = [":cuda_kernels"],    
)

cc_library(
    name = "logging",
    srcs = ["Logging.cpp"],
    hdrs = ["Logging.hpp"],
    deps = ["@cuda"],
)

cc_library(
    name = "utils",
    hdrs = ["utils/Timer.hpp"],    
)

cc_library(
    name = "gpu",
    srcs = ["GPU.cpp"],
    hdrs = ["GPU.hpp"],
    deps = [
        ":logging",
        "@cuda",
    ],
)

cc_library(
    name = "data_loader",
    srcs = ["data_loading/DataLoader.cpp"],
    hdrs = [
        "data_loading/DataLoader.hpp",
        "data_loading/dataset_generated.h",
    ],
    deps = [":tensor", ":example_source", ":local_source"],
)

cc_library(
    name = "example_source",
    srcs = ["data_loading/ExampleSource.cpp"],
    hdrs = [
        "data_loading/ExampleSource.hpp",
        "data_loading/dataset_generated.h",
    ],
    deps = [
        ":utils",
        ":logging"
    ]
)

cc_library(
    name ="data_source",
    hdrs = ["data_loading/DataSource.hpp"],
)

cc_library(
    name = "local_source",
    srcs = ["data_loading/LocalSource.cpp"],
    hdrs = ["data_loading/LocalSource.hpp"],
    deps = [":data_source", ":image_loader"]
)

cc_library(
    name = "image_loader",
    srcs = ["data_loading/ImageLoader.cpp"],
    hdrs = ["data_loading/ImageLoader.hpp"],
    deps = [":logging", ":tensor"]
)

cc_library(
    name = "tensor_headers",
    hdrs = [
        "tensor/Tensor.hpp",
        "tensor/TensorBase.hpp",
        "tensor/TensorList.hpp",
    ],
    deps = [
        ":gpu",
        ":logging",
    ],
)

cc_library(
    name = "operations",
    srcs = glob(["operations/*.cpp"]),
    hdrs = glob(["operations/*.hpp"]),
    deps = [
        "tensor_headers",
        ":gpu",
        ":kernels",
    ],
)

cc_library(
    name = "tensor",
    srcs = glob(["tensor/*.cpp"]),
    hdrs = glob(["tensor/*.hpp"]),
    deps = [
        ":gpu",
        ":logging",
        ":operations",
        ":kernels",
        ":tensor_headers",
        "@cuda",
    ],
)
