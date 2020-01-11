package(default_visibility = ["//visibility:public"])

cc_test(
    name = "tensor",
    srcs = [
        "QuickTestCPP.h",
        "TestTensor.cpp",
        "UnitTest.hpp",
    ],
    deps = [
        "//lib:kernels",
        "//lib:tensor",
        "@cuda",
    ],
)

cc_test(
    name = "convolve",
    srcs = [
        "QuickTestCPP.h",
        "TestConv.cpp",
    ],
    deps = [
        "//lib:operations",
        "//lib:tensor",
    ],
)

cc_test(
    name = "activation",
    srcs = [
        "QuickTestCPP.h",
        "TestActivation.cpp",
    ],
    deps = [
        "//lib:tensor",
    ],
)

cc_test(
    name = "gpu",
    srcs = [
        "QuickTestCPP.h",
        "TestGPU.cpp",
    ],
    deps = [
        "//lib:gpu",
        "//lib:logging",
    ],
)

filegroup(
    name = "test_data",
    srcs = glob([
        "data/*.jpg",
        "data/*.db",
    ]),
)

cc_test(
    name = "image_loader",
    srcs = [
        "QuickTestCPP.h",
        "TestImage.cpp",
    ],
    data = [
        ":test_data",
    ],
    linkstatic = True,
    deps = [
        "//lib:image_loader",
        "//lib:local_source",
        "//lib:utils",
        "//lib/external:jpec",
    ],
)

cc_test(
    name = "data_loader",
    srcs = [
        "QuickTestCPP.h",
        "TestDataLoader.cpp",
    ],
    data = [
        ":test_data",
    ],
    linkstatic = True,
    deps = [
        "//lib:data_loader",
    ],
)

cc_test(
    name = "pointwise_ops",
    srcs = [
        "QuickTestCPP.h",
        "TestTensorOp.cpp",
    ],
    deps = [
        "//lib:tensor",
    ],
)

cc_test(
    name = "softmaxce_op",
    srcs = [
        "QuickTestCPP.h",
        "TestSoftmax.cpp",
    ],
    deps = [
        "//lib:tensor",
    ],
)

cc_test(
    name = "batchnorm",
    srcs = [
        "QuickTestCPP.h",
        "TestBatchNorm.cpp",
    ],
    deps = [
        "//lib:tensor",
    ],
)

cc_test(
    name = "autodiff",
    srcs = [
        "QuickTestCPP.h",
        "TestAutoDiff.cpp",
    ],
    deps = [
        "//lib:tensor",
    ],
)

cc_test(
    name = "mnist",
    srcs = [
        "QuickTestCPP.h",
        "TestMNIST.cpp",
    ],
    data = [
        ":test_data",
    ],
    deps = [
        "//lib:data_loader",
        "//lib:example_source",
        "//lib:image_loader",
        "//lib:local_source",
    ],
)

cc_test(
    name = "network",
    srcs = [
        "QuickTestCPP.h",
        "TestNetwork.cpp",
    ],
    deps = [
        "//lib:tensor",
    ],
)
