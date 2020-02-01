load("@rules_proto_grpc//cpp:defs.bzl", "cpp_grpc_library")

package(default_visibility = ["//visibility:public"])


proto_library(
    name = "dlfs_proto",
    srcs = [
        "dlfs.proto",
        "images.proto",
        "runners.proto",
    ],
    deps = ["@com_google_protobuf//:timestamp_proto"],
)

cpp_grpc_library(
    name = "cpp_dlfs_proto",
    deps = [":dlfs_proto"]
)
