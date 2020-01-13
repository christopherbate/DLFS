# This imports the function for bazel to download archived packages from an HTTP resource
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_proto_grpc",
    sha256 = "497225bb586e8f587e139c55b0f015e93bdddfd81902985ce24623528dbe31ab",
    strip_prefix = "rules_proto_grpc-1.0.1",
    urls = ["https://github.com/rules-proto-grpc/rules_proto_grpc/archive/1.0.1.tar.gz"],
)

load("@rules_proto_grpc//:repositories.bzl", "rules_proto_grpc_repos", "rules_proto_grpc_toolchains")

rules_proto_grpc_toolchains()

rules_proto_grpc_repos()

load("@rules_proto_grpc//cpp:repositories.bzl", rules_proto_grpc_cpp_repos = "cpp_repos")

rules_proto_grpc_cpp_repos()

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

# Add cuda, cudnn
new_local_repository(
    name = "cuda",
    path = "/usr/",
    build_file = "bazel/cuda.BUILD",    
)