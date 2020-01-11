# Cuda implementation includes
prefix = "local/cuda-10.2/targets/x86_64-linux/include"         
headers = glob([prefix+"/*.hpp"])

# srcs = glob(["local/cuda-10.2/lib64/**/*.so.*"])
cc_library(
    name = "x86_64-cuda",
    # srcs = srcs,
    hdrs = headers,
    strip_include_prefix = prefix,
)

cudnn_prefix = "lib/x86_64-linux-gnu"
cudnn_inc_prefix = "include"
cudnn_so = [cudnn_prefix + "/libcudnn.so.7.6.5"]
cudnn_headers = [cudnn_inc_prefix+"/cudnn.h"]

cc_library(
    name = "x86_64-cudnn",        
    hdrs = cudnn_headers,    
    srcs =  cudnn_so,
    strip_include_prefix = cudnn_inc_prefix,
)

cc_library(
    name="cuda",
    srcs = glob(["local/cuda-10.2/lib64/*.so.*"]),
    hdrs = glob(["local/cuda-10.2/include/**/*.h"]),
    deps = ["x86_64-cuda", "x86_64-cudnn"],
    strip_include_prefix = "local/cuda-10.2/include",
    visibility = ["//visibility:public"],
)