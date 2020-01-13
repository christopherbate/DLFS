def _cc_cuda_library_impl(ctx):
    native.cc_library(
        deps = ctx.deps + ["@cuda"],
        copts = ctx.copts + ["-x cuda"],
        linkstatic = True,
        **kwargs
    )    
    

cc_cuda_library = rule (
    implementation = _cc_cuda_library_impl,    
    attrs = {
        "srcs": attr.label_list(allow_files=[".cu"]),    
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"]
)