# Adapted from
# https://github.com/gregestren/snippets/blob/master/custom_cc_toolchain_with_platforms/platforms/BUILD
# Defines a custom platform with a new CPU type.
#
# This is only necessary when you want to build for custom platforms (by
# building with the --platforms flag). By default Bazel targets the
# machine Bazel runs on, which is modeled with the auto-configured
# platform "@local_config_platform//:host".
platform(
    name = "jetson_nano",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ]
)
