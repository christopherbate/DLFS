#!/bin/bash
xdot <(bazel query --notool_deps --noimplicit_deps 'deps(//lib:operations)' \
  --output graph)