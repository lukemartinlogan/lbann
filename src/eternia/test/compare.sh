#!/bin/sh
# Compare the eternia paged forward GEMM against El::Gemm inside a running
# LBANN model, over a full training run.
#
#   LBANN=<lbann binary> CLIO_SERVER_CONF=<clio.yaml> ./compare.sh
#
# Why a training run and not a single forward pass: the paged path reads the
# weights through a device page cache, and the weights change every optimizer
# step. A single forward pass is exact even when the cache is never dropped --
# that bug survived a boundary test, a single-shot in-model test, and showed
# up only as a slightly different learning curve.
#
# LBANN_ETERNIA_CHECK=1 additionally recomputes each product with El::Gemm and
# prints the largest elementwise difference, which is the measurement that
# localises a fault to the weights rather than the arithmetic.
#
# Note: the Clio runtime currently segfaults at teardown after results are
# produced, which loses unflushed stderr; run under gdb to see the check
# output reliably.
set -e
# A correctly sized Clio config ships next to this script. Defaulting to it
# matters more than it looks: the hbm tier's capacity_limit is preallocated on
# the GPU, so borrowing a config with a large tier makes the paged path appear
# to cost far more memory than it saves.
: "${CLIO_SERVER_CONF:=$(cd "$(dirname "$0")" && pwd)/clio.yaml}"
export CLIO_SERVER_CONF
: "${LBANN:?set LBANN to the lbann binary}"
P="$(dirname "$0")/mlp.prototext"

echo "== El::Gemm =="
"$LBANN" --prototext="$P" 2>/dev/null |
  grep -oE "objective function : [0-9.]+" | awk '{printf "  %s\n",$NF}'

echo "== eternia (paged) =="
LBANN_ETERNIA_FC=1 gdb -batch -ex run --args "$LBANN" --prototext="$P" 2>/dev/null |
  grep -oE "objective function : [0-9.]+" | awk '{printf "  %s\n",$NF}'
