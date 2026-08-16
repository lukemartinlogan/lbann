# src/eternia - LBANN fully-connected forward over a paged weight matrix

The weight matrix W of a fully-connected layer lives in the Clio Context
Transfer Engine and is paged into GPU memory from inside the kernel, so the
layer can be wider than the GPU has memory:

    C (m x n)  =  W (m x k)  *  X (k x n)

LBANN's fully-connected forward is an `El::Gemm` of W against the layer
input. For a wide layer W is the thing that does not fit; the activations are
k x n with n the mini-batch, which is small beside m*k. So W is paged and the
activations stay resident.

## Status: standalone, not yet wired into LBANN

`eternia_gemm_bench.cc` is self-contained - paged kernel, CPU reference,
analytic check, config sweep - with its own CMake and clang toolchain.
Proving the kernel before wiring it in is what made the GROMACS integration
take one round where LAMMPS took three.

```
cmake -S src/eternia -B build-eternia \
  -DCMAKE_CUDA_COMPILER=clang++-22 -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_PREFIX_PATH=<clio-inst> \
  -Diowarp-core_DIR=<clio-inst>/lib/cmake/iowarp-core
cmake --build build-eternia -j
CLIO_SERVER_CONF=clio.yaml ./build-eternia/lbann_eternia_gemm_bench \
  --m 65536 --k 40960 --n 16 --page-kb 256 --blocks 64 --slots 8 --ones
```

## W is stored ROW-MAJOR, and that is not incidental

Hydrogen is column-major. This code stores W row-major on purpose: a page of
a row-major W is a run of whole rows, so a block that owns those pages can
compute the corresponding rows of C completely and the write set stays
disjoint. In column-major a page is a run of whole COLUMNS - a partial
contribution to *every* row of C at once - so every block would accumulate
into all of C.

Wiring this into LBANN therefore means transposing W once when it is loaded,
not paging Hydrogen's buffer where it lies. That is a real integration cost
and is the main open question for the next step.

## How the kernel is organised

Each block owns a **page-aligned** slice of W. Page caches are per block and
writeback granularity is a page, so blocks must never share one. (W is
read-only here, so the sharing hazard is milder than on a written grid, but
the rule is kept because it is also what makes the C write set disjoint.)

Within a slice, one thread owns one (row, column) pair of C and walks the
part of that row present in the current page - one `atomicAdd` per (row, col)
per page rather than one per multiply. A row that straddles a page boundary
is finished by the block owning the other page, which the atomic makes safe.

## Verification

Two independent checks. A full CPU reference at sizes where it is
affordable, and an analytic one for weight matrices too large to reference:
W follows a pattern whose row sums have a closed form, so with X = 1 the
expected result is O(1) per row and a matrix larger than host memory can
still be checked exactly.

| m    | k    | n  | page  | blocks | slots | max abs diff | faults / evicts |
|------|------|----|-------|--------|-------|--------------|-----------------|
| 2048 | 2048 | 16 | 256KB | 16     | 8     | 0.0          | 64 / 0          |
| 2048 | 2048 | 16 | 4KB   | 16     | 3     | 0.0          | 4096 / 4048     |
| 2048 | 2048 | 16 | 4KB   | 64     | 2     | 0.0          | 4096 / 3968     |
| 2048 | 2048 | 16 | 1MB   | 4      | 2     | 0.0          | 16 / 8          |
| 1000 | 3333 | 7  | 16KB  | 13     | 3     | 6.5e-06      | 814 / 775       |
| 2048 | 2048 | 16 | 64KB  | 8      | 3     | 0.0 (analytic) | 256 / 232     |

The 1000 x 3333 / n=7 / 13-block row is deliberately non-power-of-two: it
forces rows to straddle page boundaries and blocks to own ragged slices,
which is where the boundary handling would fail.

## Larger than VRAM

On an 8 GiB (7.99 GiB usable) RTX 4070 Laptop:

| W             | elements      | size          | faults / evicts | result |
|---------------|---------------|---------------|-----------------|--------|
| 32768 x 32768 | 1,073,741,824 | 4.00 GiB      | 16,384 / 15,872 | PASS   |
| 65536 x 40960 | 2,684,354,560 | **10.00 GiB** | 40,960 / 40,448 | PASS   |

The last row holds a weight matrix larger than the GPU's memory and pages it
throughout, against a GPU-side cache of `blocks * slots * page` = 64 * 8 *
256 KB = 128 MB, about 1.25% of the matrix, and reproduces the expected
result exactly.

## Next

Wire into `src/layers/learning/fully_connected.cpp` behind a runtime switch,
which needs the row-major transpose above and a decision about where the
transposed copy lives, then validate against an LBANN model rather than a
synthetic reference.
