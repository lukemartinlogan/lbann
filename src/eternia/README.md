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

## The transpose is free in the case that matters

An earlier version of this file said wiring into LBANN "needs W transposed
once at load". That is only half right, and the half that is wrong is the
interesting one.

Hydrogen stores a matrix COLUMN-major: for W of height h and width w,
element (i, j) sits at j*h + i. This kernel reads ROW-major: R[a][b] at
a*k + b. Take the SAME buffer and read it row-major with m = w and k = h:

    R[a][b] = buf[a*h + b] = W(b, a)      i.e.  R = W-transpose

So when LBANN's fully-connected layer computes C = W^T * X -- its
`m_transpose == true` path -- the raw column-major buffer IS the row-major
matrix this kernel wants, with no copy and no transpose kernel. The layouts
are duals and the reinterpretation is free.

The other path, C = W * X, does need a real transpose, and that is the case
to fall back to El::Gemm on rather than pay for.

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

## Validated through the LBANN boundary

`lbann_eternia_gemm_verify` exercises eternia_gemm.h with the SAME layouts
the fully-connected layer passes -- W column-major on the host, X and C
column-major on the device, computing C = W^T * X -- against an independent
host reference. That matters separately from the standalone bench: the bench
proved the kernel, but the integration adds a LAYOUT CONTRACT (Hydrogen's
column-major buffer reinterpreted as the kernel's row-major weight matrix),
and a layout contract is exactly the sort of thing that is plausible on paper
and wrong in practice.

| W          | X      | page  | blocks | slots | max abs diff | faults / evicts |
|------------|--------|-------|--------|-------|--------------|-----------------|
| 512 x 256  | 512x8  | 64KB  | 16     | 8     | 9.4e-07      | 8 / 0           |
| 512 x 256  | 512x8  | 4KB   | 16     | 3     | 9.4e-07      | 128 / 80        |
| 512 x 256  | 512x8  | 4KB   | 64     | 2     | 9.4e-07      | 128 / 0         |
| 512 x 256  | 512x8  | 1MB   | 4      | 2     | 9.4e-07      | 1 / 0           |
| 333 x 177  | 333x5  | 16KB  | 13     | 3     | 8.3e-07      | 15 / 0          |
| 2048 x 1024| 2048x32| 256KB | 32     | 4     | 1.9e-06      | 32 / 0          |

The 333 x 177 / n=5 / 13-block row is deliberately non-power-of-two, so rows
straddle page boundaries and blocks own ragged slices.

### IMPORTANT: the integration does not lift LBANN's VRAM ceiling

The numbers below are for the KERNEL, driven directly. They are not what the
integration delivers, and the difference matters.

LBANN allocates the weight matrix itself, as a Hydrogen matrix on the GPU.
This hook reads that matrix and copies it into the CTE; it does not replace
where LBANN keeps its weights. So a layer whose weights exceed VRAM fails in
LBANN's own allocator before the paged kernel is ever reached:

    32768 x 65536 = 8.00 GiB
    Failed to allocate GPU memory with message: "out of memory"
      (8589934592 bytes requested, 8032092160 bytes available)

with LBANN_ETERNIA_FC=1 set. The paged GEMM is therefore validated as a
drop-in for the forward GEMM -- exact, including across training -- and as a
kernel that scales past VRAM when driven directly. It is NOT a way to train a
layer wider than GPU memory, and nothing here should be read as claiming that.

#### How far the host-weights route gets, and where it stops

`LBANN_ETERNIA_FC_HOST_WEIGHTS=1` is a first step at removing that
allocation. `El::DistData` carries a device independently of the layer's, so
the linearity can be placed in host memory while the activations stay on the
GPU; the paged path already reads the weights through the host on their way
to the CTE, so this removes a GPU allocation rather than adding a transfer.

It is NOT sufficient, and fails loudly rather than silently:

    Must call gemm with matrices on same device.

The forward GEMM is intercepted, but backpropagation is not, and it calls
El::Gemm with the host weights and the GPU error signals. Training a layer
wider than VRAM therefore needs three more pieces, all of which this branch
has scoped but not built:

1. ~~`dL/dX = W * dL/dY`~~ -- **done**. `BackwardInput` walks the same pages
   of W as the forward and differs only in which index it accumulates into:

       forward   C[c*ldc + i]  = sum_j W[i*k + j] * X [c*ldx + j]
       backward  dX[c*ldx + j] = sum_i W[i*k + j] * dC[c*ldc + i]

   A page of W holds whole ROWS, so a block contributes to many rows of dX
   rather than owning a few, and the accumulation is a cross-block atomicAdd
   -- safe because dX is resident and only k*n. Validated against an
   independent host reference across the same paging sweep as the forward,
   at 4e-07 relative or better, including with 80 of 128 pages evicted.

2. ~~`dL/dW = X * dL/dY^T`~~ -- **done**. `WeightGradient` holds the gradient
   in its own paged vector, because it is the same size as W and so cannot be
   resident either:

       dW[i*k + j] = sum_c dC[c*ldc + i] * X[c*ldx + j]

   It is the only one of the three that WRITES a paged array, so it is the
   only one where page-aligned block ownership matters -- and it satisfies
   that by construction, since every element depends only on the resident dC
   and X, so a block writes nothing outside its own pages. Its backing store
   is created up front for the same reason the LAMMPS force vector and the
   PME grid needed it: a hold faults a page in before the kernel writes it,
   and a page with no blob comes back as a failed read.

   `ReadWeightGradient` copies it back into Hydrogen's column-major layout.

3. The optimizer update over paged W and paged dW, page by page. Still to do,
   and now the only piece missing.

All three kernels are validated together in `lbann_eternia_gemm_verify`
against independent host references, across page sizes 4KB-256KB, 13-64
blocks and 2-8 slots, including a non-power-of-two shape and cases evicting
most of what they fault:

| kernel          | relative error |
|-----------------|----------------|
| forward         | 3.5e-08        |
| backward-input  | 4.3e-07        |
| weight gradient | 5.2e-08        |

Each is checked separately rather than assumed to follow from the others:
they are three different kernels with three different access patterns, and
this project has repeatedly found that the second and third are where an
assumption breaks.

Until the optimizer step exists the host-weights option remains experimental:
the pieces are in place and validated, but nothing yet drives them as a
training step.

### Larger than VRAM (the kernel, driven directly)

On an 8 GiB (7.99 GiB usable) RTX 4070 Laptop:

| W             | elements      | size          | faults / evicts | max abs diff | result |
|---------------|---------------|---------------|-----------------|--------------|--------|
| 65536 x 40960 | 2,684,354,560 | **10.00 GiB** | 40,960 / 40,448 | 3.7e-04      | PASS   |

Against a peak of 1.09e+04 that is 3.4e-08 relative -- single-precision
rounding on a sum of 65536 products. The GPU-side cache is
`blocks * slots * page` = 64 * 8 * 256 KB = 128 MB, about 1.25% of the
matrix.

## Validated inside LBANN, on a real model

`lbann_eternia_gemm_verify` proves the boundary; it does not prove the
integration in a running model. That distinction mattered on the GROMACS side
of this project, where a boundary test passed while three assumptions about
the host's data were wrong, so the same check is made here.

A small MLP (64 -> 32 -> 4, `transpose: true`, no bias) on the synthetic data
reader, run with and without `LBANN_ETERNIA_FC=1`. The baseline is
deterministic: three consecutive runs give 1.97502 exactly, so any difference
is real rather than run-to-run noise.

**The forward pass is exact.** With the weights frozen (`learn_rate: 0.0`)
both paths give an objective of 1.98758. With one update per epoch
(mini-batch = all 256 samples) epoch 0 is 1.93652 for both -- that epoch sees
only the initial weights, so it is a direct comparison of the forward GEMM
inside a running model.

**Training matches too, once the device cache is dropped each call.** The
first attempt did not: from epoch 1 the trajectories separated, and measuring
the forward output against El::Gemm on the same inputs (`LBANN_ETERNIA_CHECK`)
showed why -- relative error 1.4e-07 in epoch 0 but 1.2e-03 by epoch 1.

The weights change every optimizer step and the host rewrites the backing
store, but NOTHING invalidates the pages already resident on the device, so
the kernel kept serving the weights from the first call. The objective still
fell, so training looked like it was working. `GemmCoro` now drops the
block's cache before reading, exactly as the LAMMPS integration has to for
coordinates.

With that, both paths agree digit for digit:

| schedule                  | El::Gemm                        | eternia                         |
|---------------------------|---------------------------------|---------------------------------|
| one update per epoch      | 1.93652 1.87546 1.87988         | 1.93652 1.87546 1.87988         |
| 4 minibatches, 4 epochs   | 1.97502 1.82216 1.7951 1.91683  | 1.97502 1.82216 1.7951 1.91683  |

and the forward difference stays at 1e-07 -- single-precision rounding -- at
every epoch rather than growing.

`LBANN_ETERNIA_CHECK=1` computes the same product with El::Gemm and reports
the largest elementwise difference. It is worth keeping: inferring
correctness from an objective several operations downstream is how a stale
cache reads as a slightly different learning curve.

## Next, and what blocks it

Wire into `src/layers/learning/fully_connected.cpp` behind a runtime switch.
Two things stand in the way, and the second is the larger:

**The row-major transpose.** Hydrogen hands over a column-major buffer and
the decomposition needs row-major (see above), so W has to be transposed once
at load and somewhere has to own that copy. That is a design decision, not an
obstacle.

**LBANN now builds in this environment**, which it did not when this file was
first written. The whole dependency stack is built from source through
`scripts/superbuild` (OpenBLAS, Aluminum, Hydrogen with CUDA, DiHydrogen,
protobuf, cereal, spdlog, zstr, Catch2, Clara, Conduit + HDF5) against CUDA
13.3 and cuDNN 9, and `bin/lbann` links and runs with LBANN_HAS_GPU,
LBANN_HAS_CUDA and LBANN_HAS_CUDNN all set.

Getting there needed nine toolchain fixes to LBANN itself, all committed on
this branch: CUDA 13 removed CUDA::nvToolsExt (referenced twice), Thrust
dropped system/cuda/detail/par.h and thrust::binary_function, cuFFT dropped
three error enumerators, data_utils links Conduit unconditionally, the
superbuild builds non-PIC statics, and conda's OpenSSL collides with the
system one at link time.

So the integration can now be compiled and run, which is the only kind worth
having here: every defect in this project was found by running something
rather than by reading it.
