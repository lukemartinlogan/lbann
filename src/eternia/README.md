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

3. The optimizer update over paged W and paged dW -- **done**. `SgdUpdate`
   computes `W -= lr * dW` entirely on the GPU with both arrays paged. Reading
   the gradient back and letting the optimizer touch a resident copy of W
   would put a full-size weight matrix in memory once per step and give the
   whole capacity argument away, so the update has to be paged too.

   It is the only kernel that holds a page from TWO paged vectors at once.
   That is safe because page caches are per block and per vector, so the two
   holds contend for nothing and neither waits behind a cross-block lock. Both
   vectors use the same page size and length, so element `e` sits at the same
   offset in page `e/pe` of each and one page of W is updated against exactly
   one page of dW.

All three kernels are validated together in `lbann_eternia_gemm_verify`
against independent host references, across page sizes 4KB-256KB, 13-64
blocks and 2-8 slots, including a non-power-of-two shape and cases evicting
most of what they fault:

| kernel          | relative error |
|-----------------|----------------|
| forward         | 3.5e-08        |
| backward-input  | 4.3e-07        |
| weight gradient | 5.2e-08        |
| SGD update      | 3.8e-08        |

Each is checked separately rather than assumed to follow from the others:
they are four different kernels with four different access patterns, and this
project has repeatedly found that the later ones are where an assumption
breaks.

The update is checked by running the FORWARD kernel again against a reference
built from the updated weights, not by reading W back. That is the stronger
check: it only passes if the update reached the backing store the forward pass
reads, so a write that stayed in a block's page cache and never flushed fails
it. The test also asserts the weights actually MOVED, because an update that
silently did nothing would track a reference built from the old weights just
as closely and pass a naive tolerance check.

The tightest configuration in the sweep is `--slots 2`: the update kernel
holds a page from each of two vectors, and two slots per vector is the least
that can serve it.

In the model
------------

`bp_compute` is now hooked the same way `fp_compute` is, gated identically --
the two must agree, because a step that pages the forward GEMM and then falls
back to `El::Gemm` for the backward one cannot run at all once the weights are
on the host.

`LBANN_ETERNIA_FC_HOST_WEIGHTS` consequently trains end to end, with the
linearity never resident on the GPU. Over three epochs of the test model the
objective is IDENTICAL to the stock path:

| run                       | epoch 1 | epoch 2 | epoch 3 |
|---------------------------|---------|---------|---------|
| `El::Gemm` (stock)        | 1.93652 | 1.87546 | 1.87988 |
| eternia paged fwd + bwd   | 1.93652 | 1.87546 | 1.87988 |
| eternia + host weights    | 1.93652 | 1.87546 | 1.87988 |

`LBANN_ETERNIA_CHECK=1` recomputes both backward products with `El::Gemm` and
reports the largest elementwise difference per call.

`mlp.prototext` is too narrow to check one of them. Its backward sums are 4
and 32 terms deep, few enough that the paged kernel matches `El::Gemm`
BITWISE -- which is not evidence of correctness, because a comparison that
cannot separate two correct implementations cannot separate a correct one from
a subtly wrong one either. `mlp_wide.prototext` (1024 -> 512 -> 4) exists for
that reason: at 512 terms deep the reassociated accumulation must differ, and
the check starts carrying information.

| model | `bp-input` | `bp-weights` |
|-------|------------|--------------|
| `mlp.prototext` (4, 32 deep)  | bitwise, uninformative | 2.5e-07 |
| `mlp_wide.prototext` (512 deep) | 7.9e-07              | 4.0e-07 |

Both models give the identical objective on all three paths, and the wide one
also puts a 2 MB weight matrix across many pages rather than one or two:

| run                     | epoch 1 | epoch 2 | epoch 3 |
|-------------------------|---------|---------|---------|
| `El::Gemm` (stock)      | 2.11687 | 2.06452 | 1.87059 |
| eternia paged fwd + bwd | 2.11687 | 2.06452 | 1.87059 |
| eternia + host weights  | 2.11687 | 2.06452 | 1.87059 |

Two things this does NOT do
---------------------------

**The default path still round-trips the gradient.** The optimizer allocates a
resident, full-size gradient buffer whatever the layer does, so `bp_compute`
folds the paged gradient into it through host memory. `LBANN_ETERNIA_FC_OWN_WEIGHTS`
is the answer to that -- see below -- but it is a mode, not the default.

Letting the paged store own the weights
---------------------------------------

`LBANN_ETERNIA_FC_OWN_WEIGHTS` makes the paged vector the authoritative copy of
W. `bp_compute` then calls `SgdUpdate` directly instead of feeding LBANN's
optimizer, so **dW never leaves the paged vector and W is never reassembled
anywhere**. Two consequences fall out of that and both had to be handled:

- The weights are uploaded ONCE. Re-uploading each call, as the default path
  does, would overwrite every update with the pre-training weights and the
  model would never learn.
- LBANN steps every weight each iteration regardless of what the layer did, so
  the gradient buffer is ZEROED rather than ignored -- a stale gradient there
  would be applied on top of the update just made.

Zeroing the gradient neutralizes LBANN's step only for a STATELESS optimizer.
Anything carrying accumulated state -- momentum, Adam's moments, Adagrad's
history -- keeps applying that state to weights the paged store has already
updated, and the run would train to a quietly wrong answer with every counter
clean. The mode therefore REFUSES rather than mis-trains:

    LBANN_ETERNIA_FC_OWN_WEIGHTS supports only SGD with zero momentum, but
    this weight has momentum 0.9. Momentum would be applied on top of the
    update the paged store just made.

    LBANN_ETERNIA_FC_OWN_WEIGHTS supports only plain SGD, but this weight
    uses Adam. A stateful optimizer would keep stepping weights the paged
    store has already updated.

The refusal is narrow, and that was checked rather than assumed: the same
momentum model runs fine on the ordinary paged path and matches stock exactly
(1.83390, distinct from the zero-momentum 1.87059 -- so momentum really is
active in the comparison).

`LBANN_ETERNIA_CHECK` is suppressed in this mode: Hydrogen's copy of W is stale
after the first step, so an `El::Gemm` reference built from it would measure
the staleness rather than the kernel.

That removes the usual way of checking the result, so the evidence is
different. LBANN's optimizer is neutralized by the zeroed buffer, meaning the
Hydrogen weights cannot change -- yet the model still learns, and learns to the
same objective as stock. It follows that the learning is coming from the paged
store and nowhere else. `LBANN_ETERNIA_STATS=1` prints the six
`own-weights step lr=0.01` lines (2 layers x 3 epochs) that confirm the branch
runs at all, which an identical objective on its own would not.

Combined with a cache one thirty-second of the weight matrix:

| run | epoch 1 | epoch 2 | epoch 3 |
|---|---|---|---|
| `El::Gemm` (stock) | 2.11687 | 2.06452 | 1.87059 |
| own-weights, default cache | 2.11687 | 2.06452 | 1.87059 |
| own-weights, 64 KiB cache vs 2 MiB W | 2.11687 | 2.06452 | 1.87059 |

The last row takes 3584 faults against 3568 evictions.

What is still NOT saved: Hydrogen's `DistMatrix` for the linearity continues to
exist, because LBANN's `weights` object owns it and the model serializes and
checkpoints through it. Removing that is a change to LBANN's weights ownership,
not to this layer.

Under real paging pressure
--------------------------

The default geometry gives a resident cache of `nblocks * slots * page_bytes`
= 128 MB, far larger than any test model's weights, so the defaults show the
paged path working without ever putting it under pressure. `LBANN_ETERNIA_PAGE_KB`,
`_BLOCKS` and `_SLOTS` shrink it, mirroring the knobs the GROMACS hook already
had.

With `PAGE_KB=4 BLOCKS=8 SLOTS=2` the cache is 64 KiB against fc1's 2 MiB of
weights -- **1/32 of the matrix resident** -- and the layer takes 2560 faults
against 2544 evictions, so essentially every page is fetched, used and
discarded:

| | value |
|---|---|
| weights | 2 MiB |
| resident cache | 64 KiB (1/32) |
| faults / evicts | 2560 / 2544 |
| `bp-input` | 6.1e-07 |
| `bp-weights` | 2.9e-07 |
| objective | 2.11687 / 2.06452 / 1.87059 -- identical to stock |

At a size where paging is the only option
-----------------------------------------

2 MiB of weights shows the mechanism but not that it holds at a size where
paging is the point. `mlp_big.prototext` (4096 -> 4096 -> 4) gives fc1 a 64 MiB
weight matrix, and it was run at two cache ratios with the paged store owning
the weights:

| resident cache | ratio | faults / evicts | wall clock | objective |
|----------------|-------|-----------------|-----------|-----------|
| stock, W in VRAM | --  | --              | 3.2 s     | 2.20537 / 2.24280 |
| 4 MiB          | 1/16  | 4096 / 4032     | --        | 2.20537 / 2.24280 |
| 1 MiB          | 1/64  | 8192 / 8160     | 16 s      | 2.20537 / 2.24280 |

Identical to every digit at 1/64 residency, with `get_errors=0` and the cache
turning over essentially completely.

The cost is real and worth stating plainly: about 5x wall clock against a stock
run that keeps the whole matrix in VRAM. That is the trade this whole thing
offers -- it is not a speedup, it is a way to run a layer that otherwise would
not fit, and comparing it against a configuration that fits will always lose.

That is the claim this integration actually supports: a layer whose weights do
not fit in the memory available to it trains to the same answer.

**It lifts the VRAM ceiling only above about a gigabyte of weights.** This
corrects what this file said for several revisions. Peak GPU memory, sampled
with nvidia-smi across the whole run, with `HOST_WEIGHTS` + `OWN_WEIGHTS` and a
cache of a few MiB:

| fc1 weights | stock (W in VRAM) | paged |
|-------------|-------------------|-------|
| 2 MiB       | 158 MiB           | 1238 MiB |
| 64 MiB      | 298 MiB           | 1250 MiB |
| 256 MiB     | 408 MiB           | 1266 MiB |
| **1 GiB**   | **4298 MiB**      | **1326 MiB** |

The paged line is nearly FLAT: 254 MiB more weights buys 28 MiB more VRAM. The
stock line grows with the weights, and faster than one-for-one because the
optimizer's gradient buffer is W-sized too.

They cross between 256 MiB and 1 GiB, and the reason is the Clio runtime's own
GPU footprint, which is fixed at roughly 1.2 GiB whatever the model does.
BELOW that crossover the paged path costs several times MORE VRAM than simply
keeping the weights resident -- so on a small layer this is strictly worse, and
the earlier claim that it "does not lift the VRAM ceiling" was drawn entirely
from models on the wrong side of it.

At 1 GiB the saving is real: **4298 MiB -> 1326 MiB, a 3.2x reduction**, with
the objective identical to stock (1.92597 / 5.80251) and `get_errors=0`. The
cost is time -- 118 s against 4.4 s, roughly 27x, at 16384 faults with a cache
1/128 the size of the matrix.

The single most valuable thing that could be done to this integration is
therefore not a kernel change at all: it is shrinking that 1.2 GiB fixed
runtime footprint, which is what puts the crossover a gigabyte up.

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
