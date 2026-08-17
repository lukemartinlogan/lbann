/* -*- c++ -*- ----------------------------------------------------------
   Eternia paged GEMM for LBANN -- HOST-SIDE BOUNDARY HEADER.

   This header is the entire contract between LBANN and the paged kernel, and
   it mentions neither CUDA nor Clio. The separation is required, not
   stylistic: the paging kernel suspends on a page fault using C++20 DEVICE
   COROUTINES, which only clang++-22 driving -x cuda compiles, while LBANN's
   CUDA is built with nvcc -- and nvcc rejects co_await in device code
   outright. So the device half is its own ExternalProject with its own
   toolchain, and everything crossing the line is a POD or an opaque pointer.

   WHAT THIS REPLACES
   ------------------
   The fully-connected forward GEMM, for the transposed case only:

       El::Gemm(TRANSPOSE, NORMAL, 1, W, X, 0, C)     i.e.  C = W^T * X

   W is held out of core and paged in from inside the kernel; X and C stay
   resident. That split follows the memory: W is (h x w) and dominates a wide
   layer, while X and C are (k x n) and (m x n) with n the mini-batch.

   WHY NO TRANSPOSE COPY IS NEEDED
   -------------------------------
   Hydrogen stores W COLUMN-major: element (i, j) at j*h + i. The kernel
   reads its weight matrix ROW-major: R[a][b] at a*k + b. Reading the same
   buffer row-major with m = w and k = h gives

       R[a][b] = buf[a*h + b] = W(b, a)      i.e.  R = W^T

   which is exactly the matrix C = W^T * X needs. The layouts are duals, so
   the transposed case costs no copy at all. The non-transposed case would
   need a real transpose and is left to El::Gemm.
------------------------------------------------------------------------- */

#ifndef LBANN_ETERNIA_GEMM_H
#define LBANN_ETERNIA_GEMM_H

#include <cstdint>

namespace eternia_lbann {

/** Geometry and cache sizing for the paged weight matrix. */
struct Config {
  /** CTE tag for this layer's weights. Must be unique per layer per run, or
   *  two layers inherit each other's pages. */
  const char* tag = "lbann_eternia_fc";
  int gpu_id = 0;

  /** Page granularity in bytes. */
  std::uint64_t page_bytes = 262144;

  /** Launch geometry. Every Eternia hold is BLOCK-COLLECTIVE, so the block is
   *  the unit that owns a page cache and the unit that suspends. nthreads
   *  must be a power of two. */
  std::uint32_t nblocks = 64;
  std::uint32_t nthreads = 128;

  /** Resident pages per block. VRAM used is about
   *  nblocks * slots * page_bytes. */
  std::uint32_t slots = 8;

  bool stats = false;
};

/** Paging activity of the last Forward call. */
struct Stats {
  std::uint64_t faults = 0;
  std::uint64_t evicts = 0;
  std::uint64_t get_errors = 0;
};

/** Opaque; defined in the clang-compiled translation unit. */
struct Context;

/**
 * Create a context for a weight matrix of `h` rows and `w` columns as
 * Hydrogen stores it, i.e. the kernel will compute C = W^T * X with W^T of
 * shape (w x h).
 *
 * @return null on failure; LastError() says why.
 */
Context* Create(const Config& cfg, int h, int w);
void Destroy(Context* ctx);

/**
 * Copy W's backing store into the CTE, page by page.
 *
 * @param w_colmajor  Hydrogen's column-major buffer, HOST memory.
 * @param ldim        leading dimension (row stride) of that buffer.
 *
 * Page by page so the host never needs a second copy of a matrix that by
 * assumption does not fit in GPU memory. Call once per weight update; the
 * device page cache is dropped at the start of every Forward, because
 * nothing invalidates resident pages when the host rewrites the store.
 */
bool UploadWeights(Context* ctx, const float* w_colmajor, int ldim);

/**
 * C = W^T * X, with W paged.
 *
 * X and C are DEVICE pointers to Hydrogen's column-major local matrices:
 * X is (k x n) with leading dimension ldx, C is (m x n) with ldc, where
 * k = h and m = w from Create.
 *
 * @return false on failure; LastError() says why.
 */
bool Forward(Context* ctx, const float* x_device, int ldx, int n,
             float* c_device, int ldc);

/**
 * dX = W * dC, the gradient with respect to the layer INPUT.
 *
 * Forward is C(m,n) = W_row(m,k) * X(k,n); this is its transpose-multiply
 * partner, dX(k,n) = W_row^T * dC(m,n), so it walks the same pages of W and
 * differs only in which index it accumulates into:
 *
 *     forward   C[c*ldc + i] = sum_j W[i*k + j] * X [c*ldx + j]
 *     backward  dX[c*ldx + j] = sum_i W[i*k + j] * dC[c*ldc + i]
 *
 * A block owning a page of W therefore contributes to MANY rows of dX rather
 * than owning a few, so the accumulation is a cross-block atomicAdd. That is
 * safe because dX is resident and k*n, not paged.
 *
 * This exists because keeping the weights off the GPU is not enough on its
 * own: backpropagation also multiplies by W, and with the weights in host
 * memory El::Gemm refuses ("Must call gemm with matrices on same device").
 *
 * @param dc_device  dC (m x n) column-major on the device, leading dim ldc
 * @param dx_device  dX (k x n) column-major on the device, leading dim ldx;
 *                   ZEROED by this call before accumulating
 */
bool BackwardInput(Context* ctx, const float* dc_device, int ldc, int n,
                   float* dx_device, int ldx);

/**
 * dW = dC * X^T, the gradient with respect to the WEIGHTS, held paged.
 *
 *     dW[i*k + j] = sum_c dC[c*ldc + i] * X[c*ldx + j]
 *
 * This is the piece that makes out-of-core training a real problem rather
 * than a kernel swap: dW is the same size as W, so it cannot live in GPU
 * memory either, and it is WRITTEN rather than read.
 *
 * That matters for correctness, not just capacity. Page caches are per block
 * and writeback granularity is a page, so two blocks owning parts of one page
 * would each cache it, each fill their own part, and each flush the whole
 * thing. Here every element of dW depends only on the resident dC and X, so a
 * block can own a PAGE-ALIGNED slice and write nothing outside it, and the
 * hazard does not arise.
 *
 * Call ReadWeightGradient to copy the result back out, page by page.
 */
bool WeightGradient(Context* ctx, const float* dc_device, int ldc, int n,
                    const float* x_device, int ldx);

/**
 * Copy the paged weight gradient back into a host buffer laid out the way
 * Hydrogen holds W: column-major (h x w) with leading dimension `ldim`.
 */
bool ReadWeightGradient(Context* ctx, float* dw_colmajor, int ldim);

Stats GetStats(Context* ctx);

/** Never null. */
const char* LastError();

/** True if this build actually contains the paged backend. The stub build
 *  returns false so the caller can fall back to El::Gemm with a clear
 *  message instead of failing to link. */
bool Available();

}  // namespace eternia_lbann

#endif
