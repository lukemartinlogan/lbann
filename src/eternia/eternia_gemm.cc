/* ----------------------------------------------------------------------
   Eternia paged GEMM for LBANN -- DEVICE SIDE.

   Compiled by clang++-22 with -x cuda -DCLIO_GPU_YIELD_CORO=ON. Nothing in
   LBANN includes this file; the only surface is eternia_gemm.h.

   The kernel is the one verified standalone in eternia_gemm_bench.cc, with
   one change: X and C are indexed COLUMN-major, because that is how Hydrogen
   hands them over. W stays row-major, which -- as the header explains -- is
   what Hydrogen's column-major buffer already is when read that way, so the
   transposed GEMM needs no copy.
------------------------------------------------------------------------- */

#include "eternia_gemm.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(ETERNIA_LBANN_ENABLED)

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/types.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

namespace eternia_lbann {

#if defined(CLIO_YIELD_CORO)
static constexpr u32 kYieldLaneBytes = 4096;
#else
static constexpr u32 kYieldLaneBytes = 256;
#endif

#if defined(CLIO_YIELD_CORO) && defined(__clang__) && defined(__CUDA__)
#define ETERNIA_LBANN_CORO 1
#endif

#if defined(ETERNIA_LBANN_CORO)

/** Wait out this block's writebacks by PARKING, not spinning. */
__device__ gy::YCoroTask FlushWaitCoro(gv::DeviceVector<float>& v)
{
  CLIO_CO_YIELD_WHEN((v.ReapFlushed(), v.ReapFetched()),
                     v.AnyTransferInFlight(), v.FlushWaitTag());
}

/**
 * C += W * X for this block's page-aligned slice of W.
 *
 * W is (m x k) row-major and PAGED. X is (k x n) and C is (m x n), both
 * COLUMN-major and resident, exactly as Hydrogen stores them.
 *
 * Each block owns a page-aligned element range of W. Within it, one thread
 * owns one (row, column) pair of C and walks the part of that row present in
 * the current page: one atomicAdd per (row, col) per page rather than one per
 * multiply. A row straddling a page boundary is finished by the block owning
 * the other page, which the atomic makes safe.
 */
__device__ gy::YCoroMain GemmCoro(gv::DeviceVector<float> W, const float* X,
                                  int ldx, float* C, int ldc, u64 m, u64 k,
                                  u64 n, const u64* elem_lo,
                                  const u64* elem_hi, u32 block)
{
  u64 run = 0;
  const u64 e0 = elem_lo[block], e1 = elem_hi[block];
  const u64 pe = W.h_->elems_per_page_;

  // DROP THIS BLOCK'S CACHE FIRST.
  //
  // The host rewrites W's backing store on every call -- the weights change
  // each optimizer step -- and NOTHING invalidates the pages already resident
  // on the device. Without this the kernel keeps serving the weights from the
  // first call for the rest of training.
  //
  // It shows up as a forward pass that is exact on the initial weights and
  // wrong afterwards: measured against El::Gemm on the same inputs, the
  // relative error was 1.4e-07 in epoch 0 and 1.2e-03 by epoch 1. The
  // objective still fell, so training looked like it was working.
  //
  // The LAMMPS integration needed exactly this and for exactly this reason.
  __syncthreads();
  if (threadIdx.x == 0) {
    W.DropAll();
  }
  __syncthreads();

  for (u64 off = e0; off < e1;) {
    const u64 seg_end = ((off / pe) + 1) * pe < e1 ? ((off / pe) + 1) * pe : e1;
    co_await W.HoldPageCoro(off, seg_end - off, &run);

    const u64 r_lo = off / k;
    const u64 r_hi = (seg_end - 1) / k;
    const u64 nrows = r_hi - r_lo + 1;

    for (u64 t = threadIdx.x; t < nrows * n; t += blockDim.x) {
      const u64 r = r_lo + t / n;   // row of C
      const u64 c = t % n;          // column of C
      const u64 rs = r * k;
      const u64 js = (off > rs ? off - rs : 0);
      const u64 je = (seg_end < rs + k ? seg_end - rs : k);
      double acc = 0.0;
      for (u64 j = js; j < je; ++j) {
        // X is COLUMN-major: X(j, c) sits at c*ldx + j.
        acc += static_cast<double>(W.at(rs + j)) *
               static_cast<double>(X[c * ldx + j]);
      }
      // C is COLUMN-major: C(r, c) sits at c*ldc + r.
      atomicAdd(&C[c * ldc + r], static_cast<float>(acc));
    }
    __syncthreads();
    off = seg_end;
  }
  co_return;
}

/**
 * dX = W_row^T * dC for this block's page-aligned slice of W.
 *
 * The same page walk as GemmCoro. The difference is the accumulation target:
 * a page of W holds whole ROWS i, and each element W[i*k + j] contributes to
 * dX(j, c) for every column c -- so a block contributes to many rows of dX
 * rather than owning a few, and the adds are cross-block.
 */
__device__ gy::YCoroMain BackwardCoro(gv::DeviceVector<float> W,
                                      const float* dC, int ldc, float* dX,
                                      int ldx, u64 m, u64 k, u64 n,
                                      const u64* elem_lo, const u64* elem_hi,
                                      u32 block)
{
  u64 run = 0;
  const u64 e0 = elem_lo[block], e1 = elem_hi[block];
  const u64 pe = W.h_->elems_per_page_;

  // Same reason as the forward: the host rewrites W between calls and nothing
  // invalidates resident pages.
  __syncthreads();
  if (threadIdx.x == 0) {
    W.DropAll();
  }
  __syncthreads();

  for (u64 off = e0; off < e1;) {
    const u64 seg_end = ((off / pe) + 1) * pe < e1 ? ((off / pe) + 1) * pe : e1;
    co_await W.HoldPageCoro(off, seg_end - off, &run);

    // One thread per (element of W, column of dC). Each W element is read
    // once and fans out across the mini-batch, which is what keeps the page
    // resident for the whole of its useful life.
    for (u64 t = threadIdx.x; t < (seg_end - off) * n; t += blockDim.x) {
      const u64 e = off + t / n;      // element of W
      const u64 c = t % n;            // column
      const u64 i = e / k;            // row of W  == row of dC
      const u64 j = e - i * k;        // column of W == row of dX
      const float w = W.at(e);
      if (w != 0.0f) {
        atomicAdd(&dX[c * ldx + j], w * dC[c * ldc + i]);
      }
    }
    __syncthreads();
    off = seg_end;
  }
  co_return;
}

/**
 * dW = dC * X^T for this block's page-aligned slice of dW.
 *
 * Every element depends only on the resident dC and X, so a block writes
 * nothing outside its own pages -- which is what makes a written paged array
 * safe here without the cross-block coordination the PME grid needed.
 */
__device__ gy::YCoroMain GradCoro(gv::DeviceVector<float> dW, const float* dC,
                                  int ldc, const float* X, int ldx, u64 m,
                                  u64 k, u64 n, const u64* elem_lo,
                                  const u64* elem_hi, u32 block)
{
  u64 run = 0;
  const u64 e0 = elem_lo[block], e1 = elem_hi[block];
  const u64 pe = dW.h_->elems_per_page_;

  __syncthreads();
  if (threadIdx.x == 0) {
    dW.DropAll();
  }
  __syncthreads();

  for (u64 off = e0; off < e1;) {
    const u64 seg_end = ((off / pe) + 1) * pe < e1 ? ((off / pe) + 1) * pe : e1;
    co_await dW.HoldPageCoro(off, seg_end - off, &run);

    for (u64 e = off + threadIdx.x; e < seg_end; e += blockDim.x) {
      const u64 i = e / k;
      const u64 j = e - i * k;
      double acc = 0.0;
      for (u64 c = 0; c < n; ++c) {
        acc += static_cast<double>(dC[c * ldc + i]) *
               static_cast<double>(X[c * ldx + j]);
      }
      // Assignment, not accumulation: this is the whole gradient for the
      // mini-batch, so a page that was never written before does not have to
      // have existed.
      dW[e] = static_cast<float>(acc);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      dW.BeginFlush(off, seg_end - off);
    }
    __syncthreads();
    off = seg_end;
  }
  co_await FlushWaitCoro(dW);
}

__global__ void GradKernel(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<float> dW, const float* dC, int ldc,
                           const float* X, int ldx, u64 m, u64 k, u64 n,
                           const u64* elem_lo, const u64* elem_hi,
                           gy::YieldableView<> yv, gy::YieldStackView ys)
{
  CLIO_GPU_INIT(info, nullptr);
  dW.block_override_ = yv.Block();
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(GradCoro(dW, dC, ldc, X, ldx, m, k, n, elem_lo, elem_hi,
                          yv.Block()));
}

__global__ void BackwardKernel(clio::run::IpcManagerGpuInfo info,
                               gv::DeviceVector<float> W, const float* dC,
                               int ldc, float* dX, int ldx, u64 m, u64 k, u64 n,
                               const u64* elem_lo, const u64* elem_hi,
                               gy::YieldableView<> yv, gy::YieldStackView ys)
{
  CLIO_GPU_INIT(info, nullptr);
  W.block_override_ = yv.Block();
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(BackwardCoro(W, dC, ldc, dX, ldx, m, k, n, elem_lo, elem_hi,
                              yv.Block()));
}

__global__ void GemmKernel(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<float> W, const float* X, int ldx,
                           float* C, int ldc, u64 m, u64 k, u64 n,
                           const u64* elem_lo, const u64* elem_hi,
                           gy::YieldableView<> yv, gy::YieldStackView ys)
{
  CLIO_GPU_INIT(info, nullptr);
  W.block_override_ = yv.Block();
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(
    GemmCoro(W, X, ldx, C, ldc, m, k, n, elem_lo, elem_hi, yv.Block()));
}

#if !CTP_IS_DEVICE_PASS
class YieldRunner
{
public:
  YieldRunner(unsigned nb, unsigned nt) : drv_(nb, nt), stack_(nb, nt, kYieldLaneBytes) {}
  template <typename LaunchT>
  u32 Run(LaunchT&& launch)
  {
    drv_.Reset();
    stack_.Reset();
    return drv_.RunToCompletion(
      [&](dim3 g, dim3 b, gy::YieldableView<> v) { launch(g, b, v, stack_.View()); },
      [] {},
      /*max_rounds=*/2000000);
  }

private:
  gy::Yieldable<> drv_;
  gy::YieldStack stack_;
};
#endif

#endif  // ETERNIA_LBANN_CORO

// HOST ONLY below: clang compiles this file twice and the device pass has
// neither std::vector's allocator machinery nor the CTE client.
#if !CTP_IS_DEVICE_PASS

namespace {
std::string g_err;
void SetErr(const char* s) { g_err = s; }
}  // namespace

struct Context
{
  Config cfg;
  int h = 0, w = 0;   // W as Hydrogen stores it
  u64 m = 0, k = 0;   // W^T shape: m = w, k = h
#if defined(ETERNIA_LBANN_CORO)
  gv::Vector<float>* W = nullptr;
  /** The weight gradient, paged: it is the same size as W, so it cannot be
   *  resident either. Created lazily -- inference never needs it. */
  gv::Vector<float>* dW = nullptr;
  u64* d_lo = nullptr;
  u64* d_hi = nullptr;
#endif
  Stats stats;
};

bool Available()
{
#if defined(ETERNIA_LBANN_CORO)
  return true;
#else
  return false;
#endif
}

const char* LastError() { return g_err.c_str(); }

Context* Create(const Config& cfg, int h, int w)
{
#if !defined(ETERNIA_LBANN_CORO)
  (void)cfg; (void)h; (void)w;
  SetErr("built without CUDA and a coroutine-capable clang");
  return nullptr;
#else
  if (h <= 0 || w <= 0) { SetErr("Create: non-positive dimensions"); return nullptr; }
  if (cfg.nthreads == 0 || cfg.nthreads > 1024 ||
      (cfg.nthreads & (cfg.nthreads - 1)) != 0) {
    SetErr("threads per block must be a power of two");
    return nullptr;
  }
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    SetErr("Clio runtime init failed");
    return nullptr;
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    SetErr("Clio CTE client init failed");
    return nullptr;
  }
  auto* ctx = new Context();
  ctx->cfg = cfg;
  ctx->h = h; ctx->w = w;
  ctx->m = static_cast<u64>(w);
  ctx->k = static_cast<u64>(h);
  try {
    ctx->W = new gv::Vector<float>(std::string(cfg.tag), {cfg.gpu_id},
                                   cfg.page_bytes, cfg.nblocks, cfg.slots,
                                   ctx->m * ctx->k);
  } catch (const std::exception& e) {
    SetErr(e.what());
    delete ctx;
    return nullptr;
  }
  if (cfg.stats) ctx->W->EnableStats();

  // Page-aligned ownership: blocks must never share a page.
  const u64 pe = cfg.page_bytes / sizeof(float);
  const u64 np = (ctx->m * ctx->k + pe - 1) / pe;
  std::vector<u64> lo(cfg.nblocks), hi(cfg.nblocks);
  for (u32 b = 0; b < cfg.nblocks; ++b) {
    const u64 p0 = (np * b) / cfg.nblocks;
    const u64 p1 = (np * (b + 1)) / cfg.nblocks;
    lo[b] = p0 * pe < ctx->m * ctx->k ? p0 * pe : ctx->m * ctx->k;
    hi[b] = p1 * pe < ctx->m * ctx->k ? p1 * pe : ctx->m * ctx->k;
  }
  cudaMalloc(&ctx->d_lo, cfg.nblocks * sizeof(u64));
  cudaMalloc(&ctx->d_hi, cfg.nblocks * sizeof(u64));
  cudaMemcpy(ctx->d_lo, lo.data(), cfg.nblocks * sizeof(u64), cudaMemcpyHostToDevice);
  cudaMemcpy(ctx->d_hi, hi.data(), cfg.nblocks * sizeof(u64), cudaMemcpyHostToDevice);
  return ctx;
#endif
}

void Destroy(Context* ctx)
{
  if (!ctx) return;
#if defined(ETERNIA_LBANN_CORO)
  delete ctx->W;
  delete ctx->dW;
  if (ctx->d_lo) cudaFree(ctx->d_lo);
  if (ctx->d_hi) cudaFree(ctx->d_hi);
#endif
  delete ctx;
}

bool UploadWeights(Context* ctx, const float* w_colmajor, int ldim)
{
#if !defined(ETERNIA_LBANN_CORO)
  (void)ctx; (void)w_colmajor; (void)ldim;
  return false;
#else
  if (!ctx || !w_colmajor) { SetErr("UploadWeights: null argument"); return false; }
  clio::cte::core::Client core(clio::cte::core::kCtePoolId);
  const u64 total = ctx->m * ctx->k;
  const u64 pe = ctx->cfg.page_bytes / sizeof(float);
  const u64 np = (total + pe - 1) / pe;
  std::vector<float> buf(pe);
  for (u64 p = 0; p < np; ++p) {
    const u64 e0 = p * pe;
    const u64 e1 = (e0 + pe < total) ? (e0 + pe) : total;
    std::memset(buf.data(), 0, buf.size() * sizeof(float));
    for (u64 e = e0; e < e1; ++e) {
      // Row-major index e over W^T maps to (a, b) = (e / k, e % k), and
      // W^T(a, b) = W(b, a), which in Hydrogen's column-major buffer with
      // leading dimension ldim sits at a*ldim + b.
      const u64 a = e / ctx->k, b = e % ctx->k;
      buf[e - e0] = w_colmajor[a * static_cast<u64>(ldim) + b];
    }
    char nm[32];
    gv::PageBlobName(p, nm);
    auto f = core.AsyncPutBlob(ctx->W->TagId(), std::string(nm), 0,
                               buf.size() * sizeof(float),
                               reinterpret_cast<const char*>(buf.data()), 1.0f);
    f.Wait();
    if (f.get() == nullptr || f->GetReturnCode() != 0) {
      SetErr("UploadWeights: a page write failed");
      return false;
    }
  }
  return true;
#endif
}

bool Forward(Context* ctx, const float* x_device, int ldx, int n,
             float* c_device, int ldc)
{
#if !defined(ETERNIA_LBANN_CORO)
  (void)ctx; (void)x_device; (void)ldx; (void)n; (void)c_device; (void)ldc;
  SetErr("built without the paged backend");
  return false;
#else
  if (!ctx || !x_device || !c_device) { SetErr("Forward: null argument"); return false; }
  auto gpu = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(ctx->cfg.gpu_id);

  // C is accumulated into, so it must start at zero: El::Gemm's beta is 0.
  cudaMemset2D(c_device, static_cast<size_t>(ldc) * sizeof(float), 0,
               ctx->m * sizeof(float), static_cast<size_t>(n));

  auto dW = ctx->W->GetDevice(ctx->cfg.gpu_id);
  YieldRunner runner(ctx->cfg.nblocks, ctx->cfg.nthreads);
  const u32 rounds = runner.Run(
    [&](dim3 g, dim3 b, gy::YieldableView<> v, gy::YieldStackView sv) {
      GemmKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
        gpu, dW, x_device, ldx, c_device, ldc, ctx->m, ctx->k,
        static_cast<u64>(n), ctx->d_lo, ctx->d_hi, v, sv);
    });
  const cudaError_t le = cudaGetLastError();
  if (le != cudaSuccess) { SetErr(cudaGetErrorString(le)); return false; }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    SetErr(cudaGetErrorString(cudaGetLastError()));
    return false;
  }
  if (rounds == 0) { SetErr("yield driver made no progress"); return false; }

  if (ctx->cfg.stats) {
    auto s = ctx->W->ReadStats(ctx->cfg.gpu_id);
    ctx->stats.faults = s.faults;
    ctx->stats.evicts = s.evicts;
    ctx->stats.get_errors = s.get_errors;
  }
  return true;
#endif
}

bool BackwardInput(Context* ctx, const float* dc_device, int ldc, int n,
                   float* dx_device, int ldx)
{
#if !defined(ETERNIA_LBANN_CORO)
  (void)ctx; (void)dc_device; (void)ldc; (void)n; (void)dx_device; (void)ldx;
  SetErr("built without the paged backend");
  return false;
#else
  if (!ctx || !dc_device || !dx_device) {
    SetErr("BackwardInput: null argument");
    return false;
  }
  auto gpu = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(ctx->cfg.gpu_id);
  // dX is accumulated into, so it starts at zero.
  cudaMemset2D(dx_device, static_cast<size_t>(ldx) * sizeof(float), 0,
               ctx->k * sizeof(float), static_cast<size_t>(n));

  auto dW = ctx->W->GetDevice(ctx->cfg.gpu_id);
  YieldRunner runner(ctx->cfg.nblocks, ctx->cfg.nthreads);
  const u32 rounds = runner.Run(
    [&](dim3 g, dim3 b, gy::YieldableView<> v, gy::YieldStackView sv) {
      BackwardKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
        gpu, dW, dc_device, ldc, dx_device, ldx, ctx->m, ctx->k,
        static_cast<u64>(n), ctx->d_lo, ctx->d_hi, v, sv);
    });
  const cudaError_t le = cudaGetLastError();
  if (le != cudaSuccess) { SetErr(cudaGetErrorString(le)); return false; }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    SetErr(cudaGetErrorString(cudaGetLastError()));
    return false;
  }
  if (rounds == 0) { SetErr("yield driver made no progress"); return false; }
  return true;
#endif
}

bool WeightGradient(Context* ctx, const float* dc_device, int ldc, int n,
                    const float* x_device, int ldx)
{
#if !defined(ETERNIA_LBANN_CORO)
  (void)ctx; (void)dc_device; (void)ldc; (void)n; (void)x_device; (void)ldx;
  SetErr("built without the paged backend");
  return false;
#else
  if (!ctx || !dc_device || !x_device) {
    SetErr("WeightGradient: null argument");
    return false;
  }
  if (ctx->dW == nullptr) {
    try {
      ctx->dW = new gv::Vector<float>(std::string(ctx->cfg.tag) + "_grad",
                                      {ctx->cfg.gpu_id}, ctx->cfg.page_bytes,
                                      ctx->cfg.nblocks, ctx->cfg.slots,
                                      ctx->m * ctx->k);
    } catch (const std::exception& e) {
      SetErr(e.what());
      return false;
    }
    // The gradient is WRITTEN, and a hold faults a page in before the kernel
    // writes it -- a page whose blob does not exist yet comes back as a failed
    // get. Create them once. Same trap as the LAMMPS force vector and the PME
    // grid.
    clio::cte::core::Client core(clio::cte::core::kCtePoolId);
    const u64 pe = ctx->cfg.page_bytes / sizeof(float);
    const u64 np = (ctx->m * ctx->k + pe - 1) / pe;
    std::vector<float> zeros(pe, 0.0f);
    for (u64 p = 0; p < np; ++p) {
      char nm[32];
      gv::PageBlobName(p, nm);
      auto f = core.AsyncPutBlob(ctx->dW->TagId(), std::string(nm), 0,
                                 zeros.size() * sizeof(float),
                                 reinterpret_cast<const char*>(zeros.data()), 1.0f);
      f.Wait();
      if (f.get() == nullptr || f->GetReturnCode() != 0) {
        SetErr("WeightGradient: could not create the gradient's backing store");
        return false;
      }
    }
  }

  auto gpu = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(ctx->cfg.gpu_id);
  auto ddW = ctx->dW->GetDevice(ctx->cfg.gpu_id);
  YieldRunner runner(ctx->cfg.nblocks, ctx->cfg.nthreads);
  const u32 rounds = runner.Run(
    [&](dim3 g, dim3 b, gy::YieldableView<> v, gy::YieldStackView sv) {
      GradKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
        gpu, ddW, dc_device, ldc, x_device, ldx, ctx->m, ctx->k,
        static_cast<u64>(n), ctx->d_lo, ctx->d_hi, v, sv);
    });
  const cudaError_t le = cudaGetLastError();
  if (le != cudaSuccess) { SetErr(cudaGetErrorString(le)); return false; }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    SetErr(cudaGetErrorString(cudaGetLastError()));
    return false;
  }
  if (rounds == 0) { SetErr("yield driver made no progress"); return false; }
  return true;
#endif
}

bool ReadWeightGradient(Context* ctx, float* dw_colmajor, int ldim)
{
#if !defined(ETERNIA_LBANN_CORO)
  (void)ctx; (void)dw_colmajor; (void)ldim;
  return false;
#else
  if (!ctx || !ctx->dW || !dw_colmajor) {
    SetErr("ReadWeightGradient: nothing to read");
    return false;
  }
  clio::cte::core::Client core(clio::cte::core::kCtePoolId);
  const u64 total = ctx->m * ctx->k;
  const u64 pe = ctx->cfg.page_bytes / sizeof(float);
  const u64 np = (total + pe - 1) / pe;
  std::vector<float> buf(pe);
  for (u64 p = 0; p < np; ++p) {
    char nm[32];
    gv::PageBlobName(p, nm);
    auto f = core.AsyncGetBlob(ctx->dW->TagId(), std::string(nm), 0,
                               buf.size() * sizeof(float), 0u,
                               reinterpret_cast<char*>(buf.data()));
    f.Wait();
    if (f.get() == nullptr || f->GetReturnCode() != 0) {
      SetErr("ReadWeightGradient: a page read failed");
      return false;
    }
    const u64 e0 = p * pe;
    const u64 e1 = (e0 + pe < total) ? (e0 + pe) : total;
    for (u64 e = e0; e < e1; ++e) {
      // row-major index e over dW maps to Hydrogen's column-major (j, i)
      const u64 i = e / ctx->k, j = e - i * ctx->k;
      dw_colmajor[i * static_cast<u64>(ldim) + j] = buf[e - e0];
    }
  }
  return true;
#endif
}

Stats GetStats(Context* ctx) { return ctx ? ctx->stats : Stats(); }

#endif  // !CTP_IS_DEVICE_PASS

}  // namespace eternia_lbann

#else   // !ETERNIA_LBANN_ENABLED

namespace eternia_lbann {
struct Context {};
bool Available() { return false; }
const char* LastError()
{
  return "this LBANN was built without the Eternia backend";
}
Context* Create(const Config&, int, int) { return nullptr; }
void Destroy(Context*) {}
bool UploadWeights(Context*, const float*, int) { return false; }
bool Forward(Context*, const float*, int, int, float*, int) { return false; }
bool BackwardInput(Context*, const float*, int, int, float*, int) { return false; }
bool WeightGradient(Context*, const float*, int, int, const float*, int) { return false; }
bool ReadWeightGradient(Context*, float*, int) { return false; }
Stats GetStats(Context*) { return Stats(); }
}  // namespace eternia_lbann

#endif  // ETERNIA_LBANN_ENABLED
