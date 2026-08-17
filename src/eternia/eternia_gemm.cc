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
Stats GetStats(Context*) { return Stats(); }
}  // namespace eternia_lbann

#endif  // ETERNIA_LBANN_ENABLED
