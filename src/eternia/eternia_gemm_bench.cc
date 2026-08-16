/* ----------------------------------------------------------------------
   LBANN fully-connected forward over an Eternia (Clio CTE gpu_vector)
   weight matrix.

   STANDALONE BY DESIGN: paged kernel, CPU reference, configuration sweep and
   an analytic check for sizes too large to reference, all in one program with
   its own main(). Not yet wired into LBANN. Proving the kernel first is what
   made the GROMACS integration take one round where LAMMPS took three.

   WHAT IS OUT OF CORE
   -------------------
   The weight matrix W, m x k floats. LBANN's fully-connected forward is an
   El::Gemm of W against the layer input, and for a wide layer W is the thing
   that does not fit: the activations are k x n with n the mini-batch, which
   is small beside m*k. So W is paged and the activations stay resident.

       C (m x n)  =  W (m x k)  *  X (k x n)

   W is stored ROW-MAJOR here, which is a deliberate departure from
   Hydrogen's column-major convention and the reason the decomposition works
   at all: a page of a row-major W is a run of whole rows, and a block that
   owns those rows can compute the corresponding rows of C completely. In
   column-major a page would be a run of whole COLUMNS, i.e. a partial
   contribution to every row of C at once, so every block would have to
   accumulate into all of C and the write set would stop being disjoint.
   Wiring this into LBANN therefore means transposing W once at load, not
   paging Hydrogen's buffer where it lies.

   HOW THE KERNEL IS ORGANISED
   ---------------------------
   Each block owns a PAGE-ALIGNED slice of W. That is not a tuning choice:
   page caches are per block and writeback granularity is a page, so two
   blocks sharing a page would each cache it and each flush it whole. (W is
   read-only here, so the hazard is milder than on a written grid, but the
   ownership rule is kept because it also makes the C write set disjoint.)

   Within a slice, one thread owns one (row, column) pair of C and walks the
   part of that row present in the current page. That gives ONE atomicAdd per
   (row, col) per page rather than one per multiply, and rows that straddle a
   page boundary are summed correctly across the two owning blocks.
------------------------------------------------------------------------- */

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/types.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

#if defined(CLIO_YIELD_CORO)
static constexpr u32 kYieldLaneBytes = 4096;
#else
static constexpr u32 kYieldLaneBytes = 256;
#endif

#if defined(CLIO_YIELD_CORO) && defined(__clang__) && defined(__CUDA__)
#define ETERNIA_GEMM_CORO 1
#endif

/**
 * The weight pattern, evaluated identically on host and device.
 *
 * A pattern rather than stored random data because at the sizes that matter
 * W does not fit in host memory either. It is cheap, it is not constant
 * (so a kernel that dropped or double-counted terms would show up), and its
 * row sums have a closed form -- see RowSumOnes.
 */
CTP_INLINE_CROSS_FUN float WeightAt(u64 i, u64 j) {
  return static_cast<float>(static_cast<int>((i + j) & 3u)) - 1.5f;
}

/**
 * Exact value of row i of C when X is all ones: sum over j of W(i,j).
 *
 * ((i+j) & 3) cycles 0,1,2,3 with period 4, so a full period contributes
 * 0+1+2+3 - 4*1.5 = 0 and only the remainder of k mod 4 survives. That makes
 * the expected result computable in O(1) per row, which is what allows a
 * grid larger than host memory to be checked at all.
 */
CTP_INLINE_CROSS_FUN double RowSumOnes(u64 i, u64 k) {
  const u64 whole = k / 4, rem = k % 4;
  double s = 0.0;   // each whole period sums to exactly zero
  (void)whole;
  for (u64 t = 0; t < rem; ++t) {
    s += static_cast<double>(static_cast<int>((i + (k - rem) + t) & 3u)) - 1.5;
  }
  return s;
}

#if defined(ETERNIA_GEMM_CORO)

/**
 * C += W * X for this block's page-aligned slice of W.
 *
 * `elem_lo/elem_hi` bound the slice. X is (k x n) row-major and resident; C
 * is (m x n) row-major and resident.
 */
__device__ gy::YCoroMain GemmCoro(gv::DeviceVector<float> W, const float *X,
                                  float *C, u64 m, u64 k, u64 n,
                                  const u64 *elem_lo, const u64 *elem_hi,
                                  u32 block) {
  u64 run = 0;
  const u64 e0 = elem_lo[block], e1 = elem_hi[block];
  const u64 pe = W.h_->elems_per_page_;

  for (u64 off = e0; off < e1; ) {
    const u64 seg_end = ((off / pe) + 1) * pe < e1 ? ((off / pe) + 1) * pe : e1;
    co_await W.HoldPageCoro(off, seg_end - off, &run);

    // Rows of W (and so of C) this page touches.
    const u64 r_lo = off / k;
    const u64 r_hi = (seg_end - 1) / k;
    const u64 nrows = r_hi - r_lo + 1;

    // One thread per (row, column) pair: one atomicAdd each, instead of one
    // per multiply.
    for (u64 t = threadIdx.x; t < nrows * n; t += blockDim.x) {
      const u64 r = r_lo + t / n;
      const u64 c = t % n;
      // The part of row r that lives in THIS page.
      const u64 rs = r * k;
      const u64 js = (off > rs ? off - rs : 0);
      const u64 je = (seg_end < rs + k ? seg_end - rs : k);
      double acc = 0.0;
      for (u64 j = js; j < je; ++j) {
        acc += static_cast<double>(W.at(rs + j)) *
               static_cast<double>(X[j * n + c]);
      }
      // atomicAdd because a row straddling a page boundary is finished by
      // the block that owns the other page.
      atomicAdd(&C[r * n + c], static_cast<float>(acc));
    }
    __syncthreads();
    off = seg_end;
  }
  co_return;
}

__global__ void GemmKernel(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<float> W, const float *X, float *C,
                           u64 m, u64 k, u64 n, const u64 *elem_lo,
                           const u64 *elem_hi, gy::YieldableView<> yv,
                           gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  W.block_override_ = yv.Block();
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(GemmCoro(W, X, C, m, k, n, elem_lo, elem_hi, yv.Block()));
}

#if !CTP_IS_DEVICE_PASS
class YieldRunner {
 public:
  YieldRunner(unsigned nblocks, unsigned nthreads)
      : drv_(nblocks, nthreads), stack_(nblocks, nthreads, kYieldLaneBytes) {}
  template <typename LaunchT>
  u32 Run(LaunchT &&launch) {
    drv_.Reset();
    stack_.Reset();
    return drv_.RunToCompletion(
        [&](dim3 g, dim3 b, gy::YieldableView<> view) {
          launch(g, b, view, stack_.View());
        },
        [] {}, /*max_rounds=*/2000000);
  }

 private:
  gy::Yieldable<> drv_;
  gy::YieldStack stack_;
};
#endif

#endif  // ETERNIA_GEMM_CORO

// HOST ONLY below: clang compiles this file twice and the device pass has
// neither std::vector's allocator machinery nor the CTE client.
#if !CTP_IS_DEVICE_PASS

int main(int argc, char **argv) {
  u64 m = 4096, k = 4096, n = 16;
  u64 page_kb = 256;
  u32 blocks = 16, threads = 128, slots = 8;
  bool ones = false;          // X = all ones, check against the analytic sum
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return static_cast<u64>(std::atoll(argv[++i])); };
    if (a == "--m") m = next();
    else if (a == "--k") k = next();
    else if (a == "--n") n = next();
    else if (a == "--page-kb") page_kb = next();
    else if (a == "--blocks") blocks = static_cast<u32>(next());
    else if (a == "--threads") threads = static_cast<u32>(next());
    else if (a == "--slots") slots = static_cast<u32>(next());
    else if (a == "--ones") ones = true;
    else {
      std::fprintf(stderr,
                   "usage: %s [--m N] [--k N] [--n N] [--page-kb N]\n"
                   "          [--blocks N] [--threads N] [--slots N] [--ones]\n"
                   "  --ones: X = 1, checked against an analytic row sum, for\n"
                   "          weight matrices too large to reference on the host\n",
                   argv[0]);
      return 2;
    }
  }

  const u64 welems = m * k;
  const double w_gib = static_cast<double>(welems) * sizeof(float) /
                       (1024.0 * 1024.0 * 1024.0);
  std::printf("LBANN fully-connected forward over an Eternia weight matrix\n"
              "  W=%llux%llu = %llu elems = %.3f GiB   X=%llux%llu\n"
              "  page=%lluKB blocks=%u threads=%u slots=%u  check=%s\n",
              (unsigned long long)m, (unsigned long long)k,
              (unsigned long long)welems, w_gib, (unsigned long long)k,
              (unsigned long long)n, (unsigned long long)page_kb, blocks,
              threads, slots, ones ? "analytic" : "cpu-reference");

#if !defined(ETERNIA_GEMM_CORO)
  std::fprintf(stderr, "built without CUDA + clang device coroutines\n");
  return 1;
#else
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "runtime init failed\n");
    return 1;
  }
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "cte client init failed\n");
    return 1;
  }
  auto gpu = CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  gv::Vector<float> W("lbann_eternia_w", {0}, page_kb * 1024, blocks, slots,
                      welems);
  W.EnableStats();

  // Fill W's backing store from the host, one page at a time. Page by page
  // because at the sizes this exists for, W does not fit in host memory
  // either -- which is also why the contents come from a pattern rather than
  // from a stored array.
  {
    clio::cte::core::Client core(clio::cte::core::kCtePoolId);
    const u64 pe = (page_kb * 1024) / sizeof(float);
    const u64 npages = (welems + pe - 1) / pe;
    std::vector<float> buf(pe);
    for (u64 p = 0; p < npages; ++p) {
      const u64 lo = p * pe;
      const u64 hi = std::min(lo + pe, welems);
      std::memset(buf.data(), 0, buf.size() * sizeof(float));
      for (u64 e = lo; e < hi; ++e) buf[e - lo] = WeightAt(e / k, e % k);
      char nm[32];
      gv::PageBlobName(p, nm);
      auto fut = core.AsyncPutBlob(W.TagId(), std::string(nm), 0,
                                   buf.size() * sizeof(float),
                                   reinterpret_cast<const char *>(buf.data()),
                                   1.0f);
      fut.Wait();
      if (fut.get() == nullptr || fut->GetReturnCode() != 0) {
        std::fprintf(stderr, "could not write W page %llu\n",
                     (unsigned long long)p);
        return 1;
      }
    }
  }

  std::vector<float> hX(k * n), hC(m * n, 0.0f);
  {
    unsigned s = 999u;
    auto rnd = [&]() {
      s = s * 1664525u + 1013904223u;
      return static_cast<float>(s >> 8) / static_cast<float>(1u << 24);
    };
    for (u64 i = 0; i < k * n; ++i) hX[i] = ones ? 1.0f : (rnd() * 2.0f - 1.0f);
  }
  float *dX = nullptr, *dC = nullptr;
  cudaMalloc(&dX, hX.size() * sizeof(float));
  cudaMalloc(&dC, hC.size() * sizeof(float));
  cudaMemcpy(dX, hX.data(), hX.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(dC, 0, hC.size() * sizeof(float));

  // Page-aligned ownership: blocks must not share a page.
  const u64 pe_host = (page_kb * 1024) / sizeof(float);
  const u64 npages_host = (welems + pe_host - 1) / pe_host;
  std::vector<u64> elem_lo(blocks), elem_hi(blocks);
  for (u32 b = 0; b < blocks; ++b) {
    const u64 p0 = (npages_host * b) / blocks;
    const u64 p1 = (npages_host * (b + 1)) / blocks;
    elem_lo[b] = std::min(p0 * pe_host, welems);
    elem_hi[b] = std::min(p1 * pe_host, welems);
  }
  u64 *dlo = nullptr, *dhi = nullptr;
  cudaMalloc(&dlo, blocks * sizeof(u64));
  cudaMalloc(&dhi, blocks * sizeof(u64));
  cudaMemcpy(dlo, elem_lo.data(), blocks * sizeof(u64), cudaMemcpyHostToDevice);
  cudaMemcpy(dhi, elem_hi.data(), blocks * sizeof(u64), cudaMemcpyHostToDevice);

  auto dW = W.GetDevice(0);
  using clock = std::chrono::high_resolution_clock;
  const auto t0 = clock::now();
  YieldRunner runner(blocks, threads);
  const u32 rounds = runner.Run(
      [&](dim3 g, dim3 b, gy::YieldableView<> vw, gy::YieldStackView sv) {
        GemmKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(gpu, dW, dX, dC, m, k, n,
                                                    dlo, dhi, vw, sv);
      });
  const cudaError_t le = cudaGetLastError();
  if (le != cudaSuccess) {
    std::fprintf(stderr, "LAUNCH FAILED: %s\n", cudaGetErrorString(le));
    return 1;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "gemm failed: %s\n",
                 cudaGetErrorString(cudaGetLastError()));
    return 1;
  }
  const double ms =
      std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost);

  const auto st = W.ReadStats(0);
  std::printf("  gemm: %.1f ms rounds=%u | faults=%llu evicts=%llu "
              "get_err=%llu\n",
              ms, rounds, (unsigned long long)st.faults,
              (unsigned long long)st.evicts,
              (unsigned long long)st.get_errors);
  if (st.get_errors) {
    std::fprintf(stderr, "FAIL: %llu failed page reads\n",
                 (unsigned long long)st.get_errors);
    return 1;
  }

  double max_rel = 0.0, peak = 0.0;
  if (ones) {
    // Analytic: every column of C equals the row sum of W. O(m), so this
    // scales to weight matrices far larger than host memory.
    for (u64 i = 0; i < m; ++i) {
      const double want = RowSumOnes(i, k);
      peak = std::max(peak, std::fabs(want));
      for (u64 c = 0; c < n; ++c) {
        const double got = hC[i * n + c];
        max_rel = std::max(max_rel, std::fabs(got - want));
      }
    }
  } else {
    // Full CPU reference, for sizes where it is affordable.
    for (u64 i = 0; i < m; ++i) {
      for (u64 c = 0; c < n; ++c) {
        double acc = 0.0;
        for (u64 j = 0; j < k; ++j) {
          acc += static_cast<double>(WeightAt(i, j)) *
                 static_cast<double>(hX[j * n + c]);
        }
        peak = std::max(peak, std::fabs(acc));
        max_rel = std::max(max_rel, std::fabs(acc - hC[i * n + c]));
      }
    }
  }
  // The accumulation is a sum of k single-precision products, so the
  // tolerance has to scale with k as well as with the magnitude present.
  const double tol = 1e-5 * std::max(peak, 1.0) * std::sqrt((double)k);
  const bool ok = max_rel <= tol;
  std::printf("  verify: max|diff|=%.4e peak=%.4e tol=%.4e\n%s\n", max_rel,
              peak, tol, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
#endif
}

#endif  // !CTP_IS_DEVICE_PASS
