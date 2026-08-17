/* ----------------------------------------------------------------------
   Validation for the paged GEMM as LBANN actually calls it.

   This exercises the eternia_gemm.h boundary with the SAME layouts the
   fully-connected layer passes: W column-major (h x w) on the host, X
   column-major (k x n) on the device, C column-major (m x n) on the device,
   computing C = W^T * X.

   That is the point of testing here rather than only in the standalone
   bench: the bench proved the kernel, but the integration adds a layout
   contract -- Hydrogen's column-major buffer reinterpreted as the kernel's
   row-major weight matrix -- and a layout contract is exactly the kind of
   thing that is plausible on paper and wrong in practice.

   The reference is computed independently on the host from the same W and X.
------------------------------------------------------------------------- */

#include "eternia_gemm.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace {

/** Reference C = W^T * X, all column-major, computed in double. */
void reference(const std::vector<float>& W, int h, int w,
               const std::vector<float>& X, int k, int n,
               std::vector<double>* C)
{
  // W is (h x w) column-major: W(i,j) = W[j*h + i]
  // W^T is (w x h), so C is (w x n): C(a,c) = sum_b W^T(a,b) * X(b,c)
  //                                        = sum_b W(b,a)   * X(b,c)
  C->assign(static_cast<size_t>(w) * n, 0.0);
  for (int a = 0; a < w; ++a) {
    for (int c = 0; c < n; ++c) {
      double acc = 0.0;
      for (int b = 0; b < k; ++b) {
        acc += static_cast<double>(W[static_cast<size_t>(a) * h + b]) *
               static_cast<double>(X[static_cast<size_t>(c) * k + b]);
      }
      (*C)[static_cast<size_t>(c) * w + a] = acc;
    }
  }
}

}  // namespace

int main(int argc, char** argv)
{
  int h = 512, w = 256, n = 8;
  std::uint64_t page_kb = 64;
  std::uint32_t blocks = 16, threads = 128, slots = 8;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return std::atoi(argv[++i]); };
    if (a == "--h") h = next();
    else if (a == "--w") w = next();
    else if (a == "--n") n = next();
    else if (a == "--page-kb") page_kb = static_cast<std::uint64_t>(next());
    else if (a == "--blocks") blocks = static_cast<std::uint32_t>(next());
    else if (a == "--threads") threads = static_cast<std::uint32_t>(next());
    else if (a == "--slots") slots = static_cast<std::uint32_t>(next());
    else {
      std::fprintf(stderr,
                   "usage: %s [--h N] [--w N] [--n N] [--page-kb N]\n"
                   "          [--blocks N] [--threads N] [--slots N]\n",
                   argv[0]);
      return 2;
    }
  }
  const int k = h, m = w;   // W^T is (m x k) = (w x h)

  if (!eternia_lbann::Available()) {
    std::fprintf(stderr, "built without the paged backend\n");
    return 1;
  }

  std::vector<float> W(static_cast<size_t>(h) * w);
  std::vector<float> X(static_cast<size_t>(k) * n);
  {
    unsigned s = 271828u;
    auto rnd = [&]() {
      s = s * 1664525u + 1013904223u;
      return static_cast<float>(s >> 8) / static_cast<float>(1u << 24) * 2.0f - 1.0f;
    };
    for (auto& v : W) v = rnd();
    for (auto& v : X) v = rnd();
  }

  std::vector<double> ref;
  reference(W, h, w, X, k, n, &ref);

  eternia_lbann::Config cfg;
  cfg.tag = "lbann_eternia_verify";
  cfg.page_bytes = page_kb * 1024;
  cfg.nblocks = blocks;
  cfg.nthreads = threads;
  cfg.slots = slots;
  cfg.stats = true;

  auto* ctx = eternia_lbann::Create(cfg, h, w);
  if (ctx == nullptr) {
    std::fprintf(stderr, "Create failed: %s\n", eternia_lbann::LastError());
    return 1;
  }
  if (!eternia_lbann::UploadWeights(ctx, W.data(), h)) {
    std::fprintf(stderr, "UploadWeights failed: %s\n", eternia_lbann::LastError());
    return 1;
  }

  float *dX = nullptr, *dC = nullptr;
  cudaMalloc(&dX, X.size() * sizeof(float));
  cudaMalloc(&dC, static_cast<size_t>(m) * n * sizeof(float));
  cudaMemcpy(dX, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice);

  if (!eternia_lbann::Forward(ctx, dX, k, n, dC, m)) {
    std::fprintf(stderr, "Forward failed: %s\n", eternia_lbann::LastError());
    return 1;
  }

  std::vector<float> got(static_cast<size_t>(m) * n);
  cudaMemcpy(got.data(), dC, got.size() * sizeof(float), cudaMemcpyDeviceToHost);

  // ---- backward: dX = W * dC, against an independent host reference ----
  // Checked separately from the forward because it is a DIFFERENT kernel with
  // a different accumulation pattern (cross-block atomicAdd into dX), not a
  // reuse of the forward one.
  std::vector<float> hdC(static_cast<size_t>(m) * n);
  {
    unsigned s2 = 31337u;
    auto rnd = [&]() {
      s2 = s2 * 1664525u + 1013904223u;
      return static_cast<float>(s2 >> 8) / static_cast<float>(1u << 24) * 2.0f - 1.0f;
    };
    for (auto& v : hdC) v = rnd();
  }
  std::vector<double> refdX(static_cast<size_t>(k) * n, 0.0);
  for (int j = 0; j < k; ++j) {
    for (int c = 0; c < n; ++c) {
      double acc = 0.0;
      for (int i = 0; i < m; ++i) {
        // W_row(i, j) is Hydrogen's W(j, i) = W[i*h + j]
        acc += static_cast<double>(W[static_cast<size_t>(i) * h + j]) *
               static_cast<double>(hdC[static_cast<size_t>(c) * m + i]);
      }
      refdX[static_cast<size_t>(c) * k + j] = acc;
    }
  }
  float *dDC = nullptr, *dDX = nullptr;
  cudaMalloc(&dDC, hdC.size() * sizeof(float));
  cudaMalloc(&dDX, static_cast<size_t>(k) * n * sizeof(float));
  cudaMemcpy(dDC, hdC.data(), hdC.size() * sizeof(float), cudaMemcpyHostToDevice);
  double bmax = 0.0, bpeak = 0.0;
  if (!eternia_lbann::BackwardInput(ctx, dDC, m, n, dDX, k)) {
    std::fprintf(stderr, "BackwardInput failed: %s\n", eternia_lbann::LastError());
    return 1;
  }
  {
    std::vector<float> gotdX(static_cast<size_t>(k) * n);
    cudaMemcpy(gotdX.data(), dDX, gotdX.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    for (size_t q = 0; q < gotdX.size(); ++q) {
      bpeak = std::max(bpeak, std::fabs(refdX[q]));
      bmax = std::max(bmax, std::fabs(refdX[q] - static_cast<double>(gotdX[q])));
    }
  }

  // ---- weight gradient: dW = dC * X^T, paged and WRITTEN ----
  // The only one of the three that writes a paged array, so it is also the
  // only one where page-aligned block ownership matters.
  std::vector<double> refdW(static_cast<size_t>(h) * w, 0.0);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      double acc = 0.0;
      for (int c = 0; c < n; ++c) {
        acc += static_cast<double>(hdC[static_cast<size_t>(c) * m + i]) *
               static_cast<double>(X[static_cast<size_t>(c) * k + j]);
      }
      // stored the way Hydrogen holds W: column-major (h x w), (j, i)
      refdW[static_cast<size_t>(i) * h + j] = acc;
    }
  }
  double gmax = 0.0, gpeak = 0.0;
  if (!eternia_lbann::WeightGradient(ctx, dDC, m, n, dX, k)) {
    std::fprintf(stderr, "WeightGradient failed: %s\n", eternia_lbann::LastError());
    return 1;
  }
  {
    std::vector<float> gotdW(static_cast<size_t>(h) * w, 0.0f);
    if (!eternia_lbann::ReadWeightGradient(ctx, gotdW.data(), h)) {
      std::fprintf(stderr, "ReadWeightGradient failed: %s\n",
                   eternia_lbann::LastError());
      return 1;
    }
    for (size_t q = 0; q < gotdW.size(); ++q) {
      gpeak = std::max(gpeak, std::fabs(refdW[q]));
      gmax = std::max(gmax, std::fabs(refdW[q] - static_cast<double>(gotdW[q])));
    }
  }

  const auto st = eternia_lbann::GetStats(ctx);
  double maxd = 0.0, peak = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    peak = std::max(peak, std::fabs(ref[i]));
    maxd = std::max(maxd, std::fabs(ref[i] - static_cast<double>(got[i])));
  }
  // The accumulation is a sum of k single-precision products, so the
  // tolerance scales with sqrt(k) as well as with the magnitude present.
  const double tol = 1e-5 * std::max(peak, 1.0) * std::sqrt(static_cast<double>(k));
  const double btol = 1e-5 * std::max(bpeak, 1.0) * std::sqrt(static_cast<double>(m));
  const double gtol = 1e-5 * std::max(gpeak, 1.0) * std::sqrt(static_cast<double>(n));
  const bool ok = (maxd <= tol) && (bmax <= btol) && (gmax <= gtol) &&
                  (st.get_errors == 0);

  std::printf("W=%dx%d (W^T = %dx%d)  X=%dx%d  page=%lluKB blocks=%u slots=%u\n"
              "  faults=%llu evicts=%llu get_errors=%llu\n"
              "  fwd max|diff|=%.4e peak=%.4e tol=%.4e\n"
              "  bwd max|diff|=%.4e peak=%.4e tol=%.4e\n"
              "  dW  max|diff|=%.4e peak=%.4e tol=%.4e\n%s\n",
              h, w, m, k, k, n, (unsigned long long)page_kb, blocks, slots,
              (unsigned long long)st.faults, (unsigned long long)st.evicts,
              (unsigned long long)st.get_errors, maxd, peak, tol,
              bmax, bpeak, btol, gmax, gpeak, gtol, ok ? "PASS" : "FAIL");
  eternia_lbann::Destroy(ctx);
  return ok ? 0 : 1;
}
