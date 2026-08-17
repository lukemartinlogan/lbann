////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#define LBANN_FULLY_CONNECTED_LAYER_INSTANTIATE
#include "lbann/layers/learning/fully_connected.hpp"

#include "eternia_gemm.h"
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <type_traits>

#include "lbann/optimizers/optimizer.hpp"
#include "lbann/weights/initializer.hpp"
#include "lbann/weights/variance_scaling_initializers.hpp"

#include "lbann/proto/datatype_helpers.hpp"

#include "lbann/proto/layers.pb.h"

#include <sstream>
#include <string>

namespace lbann {

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
fully_connected_layer<TensorDataType, T_layout, Dev>::fully_connected_layer(
  int output_size,
  bool transpose,
  WeightsType* weight,
  bool has_bias)
  : data_type_layer<TensorDataType>(nullptr),
    m_bias_gradient(nullptr),
    m_transpose(transpose)
{

  // Initialize output tensor dimensions
  this->set_output_dims({output_size});

  // Initialize bias
  m_bias_scaling_factor = (has_bias ? El::TypeTraits<TensorDataType>::One()
                                    : El::TypeTraits<TensorDataType>::Zero());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
fully_connected_layer<TensorDataType, T_layout, Dev>::fully_connected_layer()
  : fully_connected_layer(0, false, nullptr, false)
{}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
fully_connected_layer<TensorDataType, T_layout, Dev>::fully_connected_layer(
  const fully_connected_layer& other)
  : data_type_layer<TensorDataType>(other),
    m_bias_scaling_factor(other.m_bias_scaling_factor),
    m_transpose(other.m_transpose)
{

  // Deep matrix copies
  m_bias_gradient = other.m_bias_gradient;
  if (m_bias_gradient != nullptr) {
    m_bias_gradient = m_bias_gradient->Copy();
  }
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
auto fully_connected_layer<TensorDataType, T_layout, Dev>::operator=(
  const fully_connected_layer& other) -> fully_connected_layer&
{
  data_type_layer<TensorDataType>::operator=(other);
  m_bias_scaling_factor = other.m_bias_scaling_factor;
  m_transpose = other.m_transpose;

  // Deep matrix copies
  deallocate_matrices();
  m_bias_gradient = other.m_bias_gradient;
  if (m_bias_gradient != nullptr) {
    m_bias_gradient = m_bias_gradient->Copy();
  }

  return *this;
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
fully_connected_layer<TensorDataType, T_layout, Dev>::~fully_connected_layer()
{
  deallocate_matrices();
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
description
fully_connected_layer<TensorDataType, T_layout, Dev>::get_description() const
{
  auto desc = data_type_layer<TensorDataType>::get_description();
  const auto& bias_str =
    (m_bias_scaling_factor == El::TypeTraits<TensorDataType>::Zero()
       ? "disabled"
       : "enabled");
  desc.add("Bias", bias_str);
  return desc;
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void fully_connected_layer<TensorDataType, T_layout, Dev>::setup_data(
  size_t max_mini_batch_size)
{
  data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);

  // Initialize default weights if none are provided
  if (this->num_weights() > 2) {
    LBANN_ERROR("attempted to setup ",
                this->get_name(),
                " with an invalid number of weights");
  }
  if (m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    this->set_num_weights(2);
  }
  else {
    this->set_num_weights(1);
  }
  if (!this->has_weights(0)) {
    auto w = std::make_shared<WeightsType>(*this->get_comm());
    auto init = std::make_unique<he_initializer<TensorDataType>>(
      probability_distribution::gaussian);
    auto opt = this->m_model->template create_optimizer<TensorDataType>();
    w->set_name(this->get_name() + "_linearity_weights");
    w->set_initializer(std::move(init));
    w->set_optimizer(std::move(opt));
    this->set_weights(0, w);
    this->m_model->add_weights(std::move(w));
  }
  auto& linearity_weights = this->get_weights(0);

  // Initialize variance scaling initialization
  if (auto* initializer = linearity_weights.get_initializer()) {
    set_fan_in(*initializer, this->get_input_size());
    set_fan_out(*initializer, this->get_output_size());
  }

  // Input and output dimensions
  const auto& input_dims_ = this->get_input_dims();
  const auto& output_dims_ = this->get_output_dims();
  std::vector<size_t> input_dims(input_dims_.begin(), input_dims_.end());
  std::vector<size_t> output_dims(output_dims_.begin(), output_dims_.end());

  // Setup linearity weights
  auto linearity_dist = this->get_prev_activations().DistData();
  if (linearity_dist.colDist != El::MC || linearity_dist.rowDist != El::MR) {
    linearity_dist.colDist = El::STAR;
    linearity_dist.rowDist = El::STAR;
  }
  if (m_transpose) {
    linearity_weights.set_dims(input_dims, output_dims);
  }
  else {
    linearity_weights.set_dims(output_dims, input_dims);
  }
  // LBANN_ETERNIA_FC_HOST_WEIGHTS: keep the linearity OFF the GPU.
  //
  // Without this the paged GEMM does not lift anything: LBANN allocates the
  // weight matrix in VRAM through Hydrogen, and a layer whose weights exceed
  // VRAM fails in that allocator before the kernel is ever reached (measured:
  // 32768 x 65536 = 8.00 GiB, "out of memory, 8589934592 requested,
  // 8032092160 available"). Paging a copy of something already resident saves
  // nothing.
  //
  // DistData carries a device independently of the layer's, so the linearity
  // can live in host memory while the activations stay on the GPU. The paged
  // path already reads the weights through the host on its way to the CTE, so
  // this removes a GPU allocation rather than adding a transfer.
  //
  // Off by default: it makes the optimizer update a host matrix, which is
  // slower for layers that would have fit.
  if (std::getenv("LBANN_ETERNIA_FC_HOST_WEIGHTS") != nullptr) {
    linearity_dist.device = El::Device::CPU;
  }
  linearity_weights.set_matrix_distribution(linearity_dist);

  // Set up bias if needed.
  if (m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    if (!this->has_weights(1)) {
      auto w = std::make_shared<WeightsType>(*this->get_comm());
      auto opt = this->m_model->template create_optimizer<TensorDataType>();
      w->set_name(this->get_name() + "_bias_weights");
      w->set_optimizer(std::move(opt));
      this->set_weights(1, w);
      this->m_model->add_weights(std::move(w));
    }
    auto& bias_weights = this->get_weights(1);
    // Setup bias weights
    auto bias_dist = this->get_activations().DistData();
    bias_dist.rowDist = El::STAR;
    bias_weights.set_dims(output_dims);
    bias_weights.set_matrix_distribution(bias_dist);

    // Setup bias gradient
    if (Dev == El::Device::CPU) {
      if (T_layout == data_layout::MODEL_PARALLEL) {
        // Allocate a MCStarMat (RowSumMat)
        this->m_bias_gradient =
          new El::DistMatrix<TensorDataType,
                             El::MC,
                             El::STAR,
                             El::ELEMENT,
                             El::Device::CPU>(*bias_dist.grid);
      }
      else if (T_layout == data_layout::DATA_PARALLEL) {
        // Allocate a StarMat
        this->m_bias_gradient =
          new El::DistMatrix<TensorDataType,
                             El::STAR,
                             El::STAR,
                             El::ELEMENT,
                             El::Device::CPU>(*bias_dist.grid);
      }
    }
    if (this->m_bias_gradient != nullptr) {
      El::Zeros(*this->m_bias_gradient,
                bias_weights.get_matrix_height(),
                bias_weights.get_matrix_width());
    }
  }

  // Initialize freeze state
  auto const num_weights = this->num_weights();
  for (size_t ii = 0; ii < num_weights; ++ii) {
    auto& w = this->get_weights(ii);
    if (this->m_frozen) {
      w.freeze();
    }
    else {
      w.unfreeze();
    }
  }
  for (size_t ii = 0; ii < num_weights; ++ii) {
    auto& w = this->get_weights(ii);
    if (w.is_frozen() != this->is_frozen()) {
      LBANN_ERROR((this->is_frozen() ? "" : "un"),
                  "frozen ",
                  "layer \"",
                  this->get_name(),
                  "\" has ",
                  (w.is_frozen() ? "" : "un"),
                  "frozen ",
                  "weights \"",
                  w.get_name(),
                  "\"");
    }
  }
}

/** CPU implementation of forward prop computation. */
template <typename TensorDataType>
void fp_compute_impl(fully_connected_layer<TensorDataType,
                                           data_layout::MODEL_PARALLEL,
                                           El::Device::CPU>& l)
{

  // Matrices
  const auto& input = l.get_prev_activations();
  auto& output = l.get_activations();

  // Apply linearity
  // Note: Perform GEMMs independently if possible
  const auto& linearity = l.weights_values(0);
  if (!linearity.Participating()) {
    return;
  }
  if (linearity.DistSize() == 1) {
    El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(),
             linearity.LockedMatrix(),
             input.LockedMatrix(),
             El::TypeTraits<TensorDataType>::Zero(),
             output.Matrix());
  }
  else {
    El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(),
             linearity,
             input,
             El::TypeTraits<TensorDataType>::Zero(),
             output);
  }

  // Apply bias if needed
  if (l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    const auto& local_bias = l.weights_values(1).LockedMatrix();
    auto& local_output = output.Matrix();
    El::IndexDependentMap(
      local_output,
      (std::function<TensorDataType(
         El::Int,
         El::Int,
         const TensorDataType&)>)([&l, &local_bias](
                                    El::Int r,
                                    El::Int c,
                                    const TensorDataType& z) -> TensorDataType {
        return z + l.m_bias_scaling_factor * local_bias(r, 0);
      }));
  }
}

/** CPU implementation of backward prop computation. */
template <typename TensorDataType>
void bp_compute_impl(fully_connected_layer<TensorDataType,
                                           data_layout::MODEL_PARALLEL,
                                           El::Device::CPU>& l)
{

  // Matrices
  const auto& linearity = l.weights_values(0);
  const auto& input = l.get_prev_activations();
  const auto& gradient_wrt_output = l.get_prev_error_signals();
  auto& gradient_wrt_input = l.get_error_signals();
  const auto& local_linearity = linearity.LockedMatrix();
  const auto& local_input = input.LockedMatrix();
  const auto& local_gradient_wrt_output = gradient_wrt_output.LockedMatrix();
  auto& local_gradient_wrt_input = gradient_wrt_input.Matrix();
  if (!linearity.Participating()) {
    return;
  }

  // Compute gradient w.r.t. bias if needed
  if (l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    auto* bias_optimizer = l.get_weights(1).get_optimizer();
    if (bias_optimizer != nullptr) {
      El::RowSum(local_gradient_wrt_output, l.m_bias_gradient->Matrix());
      bias_optimizer->add_to_gradient(*l.m_bias_gradient,
                                      l.m_bias_scaling_factor,
                                      true);
    }
  }

  // Compute gradient w.r.t. linearity if needed
  // Note: Perform GEMMs independently if possible
  auto* linearity_optimizer = l.get_weights(0).get_optimizer();
  if (linearity_optimizer != nullptr) {
    TensorDataType dst_scale = El::TypeTraits<TensorDataType>::Zero(),
                   gradient_scale = El::TypeTraits<TensorDataType>::One();
    if (linearity.DistSize() == 1) {
      auto& linearity_gradient =
        linearity_optimizer->get_gradient_buffer(dst_scale,
                                                 gradient_scale,
                                                 true);
      if (l.m_transpose) {
        El::Gemm(El::NORMAL,
                 El::TRANSPOSE,
                 gradient_scale,
                 local_input,
                 local_gradient_wrt_output,
                 dst_scale,
                 linearity_gradient.Matrix());
      }
      else {
        El::Gemm(El::NORMAL,
                 El::TRANSPOSE,
                 gradient_scale,
                 local_gradient_wrt_output,
                 local_input,
                 dst_scale,
                 linearity_gradient.Matrix());
      }
    }
    else {
      auto& linearity_gradient =
        linearity_optimizer->get_gradient_buffer(dst_scale, gradient_scale);
      if (l.m_transpose) {
        El::Gemm(El::NORMAL,
                 El::TRANSPOSE,
                 gradient_scale,
                 input,
                 gradient_wrt_output,
                 dst_scale,
                 linearity_gradient);
      }
      else {
        El::Gemm(El::NORMAL,
                 El::TRANSPOSE,
                 gradient_scale,
                 gradient_wrt_output,
                 input,
                 dst_scale,
                 linearity_gradient);
      }
    }
  }

  // Compute gradient w.r.t. input
  // Note: Perform GEMMs independently if possible
  if (linearity.DistSize() == 1) {
    El::Gemm(l.m_transpose ? El::NORMAL : El::TRANSPOSE,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(),
             local_linearity,
             local_gradient_wrt_output,
             El::TypeTraits<TensorDataType>::Zero(),
             local_gradient_wrt_input);
  }
  else {
    El::Gemm(l.m_transpose ? El::NORMAL : El::TRANSPOSE,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(),
             linearity,
             gradient_wrt_output,
             El::TypeTraits<TensorDataType>::Zero(),
             gradient_wrt_input);
  }
}

/** CPU implementation of forward prop computation. */
template <typename TensorDataType>
void fp_compute_impl(fully_connected_layer<TensorDataType,
                                           data_layout::DATA_PARALLEL,
                                           El::Device::CPU>& l)
{

  // Matrices
  const auto& local_input = l.get_local_prev_activations();
  auto& local_output = l.get_local_activations();

  // Apply linearity
  const auto& local_linearity = l.weights_values(0).LockedMatrix();
  El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
           El::NORMAL,
           El::TypeTraits<TensorDataType>::One(),
           local_linearity,
           local_input,
           El::TypeTraits<TensorDataType>::Zero(),
           local_output);

  // Apply bias if needed
  if (l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    const auto& local_bias = l.weights_values(1).LockedMatrix();
    El::IndexDependentMap(
      local_output,
      (std::function<TensorDataType(
         El::Int,
         El::Int,
         const TensorDataType&)>)([&l, &local_bias](
                                    El::Int r,
                                    El::Int c,
                                    const TensorDataType& z) -> TensorDataType {
        return z + l.m_bias_scaling_factor * local_bias(r, 0);
      }));
  }
}

/** CPU implementation of backward prop computation. */
template <typename TensorDataType>
void bp_compute_impl(fully_connected_layer<TensorDataType,
                                           data_layout::DATA_PARALLEL,
                                           El::Device::CPU>& l)
{

  // Matrices
  const auto& local_linearity = l.weights_values(0).LockedMatrix();
  const auto& local_input = l.get_local_prev_activations();
  const auto& local_gradient_wrt_output = l.get_local_prev_error_signals();
  auto& local_gradient_wrt_input = l.get_local_error_signals();

  // Compute gradient w.r.t. bias if needed
  if (l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    auto* bias_optimizer = l.get_weights(1).get_optimizer();
    if (bias_optimizer != nullptr) {
      El::RowSum(local_gradient_wrt_output, l.m_bias_gradient->Matrix());
      bias_optimizer->add_to_gradient(*l.m_bias_gradient,
                                      l.m_bias_scaling_factor,
                                      true);
    }
  }

  // Compute gradient w.r.t. linearity if needed
  auto* linearity_optimizer = l.get_weights(0).get_optimizer();
  if (linearity_optimizer != nullptr) {
    TensorDataType dst_scale = El::TypeTraits<TensorDataType>::Zero(),
                   gradient_scale = El::TypeTraits<TensorDataType>::Zero();
    auto& linearity_gradient =
      linearity_optimizer->get_gradient_buffer(dst_scale, gradient_scale, true);
    if (l.m_transpose) {
      El::Gemm(El::NORMAL,
               El::TRANSPOSE,
               gradient_scale,
               local_input,
               local_gradient_wrt_output,
               dst_scale,
               linearity_gradient.Matrix());
    }
    else {
      El::Gemm(El::NORMAL,
               El::TRANSPOSE,
               gradient_scale,
               local_gradient_wrt_output,
               local_input,
               dst_scale,
               linearity_gradient.Matrix());
    }
  }

  // Compute gradient w.r.t. input
  El::Gemm(l.m_transpose ? El::NORMAL : El::TRANSPOSE,
           El::NORMAL,
           El::TypeTraits<TensorDataType>::One(),
           local_linearity,
           local_gradient_wrt_output,
           El::TypeTraits<TensorDataType>::Zero(),
           local_gradient_wrt_input);
}

#ifdef LBANN_HAS_GPU
/** GPU implementation of forward prop computation. */
template <typename TensorDataType>
void fp_compute_impl(fully_connected_layer<TensorDataType,
                                           data_layout::DATA_PARALLEL,
                                           El::Device::GPU>& l)
{

  // Matrices
  const auto& linearity = l.weights_values(0);
  const auto& local_input = l.get_local_prev_activations();
  auto& local_output = l.get_local_activations();

  if (!linearity.Participating()) {
    return;
  }

  // Apply linearity
  const auto& local_linearity = linearity.LockedMatrix();
  El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
           El::NORMAL,
           El::TypeTraits<TensorDataType>::One(),
           local_linearity,
           local_input,
           El::TypeTraits<TensorDataType>::Zero(),
           local_output);

  // Apply bias if needed
  if (l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    const auto& local_bias = l.weights_values(1).LockedMatrix();
    El::Matrix<TensorDataType, El::Device::GPU> ones;
#ifdef HYDROGEN_HAVE_CUB
    ones.SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif                     // HYDROGEN_HAVE_CUB
    ones.Resize(local_input.Width(), 1);
    El::Fill(ones, El::TypeTraits<TensorDataType>::One());
    El::Gemm(El::NORMAL,
             El::TRANSPOSE,
             l.m_bias_scaling_factor,
             local_bias,
             ones,
             El::TypeTraits<TensorDataType>::One(),
             local_output);
  }
}

/** GPU implementation of backward prop computation. */
template <typename TensorDataType>
void bp_compute_impl(fully_connected_layer<TensorDataType,
                                           data_layout::DATA_PARALLEL,
                                           El::Device::GPU>& l)
{

  // Matrices
  const auto& linearity = l.weights_values(0);
  const auto& local_linearity = linearity.LockedMatrix();
  const auto& local_input = l.get_local_prev_activations();
  const auto& local_gradient_wrt_output = l.get_local_prev_error_signals();
  auto& local_gradient_wrt_input = l.get_local_error_signals();
  if (!linearity.Participating()) {
    return;
  }

  // Compute gradient w.r.t. bias if needed
  if (l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    auto* bias_optimizer = l.get_weights(1).get_optimizer();
    if (bias_optimizer != nullptr) {
      TensorDataType dst_scale = El::TypeTraits<TensorDataType>::Zero(),
                     gradient_scale = El::TypeTraits<TensorDataType>::Zero();
      auto& bias_gradient =
        bias_optimizer->get_gradient_buffer(dst_scale, gradient_scale, true);
      if (local_gradient_wrt_output.Height() < 1 ||
          local_gradient_wrt_output.Width() < 1) {
        El::Scale(dst_scale, bias_gradient);
      }
      else {
        El::Matrix<TensorDataType, El::Device::GPU> ones;
#ifdef HYDROGEN_HAVE_CUB
        ones.SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif                         // HYDROGEN_HAVE_CUB
        ones.Resize(local_gradient_wrt_output.Width(), 1);
        El::Fill(ones, El::TypeTraits<TensorDataType>::One());
        El::Gemv(El::NORMAL,
                 gradient_scale,
                 local_gradient_wrt_output,
                 ones,
                 dst_scale,
                 bias_gradient.Matrix());
      }
    }
  }

  // Compute gradient w.r.t. linearity if needed
  auto* linearity_optimizer = l.get_weights(0).get_optimizer();
  if (linearity_optimizer != nullptr) {
    TensorDataType dst_scale = El::TypeTraits<TensorDataType>::Zero(),
                   gradient_scale = El::TypeTraits<TensorDataType>::Zero();
    auto& linearity_gradient =
      linearity_optimizer->get_gradient_buffer(dst_scale, gradient_scale, true);
    if (l.m_transpose) {
      El::Gemm(El::NORMAL,
               El::TRANSPOSE,
               gradient_scale,
               local_input,
               local_gradient_wrt_output,
               dst_scale,
               linearity_gradient.Matrix());
    }
    else {
      El::Gemm(El::NORMAL,
               El::TRANSPOSE,
               gradient_scale,
               local_gradient_wrt_output,
               local_input,
               dst_scale,
               linearity_gradient.Matrix());
    }
  }

  // Compute gradient w.r.t. input
  El::Gemm(l.m_transpose ? El::NORMAL : El::TRANSPOSE,
           El::NORMAL,
           El::TypeTraits<TensorDataType>::One(),
           local_linearity,
           local_gradient_wrt_output,
           El::TypeTraits<TensorDataType>::Zero(),
           local_gradient_wrt_input);
}

template <typename TensorDataType>
void fp_compute_impl(fully_connected_layer<TensorDataType,
                                           data_layout::MODEL_PARALLEL,
                                           El::Device::GPU>& l)
{

  // Matrices
  const auto& input = l.get_prev_activations();
  auto& output = l.get_activations();

  // Apply linearity
  // Note: Perform GEMMs independently if possible
  const auto& linearity = l.weights_values(0);
  if (!linearity.Participating()) {
    return;
  }
  if (linearity.DistSize() == 1) {
    El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(),
             linearity.LockedMatrix(),
             input.LockedMatrix(),
             El::TypeTraits<TensorDataType>::Zero(),
             output.Matrix());
  }
  else {
    El::Gemm(l.m_transpose ? El::TRANSPOSE : El::NORMAL,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(),
             linearity,
             input,
             El::TypeTraits<TensorDataType>::Zero(),
             output);
  }

  // Apply bias if needed
  // Note: local outer product is sufficient, no need for global GEMM
  if (l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    const auto& bias = l.weights_values(1);
    El::Matrix<TensorDataType, El::Device::GPU> ones;
#ifdef HYDROGEN_HAVE_CUB
    ones.SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif                     // HYDROGEN_HAVE_CUB
    ones.Resize(input.LocalWidth(), 1);
    El::Fill(ones, El::TypeTraits<TensorDataType>::One());
    El::Gemm(El::NORMAL,
             El::TRANSPOSE,
             l.m_bias_scaling_factor,
             bias.LockedMatrix(),
             ones,
             El::TypeTraits<TensorDataType>::One(),
             output.Matrix());
  }
}

template <typename TensorDataType>
void bp_compute_impl(fully_connected_layer<TensorDataType,
                                           data_layout::MODEL_PARALLEL,
                                           El::Device::GPU>& l)
{

  // Matrices
  const auto& linearity = l.weights_values(0);
  const auto& input = l.get_prev_activations();
  const auto& gradient_wrt_output = l.get_prev_error_signals();
  auto& gradient_wrt_input = l.get_error_signals();
  const auto& local_linearity = linearity.LockedMatrix();
  const auto& local_input = input.LockedMatrix();
  const auto& local_gradient_wrt_output = gradient_wrt_output.LockedMatrix();
  auto& local_gradient_wrt_input = gradient_wrt_input.Matrix();
  if (!linearity.Participating()) {
    return;
  }

  // Compute gradient w.r.t. bias if needed
  // Note: local GEMV is sufficient, no need for global row sum
  if (l.m_bias_scaling_factor != El::TypeTraits<TensorDataType>::Zero()) {
    auto* bias_optimizer = l.get_weights(1).get_optimizer();
    if (bias_optimizer != nullptr) {
      TensorDataType dst_scale = El::TypeTraits<TensorDataType>::Zero(),
                     gradient_scale = El::TypeTraits<TensorDataType>::Zero();
      auto& bias_gradient =
        bias_optimizer->get_gradient_buffer(dst_scale, gradient_scale, true);
      if (local_gradient_wrt_output.Height() < 1 ||
          local_gradient_wrt_output.Width() < 1) {
        El::Scale(dst_scale, bias_gradient);
      }
      else {
        El::Matrix<TensorDataType, El::Device::GPU> ones;
#ifdef HYDROGEN_HAVE_CUB
        ones.SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif                         // HYDROGEN_HAVE_CUB
        ones.Resize(local_gradient_wrt_output.Width(), 1);
        El::Fill(ones, El::TypeTraits<TensorDataType>::One());
        El::Gemv(El::NORMAL,
                 gradient_scale,
                 local_gradient_wrt_output,
                 ones,
                 dst_scale,
                 bias_gradient.Matrix());
      }
    }
  }

  // Compute gradient w.r.t. linearity if needed
  // Note: Perform GEMMs independently if possible
  auto* linearity_optimizer = l.get_weights(0).get_optimizer();
  if (linearity_optimizer != nullptr) {
    TensorDataType dst_scale = El::TypeTraits<TensorDataType>::Zero(),
                   gradient_scale = El::TypeTraits<TensorDataType>::Zero();
    if (linearity.DistSize() == 1) {
      auto& linearity_gradient =
        linearity_optimizer->get_gradient_buffer(dst_scale,
                                                 gradient_scale,
                                                 true);
      if (l.m_transpose) {
        El::Gemm(El::NORMAL,
                 El::TRANSPOSE,
                 gradient_scale,
                 local_input,
                 local_gradient_wrt_output,
                 dst_scale,
                 linearity_gradient.Matrix());
      }
      else {
        El::Gemm(El::NORMAL,
                 El::TRANSPOSE,
                 gradient_scale,
                 local_gradient_wrt_output,
                 local_input,
                 dst_scale,
                 linearity_gradient.Matrix());
      }
    }
    else {
      auto& linearity_gradient =
        linearity_optimizer->get_gradient_buffer(dst_scale, gradient_scale);
      if (l.m_transpose) {
        El::Gemm(El::NORMAL,
                 El::TRANSPOSE,
                 gradient_scale,
                 input,
                 gradient_wrt_output,
                 dst_scale,
                 linearity_gradient);
      }
      else {
        El::Gemm(El::NORMAL,
                 El::TRANSPOSE,
                 gradient_scale,
                 gradient_wrt_output,
                 input,
                 dst_scale,
                 linearity_gradient);
      }
    }
  }

  // Compute gradient w.r.t. input
  // Note: Perform GEMMs independently if possible
  if (linearity.DistSize() == 1) {
    El::Gemm(l.m_transpose ? El::NORMAL : El::TRANSPOSE,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(),
             local_linearity,
             local_gradient_wrt_output,
             El::TypeTraits<TensorDataType>::Zero(),
             local_gradient_wrt_input);
  }
  else {
    El::Gemm(l.m_transpose ? El::NORMAL : El::TRANSPOSE,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(),
             linearity,
             gradient_wrt_output,
             El::TypeTraits<TensorDataType>::Zero(),
             gradient_wrt_input);
  }
}

#endif // LBANN_HAS_GPU

template <typename T, data_layout L, El::Device D>
void fully_connected_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_fully_connected();
  msg->set_num_neurons(get_linear_size(this->get_output_dims()));
  auto const has_bias = (this->num_weights() > 1UL);
  msg->set_has_bias(has_bias);
  msg->set_transpose(m_transpose);
}

namespace {

/**
 * Create the paged context if needed and push the current weights into it.
 *
 * Shared by the forward and backward hooks because both need exactly this and
 * because they must agree on the tag: a second context for the same layer
 * would page a second, independent copy of W.
 *
 * The weights are re-uploaded on every call. Nothing invalidates resident
 * pages when the host rewrites the backing store -- the same property the
 * LAMMPS integration has to drop its caches for each step.
 */
bool eternia_fc_ready(eternia_lbann::Context*& ctx, int& cached_h,
                      int& cached_w, const void* owner, const float* w_host,
                      int w_ldim, int h, int w)
{
  if (!eternia_lbann::Available()) return false;

  if (ctx != nullptr && (cached_h != h || cached_w != w)) {
    eternia_lbann::Destroy(ctx);
    ctx = nullptr;
  }
  if (ctx == nullptr) {
    eternia_lbann::Config cfg;
    static std::string tag;
    tag = "lbann_eternia_fc_" +
          std::to_string(reinterpret_cast<std::uintptr_t>(owner));
    cfg.tag = tag.c_str();
    cfg.stats = (std::getenv("LBANN_ETERNIA_STATS") != nullptr);
    ctx = eternia_lbann::Create(cfg, h, w);
    if (ctx == nullptr) {
      LBANN_WARNING("eternia: ", eternia_lbann::LastError(), "; using El::Gemm");
      return false;
    }
    cached_h = h;
    cached_w = w;
  }
  if (!eternia_lbann::UploadWeights(ctx, w_host, w_ldim)) {
    LBANN_WARNING("eternia: ", eternia_lbann::LastError(), "; using El::Gemm");
    return false;
  }
  return true;
}

/**
 * Eternia paged forward GEMM, behind LBANN_ETERNIA_FC.
 *
 * Replaces C = W^T * X with a kernel that holds W out of core and pages it
 * into the GPU on demand, so a layer can be wider than GPU memory.
 *
 * Takes primitives rather than the layer: the caller is a member function and
 * already has access to the private state, and reaching into it from a free
 * function would need a friend declaration for something that is an
 * implementation detail.
 *
 * @param w_host   W's column-major buffer copied to the host, and its ldim
 * @param x_dev    X (k x n) column-major on the device, and its ldim
 * @param c_dev    C (m x n) column-major on the device, and its ldim
 */
bool eternia_fc_gemm(eternia_lbann::Context*& ctx, int& cached_h, int& cached_w,
                     const void* owner, const float* w_host, int w_ldim, int h,
                     int w, const float* x_dev, int ldx, int n, float* c_dev,
                     int ldc)
{
  if (!eternia_fc_ready(ctx, cached_h, cached_w, owner, w_host, w_ldim, h, w)) {
    return false;
  }
  if (!eternia_lbann::Forward(ctx, x_dev, ldx, n, c_dev, ldc)) {
    LBANN_WARNING("eternia: ", eternia_lbann::LastError(), "; using El::Gemm");
    return false;
  }
  if (std::getenv("LBANN_ETERNIA_STATS") != nullptr) {
    const auto st = eternia_lbann::GetStats(ctx);
    std::cerr << "[eternia] faults=" << st.faults << " evicts=" << st.evicts
              << " get_errors=" << st.get_errors << std::endl;
  }
  return true;
}


/**
 * Eternia paged backward pass: the gradient w.r.t. the input, and the gradient
 * w.r.t. the weights.
 *
 * This is what the forward hook alone could not deliver. Keeping W off the GPU
 * is not enough on its own, because backpropagation multiplies by W too, and
 * with the weights in host memory El::Gemm refuses outright:
 *
 *     "Must call gemm with matrices on same device"
 *
 * so LBANN_ETERNIA_FC_HOST_WEIGHTS could reach the end of a forward pass and
 * no further.
 *
 * @param dw_host  receives dL/dW as Hydrogen lays out the linearity --
 *                 column-major (h x w) -- for the caller to fold into the
 *                 optimizer's gradient buffer with its own scales.
 *
 * WHAT THIS DOES NOT DO
 * ---------------------
 * It does not reduce the memory LBANN itself holds. The optimizer allocates a
 * full-size resident gradient buffer for the linearity regardless of what this
 * layer does, so the gradient makes a round trip through host memory here
 * instead of staying paged. The paged SgdUpdate kernel is what avoids that,
 * and it does so by bypassing LBANN's optimizer rather than feeding it.
 */
bool eternia_fc_bp(eternia_lbann::Context*& ctx, int& cached_h, int& cached_w,
                   const void* owner, const float* w_host, int w_ldim, int h,
                   int w, const float* dc_dev, int ldc, int n,
                   const float* x_dev, int ldx, float* dx_dev, int ldx_out,
                   float* dw_host, int dw_ldim)
{
  if (!eternia_fc_ready(ctx, cached_h, cached_w, owner, w_host, w_ldim, h, w)) {
    return false;
  }
  if (!eternia_lbann::BackwardInput(ctx, dc_dev, ldc, n, dx_dev, ldx_out)) {
    LBANN_WARNING("eternia: ", eternia_lbann::LastError(), "; using El::Gemm");
    return false;
  }
  if (dw_host != nullptr) {
    if (!eternia_lbann::WeightGradient(ctx, dc_dev, ldc, n, x_dev, ldx)) {
      LBANN_WARNING("eternia: ", eternia_lbann::LastError(), "; using El::Gemm");
      return false;
    }
    if (!eternia_lbann::ReadWeightGradient(ctx, dw_host, dw_ldim)) {
      LBANN_WARNING("eternia: ", eternia_lbann::LastError(), "; using El::Gemm");
      return false;
    }
  }
  return true;
}

}  // namespace

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void fully_connected_layer<TensorDataType, T_layout, Dev>::fp_compute()
{
  // The paged path applies only when every one of these holds; anything else
  // falls through to El::Gemm.
  //   - LBANN_ETERNIA_FC is set
  //   - float32 on the GPU
  //   - m_transpose, so the GEMM is W^T * X and Hydrogen's column-major
  //     buffer already IS the row-major matrix the kernel reads
  //   - no bias term, which the paged path does not apply
  //   - the matrices are local
  static const bool want = (std::getenv("LBANN_ETERNIA_FC") != nullptr);
  if (want && Dev == El::Device::GPU &&
      std::is_same<TensorDataType, float>::value && this->m_transpose &&
      this->m_bias_scaling_factor == El::TypeTraits<TensorDataType>::Zero()) {
    const auto& linearity = this->weights_values(0);
    if (linearity.Participating() && linearity.DistSize() == 1) {
      const auto& lin = linearity.LockedMatrix();
      const auto& in = this->get_local_prev_activations();
      auto& out = this->get_local_activations();
      // The weights may be on either device: El::Copy below moves them to the
      // host either way, and with LBANN_ETERNIA_FC_HOST_WEIGHTS they are
      // already there. The ACTIVATIONS must be on the GPU -- the kernel reads
      // and writes them in place.
      if (in.Width() > 0 && out.Width() > 0) {
        // The uploader reads host memory; Hydrogen keeps the weights on GPU.
        El::Matrix<TensorDataType, El::Device::CPU> host_lin;
        El::Copy(lin, host_lin);
        // LBANN_ETERNIA_CHECK: compute the SAME product with El::Gemm and
        // report the largest elementwise difference. Inferring correctness
        // from an objective several operations downstream is how a paging bug
        // gets read off an energy total; this measures the thing itself.
        static const bool check = (std::getenv("LBANN_ETERNIA_CHECK") != nullptr);
        El::Matrix<TensorDataType, El::Device::GPU> ref;
        if (check) {
          El::Copy(out, ref);
          El::Gemm(El::TRANSPOSE,
                   El::NORMAL,
                   El::TypeTraits<TensorDataType>::One(),
                   lin,
                   in,
                   El::TypeTraits<TensorDataType>::Zero(),
                   ref);
        }
        if (eternia_fc_gemm(
              m_eternia_ctx, m_eternia_h, m_eternia_w, this,
              reinterpret_cast<const float*>(host_lin.LockedBuffer()),
              host_lin.LDim(), lin.Height(), lin.Width(),
              reinterpret_cast<const float*>(in.LockedBuffer()), in.LDim(),
              in.Width(), reinterpret_cast<float*>(out.Buffer()), out.LDim())) {
          if (check) {
            El::Matrix<TensorDataType, El::Device::CPU> a, b;
            El::Copy(out, a);
            El::Copy(ref, b);
            double maxd = 0.0, peak = 0.0;
            for (El::Int j = 0; j < a.Width(); ++j) {
              for (El::Int i = 0; i < a.Height(); ++i) {
                const double x = a(i, j), y = b(i, j);
                peak = std::max(peak, std::abs(y));
                maxd = std::max(maxd, std::abs(x - y));
              }
            }
            std::cerr << "[eternia-check] " << a.Height() << "x" << a.Width()
                      << " max|diff|=" << maxd << " peak=" << peak
                      << " rel=" << (peak > 0 ? maxd / peak : 0.0) << std::endl;
          }
          return;
        }
      }
    }
  }
  fp_compute_impl<TensorDataType>(*this);
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void fully_connected_layer<TensorDataType, T_layout, Dev>::bp_compute()
{
  // Gated identically to fp_compute -- the two must agree, because a step that
  // pages the forward GEMM and then falls back to El::Gemm for the backward
  // one is not a configuration anybody wants to reason about, and with the
  // weights on the host the fallback cannot run at all.
  static const bool want = (std::getenv("LBANN_ETERNIA_FC") != nullptr);
  if (want && Dev == El::Device::GPU &&
      std::is_same<TensorDataType, float>::value && this->m_transpose &&
      this->m_bias_scaling_factor == El::TypeTraits<TensorDataType>::Zero()) {
    const auto& linearity = this->weights_values(0);
    if (linearity.Participating() && linearity.DistSize() == 1) {
      const auto& lin = linearity.LockedMatrix();
      const auto& in = this->get_local_prev_activations();
      const auto& dout = this->get_local_prev_error_signals();
      auto& din = this->get_local_error_signals();
      if (in.Width() > 0 && dout.Width() > 0 && din.Width() > 0) {
        El::Matrix<TensorDataType, El::Device::CPU> host_lin;
        El::Copy(lin, host_lin);

        auto* opt = this->get_weights(0).get_optimizer();
        TensorDataType dst_scale = El::TypeTraits<TensorDataType>::Zero(),
                       gradient_scale = El::TypeTraits<TensorDataType>::Zero();
        El::Matrix<TensorDataType, El::Device::CPU> host_dw;
        if (opt != nullptr) {
          host_dw.Resize(lin.Height(), lin.Width());
        }

        // LBANN_ETERNIA_CHECK: the same two products via El::Gemm, compared
        // elementwise. The forward hook measures only the forward GEMM, and
        // the backward pass is two different kernels with two different
        // accumulation patterns, so it needs its own check rather than
        // inheriting confidence from the forward one.
        static const bool check = (std::getenv("LBANN_ETERNIA_CHECK") != nullptr);
        El::Matrix<TensorDataType, El::Device::GPU> ref_din, ref_dw;
        if (check) {
          ref_din.Resize(din.Height(), din.Width());
          El::Gemm(El::NORMAL,
                   El::NORMAL,
                   El::TypeTraits<TensorDataType>::One(),
                   lin,
                   dout,
                   El::TypeTraits<TensorDataType>::Zero(),
                   ref_din);
          ref_dw.Resize(lin.Height(), lin.Width());
          El::Gemm(El::NORMAL,
                   El::TRANSPOSE,
                   El::TypeTraits<TensorDataType>::One(),
                   in,
                   dout,
                   El::TypeTraits<TensorDataType>::Zero(),
                   ref_dw);
        }

        if (eternia_fc_bp(
              m_eternia_ctx, m_eternia_h, m_eternia_w, this,
              reinterpret_cast<const float*>(host_lin.LockedBuffer()),
              host_lin.LDim(), lin.Height(), lin.Width(),
              reinterpret_cast<const float*>(dout.LockedBuffer()), dout.LDim(),
              dout.Width(),
              reinterpret_cast<const float*>(in.LockedBuffer()), in.LDim(),
              reinterpret_cast<float*>(din.Buffer()), din.LDim(),
              opt != nullptr ? reinterpret_cast<float*>(host_dw.Buffer())
                             : nullptr,
              opt != nullptr ? host_dw.LDim() : 0)) {
          if (opt != nullptr) {
            // The optimizer owns a resident full-size gradient buffer and its
            // own accumulation scales, so the paged gradient is folded in
            // rather than handed over.
            //
            // The buffer's device follows the WEIGHTS, not the activations:
            // under LBANN_ETERNIA_FC_HOST_WEIGHTS the linearity is on the
            // host, so this buffer is a CPU matrix even though everything
            // else in this function is on the GPU. Branching on it is the
            // whole point of the option -- assuming GPU here is what made the
            // host-weights path abort with "Axpy: Incompatible devices!".
            auto& gbuf =
              opt->get_gradient_buffer(dst_scale, gradient_scale, true);
            El::Scale(dst_scale, gbuf.Matrix());
            if (gbuf.Matrix().GetDevice() == El::Device::CPU) {
              El::Axpy(
                gradient_scale,
                host_dw,
                static_cast<El::Matrix<TensorDataType, El::Device::CPU>&>(
                  gbuf.Matrix()));
            }
            else {
              El::Matrix<TensorDataType, El::Device::GPU> dev_dw;
              El::Copy(host_dw, dev_dw);
              El::Axpy(
                gradient_scale,
                dev_dw,
                static_cast<El::Matrix<TensorDataType, El::Device::GPU>&>(
                  gbuf.Matrix()));
            }
          }
          if (check) {
            auto report = [](const char* what,
                             const El::Matrix<TensorDataType, El::Device::GPU>& g,
                             const El::Matrix<TensorDataType, El::Device::GPU>& r) {
              El::Matrix<TensorDataType, El::Device::CPU> a, b;
              El::Copy(g, a);
              El::Copy(r, b);
              double maxd = 0.0, peak = 0.0;
              for (El::Int j = 0; j < a.Width(); ++j) {
                for (El::Int i = 0; i < a.Height(); ++i) {
                  const double x = a(i, j), y = b(i, j);
                  peak = std::max(peak, std::abs(y));
                  maxd = std::max(maxd, std::abs(x - y));
                }
              }
              std::cerr << "[eternia-check] " << what << " " << a.Height() << "x"
                        << a.Width() << " max|diff|=" << maxd
                        << " peak=" << peak
                        << " rel=" << (peak > 0 ? maxd / peak : 0.0)
                        << std::endl;
            };
            report("bp-input", din, ref_din);
            if (opt != nullptr) {
              El::Matrix<TensorDataType, El::Device::GPU> got_dw;
              El::Copy(host_dw, got_dw);
              report("bp-weights", got_dw, ref_dw);
            }
          }
          return;
        }
      }
    }
  }
  bp_compute_impl<TensorDataType>(*this);
}

#ifdef LBANN_HAS_ONNX
template <typename T, data_layout L, El::Device D>
void fully_connected_layer<T, L, D>::fill_onnx_node(
  onnx::GraphProto& graph) const
{
  auto const& parent = this->get_parent_layer(0);
  auto const has_bias = (this->num_weights() > 1UL);

  // Setup the inputs.
  auto const layer_name = this->get_name();
  auto const A_name = layer_name + "_" + parent.get_name() + "_reshape";
  auto const B_name = this->get_weights(0).get_name();
  auto const C_name = (has_bias ? layer_name + "_bias_reshape" : std::string{});

  // Flatten the input tensor (A, wrt GEMM).
  {
    auto* flatten_input_shape = graph.add_initializer();
    flatten_input_shape->set_name(layer_name + "_" + parent.get_name() +
                                  "_shape");
    flatten_input_shape->set_data_type(onnx::TensorProto::INT64);
    flatten_input_shape->add_dims(2);
    flatten_input_shape->add_int64_data(0);
    flatten_input_shape->add_int64_data(-1);
    flatten_input_shape->set_doc_string(
      "Shape for " + layer_name + " reshape " + parent.get_name() + " node");

    onnx::NodeProto* flatten_input = graph.add_node();
    flatten_input->add_input(
      parent.get_name() + "_" +
      std::to_string(parent.find_child_layer_index(*this)));
    flatten_input->add_input(flatten_input_shape->name());
    flatten_input->add_output(A_name);
    flatten_input->set_name(A_name);
    flatten_input->set_op_type("Reshape");
    flatten_input->set_domain("");
    flatten_input->set_doc_string("Reshape " + parent.get_name() +
                                  " for Fully Connected Layer");
  }

  // Setup the bias node, if applicable.
  if (has_bias) {
    // bias = Reshape(data=bias, shape=[1,-1])
    auto* shape = graph.add_initializer();
    shape->set_name(this->get_name() + "_bias_shape");
    shape->set_data_type(onnx::TensorProto::INT64);
    shape->add_dims(2);
    shape->add_int64_data(1);
    shape->add_int64_data(-1);
    shape->set_doc_string("Shape for " + layer_name + " Bias");

    auto* bias = graph.add_node();
    bias->add_input(this->get_weights(1).get_name());
    bias->add_input(shape->name());
    bias->add_output(this->get_name() + "_bias_reshape");
    bias->set_name(this->get_name() + "_bias_reshape");
    bias->set_op_type("Reshape");
    bias->set_domain("");
    bias->set_doc_string("Reshape bias for Fully Connected Layer");
  }

  auto* gemm = graph.add_node();
  gemm->add_input(A_name);
  gemm->add_input(B_name);
  if (has_bias)
    gemm->add_input(C_name);

  gemm->add_output(layer_name + "_0");
  gemm->set_name(layer_name + "_0");
  gemm->set_op_type("Gemm");
  gemm->set_domain("");
  gemm->set_doc_string("Gemm node for Fully Connected Layer");

  {
    auto* alpha = gemm->add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(onnx::AttributeProto::FLOAT);
    alpha->set_f(1);
  }
  {
    auto* beta = gemm->add_attribute();
    beta->set_name("beta");
    beta->set_type(onnx::AttributeProto::FLOAT);
    beta->set_f(has_bias ? 1 : 0);
  }
  {
    auto* transA = gemm->add_attribute();
    transA->set_name("transA");
    transA->set_type(onnx::AttributeProto::INT);
    transA->set_i(m_transpose ? 1 : 0);
  }
  {
    auto* transB = gemm->add_attribute();
    transB->set_name("transB");
    transB->set_type(onnx::AttributeProto::INT);
    transB->set_i(1); // Should be 1 because ONNX will do x*W^T + b
  }
}
#endif // LBANN_HAS_ONNX

template <typename TensorDataType, data_layout layout, El::Device device>
std::unique_ptr<Layer>
build_fully_connected_layer_from_pbuf(lbann_comm* comm,
                                      lbann_data::Layer const& layer_msg)
{
  using LayerType = fully_connected_layer<TensorDataType, layout, device>;
  const auto& params = layer_msg.fully_connected();
  return std::make_unique<LayerType>(params.num_neurons(),
                                     params.transpose(),
                                     nullptr,
                                     params.has_bias());
}

#define PROTO_DEVICE(T, Device)                                                \
  template class fully_connected_layer<T, data_layout::DATA_PARALLEL, Device>; \
  template class fully_connected_layer<T,                                      \
                                       data_layout::MODEL_PARALLEL,            \
                                       Device>;                                \
  template std::unique_ptr<Layer>                                              \
  build_fully_connected_layer_from_pbuf<T,                                     \
                                        data_layout::DATA_PARALLEL,            \
                                        Device>(lbann_comm*,                   \
                                                lbann_data::Layer const&);     \
  template std::unique_ptr<Layer>                                              \
  build_fully_connected_layer_from_pbuf<T,                                     \
                                        data_layout::MODEL_PARALLEL,           \
                                        Device>(lbann_comm*,                   \
                                                lbann_data::Layer const&)

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
