#pragma once

#include <cuda_runtime_api.h>

#include "pearl_api_params.h"

void run_pearl_gemm_sm86(PearlAPIParams const& params, cudaStream_t stream);
void run_pearl_noisy_gemm_sm86(PearlAPIParams const& params, cudaStream_t stream);
void run_noise_generation_sm86(Noise_gen_params const& params,
                               cudaStream_t stream);
