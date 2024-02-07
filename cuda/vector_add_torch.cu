#include <torch/extension.h>
#include <stdio.h>

__global__ void vector_add(float *out, float *a, float *b, int N);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b)
{
    return 1 + (a - 1) / b;
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b, int N)
{
    CHECK_INPUT(a);
    CHECK_INPUT(b);

    auto out = torch::empty({N}, a.options());

    const unsigned int numThreads = 256;
    unsigned int numBlocks = cdiv(N, numThreads);

    vector_add<<<numBlocks, numThreads>>>(out.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), N);

    return out;
}