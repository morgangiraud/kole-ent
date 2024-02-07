#include <cuda.h>
#include <stdio.h>

__global__ void vector_add(float *out, float *a, float *b, int N);

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            exit(code);
        }
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b)
{
    return 1 + (a - 1) / b;
}

void gpuVecAdd(float *a, float *b, float *out, int N)
{
    float *a_d, *b_d, *out_d;
    size_t size = N * sizeof(float);

    cudaMalloc((void **)&a_d, size);
    cudaMalloc((void **)&b_d, size);
    cudaMalloc((void **)&out_d, size);

    cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, size, cudaMemcpyHostToDevice);

    const unsigned int numThreads = 256;
    unsigned int numBlocks = cdiv(N, numThreads);

    vector_add<<<numBlocks, numThreads>>>(out_d, a_d, b_d, N);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaMemcpy(out, out_d, size, cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(out_d);
}

int main()
{
    const int N = 1000;
    float a[N];
    float b[N];
    float out[N];

    for (int i = 0; i < N; i++)
    {
        a[i] = float(i);
        b[i] = a[i] / 1000.0f;
    }

    gpuVecAdd(a, b, out, N);

    for (int i = 0; i < N; i++)
    {
        if (i > 0)
        {
            printf(", ");
            if (i % 10 == 0)
            {
                printf("\n");
            }
        }
        printf("%8.3f", out[i]);
    }
    printf("\n");
    return 0;
}