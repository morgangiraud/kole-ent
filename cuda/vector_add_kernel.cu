__global__ void vector_add(float *out, float *a, float *b, int N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        out[i] = a[i] + b[i];
    }
}
