import os
import torch
from torch.utils.cpp_extension import load_inline

cpp_source_file = os.path.join(os.path.dirname(__file__), "vector_add_torch.h")
cuda_source_file1 = os.path.join(os.path.dirname(__file__), "vector_add_torch.cu")
cuda_source_file2 = os.path.join(os.path.dirname(__file__), "vector_add_kernel.cu")

with open(cpp_source_file, "r") as f:
    cpp_source = f.read()
    print(cpp_source)

with open(cuda_source_file1, "r") as f:
    cuda_source1 = f.read()
    print(cpp_source)

with open(cuda_source_file2, "r") as f:
    cuda_source2 = f.read()
    print(cpp_source)


my_module = load_inline(
    name="vector_add",
    cuda_sources=[cuda_source1, cuda_source2],
    cpp_sources=[cpp_source],
    functions=["vector_add"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)

N = 1000
a = torch.arange(0, N).type(torch.float32).contiguous().cuda()
b = (torch.arange(0, N) / N).type(torch.float32).contiguous().cuda()

out = my_module.vector_add(a, b, N).cpu()

print(out)
