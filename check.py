import pkg_resources

import triton
import triton.language as tl
import torch

torch_v = pkg_resources.get_distribution("torch").version
triton_v = pkg_resources.get_distribution("triton").version
print(f"torch: {torch_v}")
print(f"triton: {triton_v}")

print(f"is_available: {torch.cuda.is_available()}")
d_count = torch.cuda.device_count()
print(f"device_count(): {d_count}")
for idx in range(d_count):
  print(f"get_device_name({idx}): {torch.cuda.get_device_name(idx)}")
  print(f"get_device_properties({idx}): {torch.cuda.get_device_properties(idx)}")
  print(f"get_device_capability({idx}): {torch.cuda.get_device_capability(idx)}")
