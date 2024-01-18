import os
from torch.utils.cpp_extension import load_inline

fullpath = os.path.join(os.path.dirname(__file__), "cuda", "hello.cpp")

with open(fullpath, "r") as f:
    cpp_source = f.read()
    print(cpp_source)


my_module = load_inline(
    name="hello_module",
    cpp_sources=[cpp_source],
    functions=["hello_world"],
    verbose=True,
)

print(my_module.hello_world())
