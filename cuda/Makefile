NVCC        = /usr/local/cuda/bin/nvcc

default: vector_add

vector_add_main.o: vector_add_main.cu
	$(NVCC) -c -o $@ vector_add_main.cu

vector_add.o: vector_add_kernel.cu 
	$(NVCC) -c -o $@ vector_add_kernel.cu 

vector_add: vector_add.o vector_add_main.o
	$(NVCC) vector_add.o vector_add_main.o -o vector_add

clean:
	rm -f vector_add vector_add.o vector_add_main.o

.PHONY: clean vector_add