NVCC = nvcc --compiler-options="-Wall -Wextra -O3" -std=c++11 -arch=compute_62 -code=sm_62 -lcublas

default: UnifiedMemory ExplicitMemory

UnifiedMemory: Makefile UnifiedMemory.cu
	$(NVCC) -o UnifiedMemory UnifiedMemory.cu

ExplicitMemory: Makefile ExplicitMemory.cu
	$(NVCC) -o ExplicitMemory ExplicitMemory.cu

clean:
	rm -f UnifiedMemory ExplicitMemory
