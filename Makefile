NVCC     ?= nvcc
ARCH     ?= -gencode arch=compute_80,code=sm_80
CFLAGS   := -O3 $(ARCH)
LDFLAGS  := -lcublas

all: bmm_bench

bmm_bench: bmm.cu bmm_bench.cu bmm.h
	$(NVCC) $(CFLAGS) -o $@ bmm.cu bmm_bench.cu $(LDFLAGS)

clean:
	rm -f bmm_bench

.PHONY: all clean
