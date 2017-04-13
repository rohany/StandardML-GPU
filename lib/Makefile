all:
	nvcc -c -o clib.o cudalib/*.cu
	nvcc -c -o kernels.o userkernels/*.cu
	ld -r clib.o kernels.o -o gpublob.o
	rm clib.o kernels.o

