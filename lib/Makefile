all:
	nvcc -c -o clib.o cudalib/*.cu
	nvcc -c -o kernels.o userkernels/*.cu
	nvcc -c -o sequence.o gpusequence/*.cu
	ld -r clib.o kernels.o sequence.o -o gpublob.o
	rm clib.o kernels.o sequence.o

