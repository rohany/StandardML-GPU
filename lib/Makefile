all:
	nvcc -c -o clib.o cudalib/*.cu
	nvcc -c -o mandel.o userkernels/mandel.cu
	nvcc -c -o kernels.o userkernels/built_in_kernels.cu
	nvcc -c -o sequence.o gpusequence/*.cu
	ld -r clib.o kernels.o sequence.o mandel.o -o gpublob.o
	rm clib.o kernels.o sequence.o mandel.o

