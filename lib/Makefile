all:
	nvcc -c -o allocate.o cudalib/allocate.cu -arch=compute_61 -code=sm_61
	nvcc -c -o sequence.o cudalib/naive_int_sequence.cu -arch=compute_61 -code=sm_61
	nvcc -c -o reduce.o cudalib/reduce.cu -arch=compute_61 -code=sm_61
	nvcc -c -o mandel.o userkernels/mandel.cu -arch=compute_61 -code=sm_61
	nvcc -c -o kernels.o userkernels/built_in_kernels.cu -arch=compute_61 -code=sm_61
	ld -r allocate.o kernels.o sequence.o reduce.o mandel.o -o gpublob.o
	rm allocate.o kernels.o sequence.o mandel.o reduce.o

