NVFLAGS = -arch=compute_61 -code=sm_61 -O3
OUTDIR = objs

GOBJS = $(OUTDIR)/allocate.o $(OUTDIR)/sequence.o $(OUTDIR)/reduce_scan.o \
				$(OUTDIR)/mandel.o  $(OUTDIR)/filter.o


gpublob.o : $(OUTDIR) $(GOBJS)
	ld -r $(OUTDIR)/*.o -o gpublob.o
objs : 
	mkdir -p $(OUTDIR)
$(OUTDIR)/allocate.o : cudalib/allocate.cu
	nvcc -c -o $(OUTDIR)/allocate.o cudalib/allocate.cu $(NVFLAGS)
$(OUTDIR)/sequence.o : cudalib/naive_int_sequence.cu
	nvcc -c -o $(OUTDIR)/sequence.o cudalib/naive_int_sequence.cu $(NVFLAGS)
$(OUTDIR)/reduce_scan.o : cudalib/reduce_scan.cu
	nvcc -c -o $(OUTDIR)/reduce_scan.o cudalib/reduce_scan.cu $(NVFLAGS)
$(OUTDIR)/filter.o : cudalib/filter.cu
	nvcc -c -o $(OUTDIR)/filter.o cudalib/filter.cu $(NVFLAGS)
$(OUTDIR)/mandel.o : userkernels/mandel.cu
	nvcc -c -o $(OUTDIR)/mandel.o userkernels/mandel.cu $(NVFLAGS)

clean : 
	(rm *.o;rm -rf objs)
