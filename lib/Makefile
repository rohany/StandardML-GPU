NVFLAGS = -arch=compute_61 -code=sm_61
OUTDIR = objs

GOBJS = $(OUTDIR)/allocate.o $(OUTDIR)/sequence.o $(OUTDIR)/reduce.o \
				$(OUTDIR)/mandel.o  $(OUTDIR)/scan.o


gpublob.o : $(OUTDIR) $(GOBJS)
	ld -r $(OUTDIR)/*.o -o gpublob.o
objs : 
	mkdir -p $(OUTDIR)
$(OUTDIR)/allocate.o : cudalib/allocate.cu
	nvcc -c -o $(OUTDIR)/allocate.o cudalib/allocate.cu $(NVFLAGS)
$(OUTDIR)/sequence.o : cudalib/naive_int_sequence.cu
	nvcc -c -o $(OUTDIR)/sequence.o cudalib/naive_int_sequence.cu $(NVFLAGS)
$(OUTDIR)/reduce.o : cudalib/reduce.cu
	nvcc -c -o $(OUTDIR)/reduce.o cudalib/reduce.cu $(NVFLAGS)
$(OUTDIR)/scan.o : cudalib/scan.cu
	nvcc -c -o $(OUTDIR)/scan.o cudalib/scan.cu $(NVFLAGS)
$(OUTDIR)/mandel.o : userkernels/mandel.cu
	nvcc -c -o $(OUTDIR)/mandel.o userkernels/mandel.cu $(NVFLAGS)

clean : 
	(rm *.o;rm -rf objs)
