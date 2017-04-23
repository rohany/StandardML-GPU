NVFLAGS = -arch=compute_61 -code=sm_61 -O3
OUTDIR = objs

GOBJS = $(OUTDIR)/allocate.o $(OUTDIR)/int_sequence.o $(OUTDIR)/mandel.o


gpublob.o : $(OUTDIR) $(GOBJS)
	ld -r $(OUTDIR)/*.o -o gpublob.o
objs : 
	mkdir -p $(OUTDIR)
$(OUTDIR)/allocate.o : cudalib/allocate.cu
	nvcc -c -o $(OUTDIR)/allocate.o cudalib/allocate.cu $(NVFLAGS)
$(OUTDIR)/int_sequence.o : cudalib/int_sequence.cu
	nvcc -c -o $(OUTDIR)/sequence.o cudalib/int_sequence.cu $(NVFLAGS)
$(OUTDIR)/mandel.o : userkernels/mandel.cu
	nvcc -c -o $(OUTDIR)/mandel.o userkernels/mandel.cu $(NVFLAGS)

clean : 
	(rm *.o;rm -rf objs)
