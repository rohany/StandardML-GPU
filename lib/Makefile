NVFLAGS = -arch=compute_61 -code=sm_61 -O3
OUTDIR = objs

GOBJS = $(OUTDIR)/allocate.o $(OUTDIR)/int_sequence.o $(OUTDIR)/mandel.o \
				$(OUTDIR)/float_sequence.o $(OUTDIR)/int_tuple_sequence.o


gpublob.o : $(OUTDIR) $(GOBJS)
	ld -r $(OUTDIR)/*.o -o gpublob.o
objs : 
	mkdir -p $(OUTDIR)
$(OUTDIR)/allocate.o : cudalib/allocate.cu
	nvcc -c -o $(OUTDIR)/allocate.o cudalib/allocate.cu $(NVFLAGS)
$(OUTDIR)/int_sequence.o : cudalib/int_sequence.cu
	nvcc -c -o $(OUTDIR)/int_sequence.o cudalib/int_sequence.cu $(NVFLAGS)
$(OUTDIR)/int_tuple_sequence.o : cudalib/int_tuple_sequence.cu
	nvcc -c -o $(OUTDIR)/int_tuple_sequence.o cudalib/int_tuple_sequence.cu $(NVFLAGS)
$(OUTDIR)/float_sequence.o : cudalib/float_sequence.cu
	nvcc -c -o $(OUTDIR)/float_sequence.o cudalib/float_sequence.cu $(NVFLAGS)
$(OUTDIR)/mandel.o : userkernels/mandel.cu
	nvcc -c -o $(OUTDIR)/mandel.o userkernels/mandel.cu $(NVFLAGS)

clean : 
	(rm *.o;rm -rf objs)
