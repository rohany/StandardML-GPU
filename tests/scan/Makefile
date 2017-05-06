MLTONFLAGS = -default-ann 'allowFFI true' -link-opt \
						 '-rdynamic -L/usr/local/depot/cuda-8.0/lib64/ -lcudart -lstdc++'
LIBPATH = ../../lib/
BLOB = $(LIBPATH)/gpublob.o

all: scan thrust

scan : scan.sml $(BLOB)
	mlton $(MLTONFLAGS) scan.mlb ../../lib/gpublob.o
thrust : thrust.cu
	nvcc -std=c++11 -o thrust thrust.cu
$(LIBPATH)/gpublob.o :
	(cd ../../lib/;make)

clean : 
	rm scan thrust
