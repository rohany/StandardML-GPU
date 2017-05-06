MLTONFLAGS = -default-ann 'allowFFI true' -link-opt \
						 '-rdynamic -L/usr/local/depot/cuda-8.0/lib64/ -lcudart -lstdc++'
LIBPATH = ../../lib/
BLOB = $(LIBPATH)/gpublob.o

all : map reduce thrust

map : map.sml $(BLOB)
	mlton $(MLTONFLAGS) map.mlb ../../lib/gpublob.o
reduce : reduce.sml $(BLOB)
	mlton $(MLTONFLAGS) reduce.mlb ../../lib/gpublob.o
thrust : mapthrust.cu
	nvcc -std=c++11 -o thrust mapthrust.cu
$(LIBPATH)/gpublob.o :
	(cd ../../lib/;make)

clean : 
	rm map reduce thrust
