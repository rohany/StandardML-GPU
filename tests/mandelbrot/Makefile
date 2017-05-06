MLTONFLAGS = -default-ann 'allowFFI true' -link-opt \
						 '-rdynamic -L/usr/local/depot/cuda-8.0/lib64/ -lcudart -lstdc++'
LIBPATH = ../../lib/
BLOB = $(LIBPATH)/gpublob.o

mandel : mandel.sml $(BLOB)
	mlton $(MLTONFLAGS) mandel.mlb ../../lib/gpublob.o
$(LIBPATH)/gpublob.o :
	(cd ../../lib/;make)

clean : 
	rm mandel
