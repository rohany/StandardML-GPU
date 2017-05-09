MLTONFLAGS = -default-ann 'allowFFI true' -link-opt \
						 '-rdynamic -L/usr/local/cuda-8.0/lib64/ -lcudart -lstdc++'
LIBPATH = ../../lib/

test : test.sml $(LIBPATH)/gpublob.o
	mlton $(MLTONFLAGS) test.mlb ../../lib/gpublob.o
$(LIBPATH)/gpublob.o : 
	(cd ../../lib/; make)
clean : 
	rm test
