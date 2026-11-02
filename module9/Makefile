LIBS := -lnppc -lnppial -lnppicc -lnppidei -lnppif \
            -lnppig -lnppim -lnppist -lnppisu -lnppitc -lcublas
SRCS := assignment.cu conv.cu
HDRS := conv.h check.h

all: assignment.exe

assignment.exe: $(SRCS) $(HDRS)
	nvcc -O3 $(SRCS) -o $@ $(LIBS)
