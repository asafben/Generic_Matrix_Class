CPPFLAGS = -Wextra -Wall -Wvla -pthread -c -g -std=c++11
GCC = g++

all: GenericMatrixDriver
	./GenericMatrixDriver

Complex.o: Complex.cpp Complex.h
	$(GCC) $(CPPFLAGS) Complex.cpp

GenericMatrixDriver.o: GenericMatrixDriver.cpp Complex.h Matrix.hpp.gch
	$(GCC) $(CPPFLAGS) GenericMatrixDriver.cpp

Matrix.hpp.gch: Matrix.hpp Complex.h
	$(GCC) $(CPPFLAGS) Matrix.hpp Complex.h

Complex.o GenericMatrixDriver.o: Complex.h

GenericMatrixDriver.o: Matrix.hpp

GenericMatrixDriver: GenericMatrixDriver.o Complex.o
	$(GCC) $^ -lpthread -o $@

tar:
	tar cvf ex3.tar Matrix.hpp Makefile README

val:
	valgrind --leak-check=full --show-possibly-lost=yes --show-reachable=yes \
	--undef-value-errors=yes GenericMatrixDriver

clean:
	-rm -f *.o GenericMatrixDriver

.PHONY: clean
