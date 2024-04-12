all: main.cpp Utilities.cpp MatMatMultiply.cpp
	icc -O3 main.cpp Utilities.cpp MatMatMultiply.cpp -mkl -qopenmp

clean:
	rm a.out