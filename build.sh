nvcc -c kernel.cu physics.cu
clang++ -c example.cpp
clang++ -L/usr/local/cuda/lib64 kernel.o physics.o example.o -lcudart -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
