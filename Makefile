CXX = nvcc
CXX_FLAGS = -g -O3 -arch=sm_21
CXX_LIBS = 
BIN = sofia_gpu

main:
	$(CXX) $(CXX_FLAGS) sofia_gpu.cu -o $(BIN) $(CXX_LIBS)
