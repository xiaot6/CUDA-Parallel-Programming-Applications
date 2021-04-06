#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define MASK_RADIUS 1

#define TILE_WIDTH 8

//@@ Define constant memory for device kernel here

__constant__ float deviceKernel[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];


__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int ix = bx * blockDim.x + tx;
  int iy = by * blockDim.y + ty;
  int iz = bz * blockDim.z + tz;
  
  __shared__ float N_ds[TILE_WIDTH][TILE_WIDTH][TILE_WIDTH];
  
  if(ix < x_size && iy < y_size && iz < z_size){
    N_ds[tz][ty][tx] = input[iz * (y_size * x_size) + iy * (x_size) + ix];
  } else {
    N_ds[tz][ty][tx] = 0.0f;
  }
  
  __syncthreads();
  
  float pValue = 0.0f;
  for(int z = -MASK_RADIUS; z <= MASK_RADIUS; z++){
    for(int y = -MASK_RADIUS; y <= MASK_RADIUS; y++){
      for(int x = -MASK_RADIUS; x <= MASK_RADIUS; x++){
        int z_in = tz + z;
        int y_in = ty + y;
        int x_in = tx + x;
        
        if(z_in >= 0 && z_in < TILE_WIDTH 
           && y_in >= 0 && y_in < TILE_WIDTH 
           && x_in >= 0 && x_in < TILE_WIDTH){
          pValue += deviceKernel[z + MASK_RADIUS][y + MASK_RADIUS][x + MASK_RADIUS] 
                    * N_ds[z_in][y_in][x_in];
        } else {
          if(ix + x >= 0 && iy + y >= 0 && iz + z >= 0 && 
            ix + x < x_size && iy+ y < y_size && iz + z < z_size){
            pValue += deviceKernel[z + MASK_RADIUS][y + MASK_RADIUS][x + MASK_RADIUS] 
                      * input[(ix+x) + (x_size * (iy+y)) + (x_size * y_size * (iz+z))];
                
          }
        }
       
      }
    }
  }
  
  if(ix < x_size && iy < y_size && iz < z_size){
    output[iz * (y_size * x_size) + iy * (x_size) + ix] = pValue;
  }
  
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  
  cudaMalloc((void **) &deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void **) &deviceOutput, (inputLength - 3) * sizeof(float));
  
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  
  cudaMemcpy(deviceInput, &hostInput[3], (inputLength-3)* sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength * sizeof(float));
  
  
  
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  
  dim3 DimGrid(ceil(x_size/(TILE_WIDTH * 1.0)), ceil(y_size/(TILE_WIDTH * 1.0)),ceil(z_size/(TILE_WIDTH * 1.0)));
  dim3 DimBlock(TILE_WIDTH,TILE_WIDTH,TILE_WIDTH);
               

  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid,DimBlock>>>(deviceInput, deviceOutput,z_size,y_size, x_size);
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[3], deviceOutput, (inputLength-3)*sizeof(float), cudaMemcpyDeviceToHost);
  
  
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
