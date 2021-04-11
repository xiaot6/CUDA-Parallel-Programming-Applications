// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 32

//@@ insert code here
__global__ void scan(int *input, int *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ int T[2*HISTOGRAM_LENGTH];
  int i =2 * blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len){
    T[threadIdx.x] = input[i];
  }
  else {
    T[threadIdx.x] = 0;
  }
  
  if(i + BLOCK_SIZE < len){
    T[threadIdx.x + blockDim.x] = input[i + blockDim.x];
  } else {
    T[threadIdx.x + blockDim.x] =0;
  }
  
  
  int stride = 1;
  while(stride < 2 * HISTOGRAM_LENGTH){
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 -1;
    if(index < 2 * HISTOGRAM_LENGTH && (index-stride) >= 0){
      T[index] += T[index - stride];
    }
    stride *= 2;
  }
  //__syncthreads();
  stride = HISTOGRAM_LENGTH / 2;
  while(stride > 0){
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 -1;
    if(index + stride < 2 * HISTOGRAM_LENGTH){
      T[index + stride] += T[index];
    }
    stride /= 2;
  }
  __syncthreads();
  if(i < len) output[i] = T[threadIdx.x];
  if(i + blockDim.x < len) output[i+blockDim.x] = T[threadIdx.x + blockDim.x];
}

__global__ void RGB2Gray(float* RGB, int* gray, int* hist, int w, int h){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x < w && y < h){
    int idx = x + y * w ;
    float r = RGB[idx*3];
    float g = RGB[idx*3+1];
    float b = RGB[idx*3+2];
    gray[idx] = 255* (0.21*r + 0.71*g + 0.07*b);
    atomicAdd(&(hist[gray[idx]]),1);
  }
}

__global__ void correct(float* input, float* output, int* cdf, int w, int h){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x < w && y < h){
    for(int c = 0; c < 3; c++){
      int idx = 3 * (x +y* w ) + c;
      int color = int (input[idx] * 255.0);
      int corrected_color = 255 * (cdf[color] - cdf[0]) / (w * h - cdf[0]);
      if (corrected_color < 0) corrected_color = 0;
      if (corrected_color > 255) corrected_color = 255;
      output[idx] = corrected_color / 255.0;
      
    }
  }
  
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *input;
  int *gray;
  float *output;
  int *histogram;
  int *cdf;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
 
  cudaMalloc((void **) & input, imageWidth*imageHeight* imageChannels * sizeof(float));
  cudaMalloc((void **) & gray, imageWidth*imageHeight * sizeof(float));
  cudaMalloc((void **) & output, imageWidth*imageHeight* imageChannels * sizeof(float));
  cudaMalloc((void **) & histogram, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void **) & cdf, HISTOGRAM_LENGTH* sizeof(float));

  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  
  cudaMemcpy(input, hostInputImageData,imageWidth*imageHeight* imageChannels * sizeof(float),cudaMemcpyHostToDevice);
  
  dim3 gridDim(ceil(imageWidth / (BLOCK_SIZE * 1.0)), ceil(imageHeight / (BLOCK_SIZE * 1.0)), 1);
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
  RGB2Gray<<<gridDim, blockDim>>>(input, gray, histogram, imageWidth, imageHeight);

  dim3 gridDimCalcCdf(1, 1, 1);
  dim3 blockDimCalcCdf(HISTOGRAM_LENGTH, 1, 1);
  scan<<<gridDimCalcCdf, blockDimCalcCdf>>>(histogram, cdf, HISTOGRAM_LENGTH);

  dim3 gridDimCorrect(ceil(imageWidth / (BLOCK_SIZE * 1.0)), ceil(imageHeight / (BLOCK_SIZE * 1.0)), 1);
  dim3 blockDimCorrect(BLOCK_SIZE, BLOCK_SIZE, 1);
  correct<<<gridDim, blockDim>>>(input, output, cdf, imageWidth, imageHeight);


  cudaMemcpy(hostOutputImageData, output,imageWidth*imageHeight* imageChannels * sizeof(float),cudaMemcpyDeviceToHost);
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(input);
  cudaFree(gray);
  cudaFree(output);
  cudaFree(histogram);
  cudaFree(cdf);

  return 0;
}
