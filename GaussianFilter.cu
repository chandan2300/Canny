#include "GaussianFilter.cuh"
#include "SM_Loader.cuh"

void calc_filter(Kernel_Gaussian_weights &kernel, unsigned char kernel_size, double sigma);
__global__ void gaussian_kernel(unsigned char* img_device, unsigned char* out_device, unsigned char kernel_size, unsigned char radius, unsigned int sizeSM, short rows, short cols);


__constant__ Kernel_Gaussian_weights const_gw;


void GaussianFilterGPU(unsigned char* img_host,
                       unsigned char* out,
                       unsigned char kernel_size,
                       double sigma,
                       short rows,
                       short cols){

    unsigned char *out_device, *img_device;
    unsigned char radius;
    unsigned int size=rows*cols*sizeof(unsigned char);
    int sizeSMbyte;
    dim3 num_blocks, num_threads_per_block;

  
    unsigned int factor=16;
    num_threads_per_block.y=factor;
    num_threads_per_block.x=factor;
    
    num_blocks.y = rows/num_threads_per_block.y+((rows%num_threads_per_block.y)==0? 0:1);
    num_blocks.x = cols/num_threads_per_block.x+((cols%num_threads_per_block.x)==0? 0:1);

    
    Kernel_Gaussian_weights kernel;
   
    calc_filter(kernel, kernel_size, sigma);
  
    cudaMemcpyToSymbol(const_gw, &kernel, sizeof(kernel));

    
    cudaMalloc((void**)&img_device, size);
    cudaMalloc((void**)&out_device, size);

    
    cudaMemcpy(img_device, img_host, size, cudaMemcpyHostToDevice);

    
    sizeSMbyte = (num_threads_per_block.y+kernel_size-1)*(num_threads_per_block.x+kernel_size-1)*sizeof(unsigned char);
    unsigned int sizeSM = (num_threads_per_block.x+kernel_size-1);

   
    radius=int(floor((kernel_size-1)/2));

   
    gaussian_kernel<<<num_blocks, num_threads_per_block, sizeSMbyte>>>(img_device, out_device, kernel_size, radius, sizeSM, rows, cols);
    cudaDeviceSynchronize();

    
    cudaMemcpy(out, out_device, size, cudaMemcpyDeviceToHost);

    
    cudaFree(img_device);
    cudaFree(out_device);

}


__global__ void gaussian_kernel(unsigned char* img_device,
                                unsigned char* out_device,
                                unsigned char kernel_size,
                                unsigned char radius,
                                unsigned int sizeSM,
                                short rows,
                                short cols){

    
    extern __shared__ unsigned char sm_gaussian[];

    
    SM_data_loader(sm_gaussian, img_device, sizeSM, radius, rows, cols);

    
    __syncthreads();

  

 
    float sum=0;
    for (int y=0; y<kernel_size; y++)
        for (int x=0; x<kernel_size; x++)
            sum += sm_gaussian[(threadIdx.y+y)*sizeSM+(threadIdx.x+x)]*const_gw.weights[y][x];

    
    unsigned int y = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int x = blockIdx.x*blockDim.y+threadIdx.x;

    
    if (y<rows && x<cols)
          out_device[y*cols+x] = (unsigned char)(floor(sum));

}


void calc_filter(Kernel_Gaussian_weights &kernel, unsigned char kernel_size, double sigma){

    memset(&kernel, 0, sizeof(kernel));
    double r, s = 2.0*sigma*sigma;
    double sum = 0.0;
    unsigned char radius = floor(kernel_size/2);

    
    for (int x=-radius; x<=radius; x++)
      for(int y=-radius; y<=radius; y++){
              r = sqrt(x*x + y*y);
              kernel.weights[x+radius][y+radius] = (exp(-(r*r)/s))/(M_PI*s);
              sum += kernel.weights[x+radius][y+radius];
          }

    
    for(int i=0; i<kernel_size; ++i)
        for(int j=0; j<kernel_size; ++j)
            kernel.weights[i][j] /= sum;

}
