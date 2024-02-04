#include "CannyGPU.cuh"
#include "SM_Loader.cuh"


__constant__ Kernel_weights const_weights;

__global__ void sobel_kernel(unsigned char* img, unsigned char* sobel_module, float* sobel_dir, unsigned char kernel_size, unsigned char radius,
                             unsigned int sizeSM, short rows, short cols, unsigned char L2_norm);

__global__ void non_max_suppresion_kernel(unsigned char* sobel_module, float* sobel_dir, unsigned char* out,
                                          unsigned int sizeSM, unsigned char radius, short rows, short cols);

__global__ void hysteresis_kernel(unsigned char* img_non_max_sup, unsigned char* out, unsigned char* sobel_module,
                                  unsigned int sizeSM, unsigned char radius, short rows, short cols, int low_tr, int high_tr);

void CannyGPU(unsigned char* img_host,
              unsigned char* out,
              short rows,
              short cols,
              unsigned char kernel_size,
              int low_tr,
              int high_tr,
              unsigned char L2_norm){

    unsigned char *out_non_max_device, *out_device;
    float *sobel_dir_device;
    unsigned char *img_device, *sobel_module_device;
    unsigned int factor=16;
    unsigned int size;
    int sizeSMbyte;
    unsigned char radius;


    dim3 num_blocks, num_threads_per_block;
    
    num_threads_per_block.y=factor;
    num_threads_per_block.x=factor;

    num_blocks.y = rows/num_threads_per_block.y+((rows%num_threads_per_block.y)==0? 0:1);
    num_blocks.x = cols/num_threads_per_block.x+((cols%num_threads_per_block.x)==0? 0:1);

    
    size=rows*cols*sizeof(unsigned char);

    
    cudaMalloc((void**)&img_device, size);
    cudaMalloc((void**)&sobel_dir_device, rows*cols*sizeof(float));
    cudaMalloc((void**)&sobel_module_device, size);
    cudaMalloc((void**)&out_non_max_device, size);
    cudaMalloc((void**)&out_device, size);

    
    cudaMemcpy(img_device, img_host, size, cudaMemcpyHostToDevice);

   
    Kernel_weights k;
    init_kernel_weights(k, kernel_size);
    
    cudaMemcpyToSymbol(const_weights, &k, sizeof(k));

    
    radius=int(floor((kernel_size-1)/2));

    
    sizeSMbyte = (num_threads_per_block.y+kernel_size-1)*(num_threads_per_block.x+kernel_size-1)*sizeof(unsigned char);
    unsigned int sizeSM = (num_threads_per_block.x+kernel_size-1);


    
    sobel_kernel<<<num_blocks, num_threads_per_block, sizeSMbyte>>>(img_device, sobel_module_device, sobel_dir_device, kernel_size, radius, sizeSM, rows, cols, L2_norm);
    cudaDeviceSynchronize();

    
    sizeSMbyte = (num_threads_per_block.x+3-1)*(num_threads_per_block.y+3-1)*sizeof(unsigned char);
    sizeSM = (num_threads_per_block.x+3-1);
    radius=1; 
    non_max_suppresion_kernel<<<num_blocks, num_threads_per_block, sizeSMbyte>>>(sobel_module_device, sobel_dir_device, out_non_max_device, sizeSM, radius, rows, cols);
    cudaDeviceSynchronize();
    
    sizeSMbyte = 2*(num_threads_per_block.x+3-1)*(num_threads_per_block.y+3-1)*sizeof(unsigned char);
    hysteresis_kernel<<<num_blocks, num_threads_per_block, sizeSMbyte>>>(out_non_max_device, out_device, sobel_module_device, sizeSM, radius, rows, cols, low_tr, high_tr);
    cudaDeviceSynchronize();

   
    cudaMemcpy(out, out_device, size, cudaMemcpyDeviceToHost);

    
    cudaFree(out_non_max_device);
    cudaFree(out_device);
    cudaFree(img_device);
    cudaFree(sobel_module_device);
    cudaFree(sobel_dir_device);

}


__global__ void hysteresis_kernel(unsigned char* img_non_max_sup,
                                  unsigned char* out,
                                  unsigned char* sobel_module,
                                  unsigned int sizeSM,
                                  unsigned char radius,
                                  short rows,
                                  short cols,
                                  int low_tr,
                                  int high_tr){

   
    extern __shared__ unsigned char sm[];
    unsigned char* sm_mag = &sm[0];
    unsigned char* sm_non_max = &sm[sizeSM*sizeSM];

    
    SM_data_loader(sm_mag, sobel_module, sizeSM, radius, rows, cols);
    
    SM_data_loader(sm_non_max, img_non_max_sup, sizeSM, radius, rows, cols);

   
    __syncthreads();

    
    unsigned int y = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int x = blockIdx.x*blockDim.x+threadIdx.x;

   
    if(y<rows && x<cols){

      
        int local_index = (threadIdx.y+radius)*sizeSM+(threadIdx.x+radius);

        
        if(sm_non_max[local_index]==0) out[y*cols+x]=0;
      
        else{

            bool edge = false;
        
            if(sm_mag[local_index]>high_tr) edge=true;
          
            else if(sm_mag[local_index]<low_tr) edge=false;
            
            else if(sm_mag[local_index]>=low_tr && sm_mag[local_index]<=high_tr){

                for(int i=0; i<3; i++)
                    for(int j=0; j<3; j++){

                      
                        if(sm_mag[(threadIdx.y+i)*sizeSM+(threadIdx.x+j)]>high_tr){
                              edge=true;
                              
                              i=j=3;
                          }
                    }
            }

            // Mark whether it is an edge pixel or not.
            if(edge) out[y*cols+x]=255;
              else out[y*cols+x]=0;

        }
    }
}

__global__ void non_max_suppresion_kernel(unsigned char* sobel_module,
                                          float* sobel_dir,
                                          unsigned char* out,
                                          unsigned int sizeSM,
                                          unsigned char radius,
                                          short rows,
                                          short cols){

   
    extern __shared__ unsigned char sm[];
  
    SM_data_loader(sm, sobel_module, sizeSM, radius, rows, cols);

   
    __syncthreads();

   
    unsigned int y = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int x = blockIdx.x*blockDim.x+threadIdx.x;

   
    float currDir = sobel_dir[y*cols+x];
    
    unsigned char mag = sm[(threadIdx.y+radius)*sizeSM+(threadIdx.x+radius)];
    
    while(currDir<0) currDir+=180;

    bool check=true;

    if(y>=rows-1 || y<=0 || x>=cols-1 || x<=0) check=false;
    else{
       
        if(currDir>22.5 && currDir<=67.5){
            if(mag<sm[(threadIdx.y-1 +radius)*sizeSM+(threadIdx.x-1 +radius)] ||
               mag<sm[(threadIdx.y+1 +radius)*sizeSM+(threadIdx.x+1 +radius)]) check = false;
        }

        else if(currDir>67.5 && currDir<=112.5){
            if(mag<sm[(threadIdx.y-1 +radius)*sizeSM+(threadIdx.x +radius)] ||
               mag<sm[(threadIdx.y+1 +radius)*sizeSM+(threadIdx.x +radius)]) check = false;

        }

        else if(currDir>112.5 && currDir<=157.5){
            if(mag<sm[(threadIdx.y+1 +radius)*sizeSM+(threadIdx.x-1 +radius)] ||
              mag<sm[(threadIdx.y-1 +radius)*sizeSM+(threadIdx.x+1 +radius)]) check = false;

        }

        else{
            if(mag<sm[(threadIdx.y +radius)*sizeSM+(threadIdx.x-1 +radius)] ||
              mag<sm[(threadIdx.y +radius)*sizeSM+(threadIdx.x+1 +radius)]) check = false;
        }

    }
    if(check) out[y*cols+x]=255;
    else out[y*cols+x]=0;

}


__global__ void sobel_kernel(unsigned char* img,
                             unsigned char* sobel_module,
                             float* sobel_dir,
                             unsigned char kernel_size,
                             unsigned char radius,
                             unsigned int sizeSM,
                             short rows,
                             short cols,
                             unsigned char L2_norm){

    
    extern __shared__ unsigned char sm[];

    // Upload the necessary data from GM to SM.
    SM_data_loader(sm, img, sizeSM, radius, rows, cols);


    __syncthreads();

    
    float sumX=0, sumY=0;
    for (int y=0; y<kernel_size; y++)
        for (int x=0; x<kernel_size; x++){
          sumY += sm[(threadIdx.y+y)*sizeSM+(threadIdx.x+x)]*const_weights.y[y][x];
          sumX += sm[(threadIdx.y+y)*sizeSM+(threadIdx.x+x)]*const_weights.x[y][x];
        }

   
    unsigned int y = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int x = blockIdx.x*blockDim.y+threadIdx.x;

    
    if (y<rows && x<cols){

      
      int pixel_intensity;
      if(L2_norm==0){
          pixel_intensity = abs(sumY) + abs(sumX);
      }
      else{
          pixel_intensity = sqrt((sumY*sumY)+(sumX*sumX));
      }
      pixel_intensity = pixel_intensity > 255? 255: pixel_intensity < 0? 0: pixel_intensity;

      
      sobel_module[y*cols+x] = pixel_intensity;

      
      sobel_dir[y*cols+x] = atan2(sumY,sumX)*(180/M_PI);

    }
}


void init_kernel_weights(Kernel_weights &k, unsigned char kernel_size){

    memset(&k, 0, sizeof(k));

    if(kernel_size==3){
        k.y[0][0]=1; k.y[0][1]=2; k.y[0][2]=1;
        k.y[1][0]=0; k.y[1][1]=0; k.y[1][2]=0;
        k.y[2][0]=-1; k.y[2][1]=-2; k.y[2][2]=-1;

        k.x[0][0]=1; k.x[0][1]=0; k.x[0][2]=-1;
        k.x[1][0]=2; k.x[1][1]=0; k.x[1][2]=-2;
        k.x[2][0]=1; k.x[2][1]=0; k.x[2][2]=-1;
    }
    else if(kernel_size==5){

    }
    else if(kernel_size==7){

    }


}
