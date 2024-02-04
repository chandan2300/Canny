#include "SM_Loader.cuh"

__device__ void SM_data_loader(unsigned char *sm,
                               unsigned char *img,
                               unsigned int sizeSM,
                               unsigned char radius,
                               short rows,
                               short cols){

   
    int dest = threadIdx.y*blockDim.x+threadIdx.x;

   
    int destY = dest/sizeSM;
    int destX = dest%sizeSM;

   
    int srcY = blockIdx.y*blockDim.y+destY-radius;
    int srcX = blockIdx.x*blockDim.x+destX-radius;
   
    int src = srcY*cols+srcX;

    
    if (srcY>=0 && srcY<rows && srcX>=0 && srcX<cols)
        sm[destY*sizeSM+destX] = img[src];
    else
        sm[destY*sizeSM+destX] = 0;

    

    for (int iter=1; iter <= (sizeSM*sizeSM)/(blockDim.y*blockDim.x); iter++)
    {
        dest = threadIdx.y * blockDim.x + threadIdx.x + (blockDim.y * blockDim.x); // offset aggiunto
        destY = dest/sizeSM;
        destX = dest%sizeSM;
        srcY = blockIdx.y*blockDim.x+destY-radius;
        srcX = blockIdx.x*blockDim.x+destX-radius;
        src = srcY*cols+srcX;
        if (destY < sizeSM){
            if (srcY >=0 && srcY<rows && srcX>=0 && srcX<cols)
                sm[destY*sizeSM+destX] = img[src];
            else
                sm[destY*sizeSM+destX] = 0;
        }
    }


}
