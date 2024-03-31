#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include "utility.hpp"
#include "CannyGPU.cuh"
#include "GaussianFilter.cuh"
#include "CudaTimer.cuh"

using namespace std;
using namespace cv;
using namespace cuda;

int main(int argc, char **argv){
    
    Mat img = imread(argv[1], 0);
    
    Mat img_canny_CPU = Mat(img.rows, img.cols, CV_8U, Scalar(0));
    Mat img_filtered = Mat(img.rows, img.cols, CV_8U, Scalar(0));
    Mat img_canny_GPU_custom = Mat(img.rows, img.cols, CV_8U, Scalar(0));
    Mat img_canny_GPU_opencv = Mat(img.rows, img.cols, CV_8U, Scalar(0));
    Mat img_diff = Mat(img.rows, img.cols, CV_8U, Scalar(0));

    
    unsigned char kernel_size;
    int low_tr;
    int high_tr;
    int mode_on;
    unsigned char L2_norm;
    double sigma=1.4;

    
    if(img.empty()){
        cerr<<"Image is empty"<<endl;
        exit(1);
    }
    if(argc!=7){
        cerr<<"Incorrect number of parameters"<<endl;
        exit(1);
    }

    kernel_size = (unsigned char)atoi(argv[2]);
    low_tr = atoi(argv[3]);
    high_tr = atoi(argv[4]);
    L2_norm = (unsigned char)atoi(argv[5]);
    mode_on = atoi(argv[6]);

    int rows = img.rows; //y
    int cols = img.cols; //x

   
    CudaTimer cuda_timer;

    
    GaussianBlur(img, img_filtered, Size(3,3), sigma);
    

    if(mode_on==0 || mode_on==2){
       

        
        cuda_timer.start_timer();
        CannyCPU(img_filtered, img_canny_CPU, kernel_size, L2_norm, low_tr, high_tr);
        cuda_timer.stop_timer();
        
        printf("Time taken for CPU : %f ms\n", cuda_timer.get_time());
        imwrite("Canny_imp_CPU.jpg", img_canny_CPU);
    }
    if(mode_on==1 || mode_on==2){
       
        unsigned char *img_host = (unsigned char*)malloc(rows*cols*sizeof(unsigned char));
       
        convertImg(img_filtered, img_host, rows, cols);

        unsigned char *img_out_host = (unsigned char*)malloc(rows*cols*sizeof(unsigned char));
        unsigned char *img_out_host_gaussian = (unsigned char*)malloc(rows*cols*sizeof(unsigned char));

        
        cuda_timer.start_timer();
        CannyGPU(img_host, img_out_host, rows, cols, kernel_size, low_tr, high_tr, L2_norm);
        cuda_timer.stop_timer();
        printf("Time taken for GPU : %f ms\n", cuda_timer.get_time());
        

        
        convertImg2(img_canny_GPU_custom, img_out_host, rows, cols);
        imwrite("Canny_imp_GPU_custom.jpg", img_canny_GPU_custom);

        free(img_host);
        free(img_out_host);
    }

    exit(1);

}
