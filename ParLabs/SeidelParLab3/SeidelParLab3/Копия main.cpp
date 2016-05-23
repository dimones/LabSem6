//
//  main.cpp
//  SeidelParLab3
//
//  Created by Дмитрий Богомолов on 21.05.16.
//  Copyright © 2016 Дмитрий Богомолов. All rights reserved.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <OpenCL/OpenCL.h>
#define eps 0.01
using namespace std;

void printMatrix(float **m,float *b,int n){
    for(int i = 0;i < n;i++){
        for (int j = 0; j < n; j++)
            cout << m[i][j] <<"\t";
        cout << "= " << b[i];
        cout << endl;
    }
}


bool converge(float *xk, float *xkp,int n)
{
    float norm = 0;
    for (int i = 0; i < n; i++)
    {
        norm += (xk[i] - xkp[i])*(xk[i] - xkp[i]);
    }
    if(sqrt(norm) >= eps)
        return false;
    return true;
}

float *seidel(float **a,float *b,const int n){
//    printMatrix(a, b, n);
    float *x = new float[n];
    float *p = new float[n];
    for(int i = 0;i < n;i++)
        x[i] = 0;
    int ccc = 0;
    do
    {
        for (int i = 0; i < n; i++)
            p[i] = x[i];
        
        for (int i = 0; i < n; i++)
        {
            float var = 0;
            for (int j = 0; j < i; j++)
                var += (a[i][j] * x[j]);
            for (int j = i + 1; j < n; j++)
                var += (a[i][j] * p[j]);
            x[i] = (b[i] - var) / a[i][i];
        }
        ccc +=1;
    }
    while (!converge(x, p, n));
    cout << "steps: " << ccc << endl;
    return x;
}
float randMToN(float M, float N)
{
    return M + (rand() / ( RAND_MAX / (N-M) ) ) ;
}
float **generateMatrix(const int n){
    float **output = new float*[n];
    for (int i = 0; i < n; i++) {
        output[i] = new float[n];
        for(int j = 0; j < n; j++)
            output[i][j] = randMToN(-10, 10);
    }
    return output;
}

float *generateRightPart(const int n){
    float *output = new float[n];
    for (int i = 0; i < n; i++) {
        output[i] = randMToN(-10, 10);
    }
    return output;
}
float *opencl_main(float **a,float *b, int n){
    cl_platform_id* platfrorms = new cl_platform_id[2];
    cl_uint ret_num;
    cl_int ret;
    cout << "clGetPlatformIDs ERR=";
    ret = clGetPlatformIDs(2, platfrorms, &ret_num);
    cout << ret << endl;
    
    char* name = new char[100];
    ret = clGetPlatformInfo(platfrorms[0], CL_PLATFORM_NAME, 100, name, NULL);
    cout << "Platform  " << name << endl;
    int v = 1;
    platfrorms[0] = platfrorms[v - 1];
    
    cl_device_id* device = new cl_device_id[1];
    cout << "clGetDeviceIDs ERR=";
    ret = clGetDeviceIDs(platfrorms[0], CL_DEVICE_TYPE_DEFAULT, 1, device, &ret_num);
    cout << ret << endl;
    
    clGetDeviceInfo(device[0], CL_DEVICE_NAME, 100, name, NULL);
    cout << "Device name: " << name << endl;
    
    size_t global_work_size;
    clGetDeviceInfo(device[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &global_work_size, NULL);
    //global_work_size = global_work_size > l ? l : global_work_size;
    
    cout << "clCreateContext ERR=";
    cl_context context = clCreateContext(NULL, 1, device, NULL, NULL, &ret);
    cout << ret << endl;
    //cl_queue_properties* prop = new cl_queue_properties[3]{ CL_QUEUE_ON_DEVICE, CL_QUEUE_ON_DEVICE_DEFAULT, 0 };
    cout << "clCreateCommandQueueWithProperties ERR=";
    cl_command_queue command_queue = clCreateCommandQueue(context, device[0], NULL, &ret);
    cout << ret << endl;
    
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    FILE *fp;
    const char fileName[] = "/Users/dimones-dev/Projects/SeidelParLab3/SeidelParLab3/cl_l3.cl";
    size_t source_size;
    char *source_str;
    try
    {
        fp = fopen(fileName, "r");
        if (!fp)
        {
            cout << "Failed to load kernel. "  << endl;
//            return NULL;
        }
        source_str = (char *)malloc(1024);
        source_size = fread(source_str, 1, 1024, fp);
        source_str[source_size] = '\0';
        fclose(fp);
    }
    catch (const exception& e) 
    {
        cout << e.what() << endl;
//        return ;
    }
    
    cout << "clCreateProgramWithSource ERR=";
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
    cout << ret << endl;
    
    cout << "clBuildProgram ERR=";
    ret = clBuildProgram(program, 1, device, NULL, NULL, NULL);
    cout << ret << endl;
    if (ret == CL_BUILD_PROGRAM_FAILURE) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, device[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        
        // Allocate memory for the log
        char *log = (char *)malloc(log_size);
        
        // Get the log
        clGetProgramBuildInfo(program, device[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        
        // Print the log
        //printf("%s\n", log);
        cout << log << endl;
    }
    cout << "clCreateKernel ERR=";
    kernel = clCreateKernel(program, "converge", &ret);
    cout << ret << endl;
    
    
    float *x = new float[n];
    float *p = new float[n];
    for(int i = 0;i < n;i++)
        x[i] = 0;
    int ccc = 0;
    bool needWhile =true;
//    !converge(x, p, n)
    do
    {
        for (int i = 0; i < n; i++)
            p[i] = x[i];
        
        for (int i = 0; i < n; i++)
        {
            float var = 0;
            for (int j = 0; j < i; j++)
                var += (a[i][j] * x[j]);
            for (int j = i + 1; j < n; j++)
                var += (a[i][j] * p[j]);
            x[i] = (b[i] - var) / a[i][i];
        }
        ccc +=1;
        
        //init x array
        
        cout << "clCreateBuffer (x) ERR=";
        cl_mem xbuff = clCreateBuffer(context,  CL_MEM_READ_ONLY , n * sizeof(float), NULL, &ret);
        cout << ret << endl;
        cout << "x init: " << clEnqueueWriteBuffer(command_queue,xbuff,CL_TRUE,0,sizeof(float) * n, x,0,NULL,NULL) << endl;
        
        cout << "clCreateBuffer (p) ERR=";
        cl_mem pbuff = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &ret);
        cout << ret << endl;
        cout << "p init: " << clEnqueueWriteBuffer(command_queue,pbuff,CL_TRUE,0,sizeof(float) * n, p,0,NULL,NULL) << endl;
        
        float temp_norm = 0;
        cout << "set x: " << clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&xbuff) << endl;
        
        cout << "set p: " << clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&pbuff) << endl;
        
        cout << "set n: " << clSetKernelArg(kernel, 2, sizeof(int), (void *)&n) << endl;
        

        
        
        cl_mem normbuff = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, &ret);
//        cout << "create buffer norm: " << ret << endl;
        
        cout << "norm init: " << clEnqueueWriteBuffer(command_queue,normbuff,CL_TRUE,0,sizeof(float), &temp_norm,0,NULL,NULL) << endl;
        
        ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&normbuff);
        cout << "test set norm: " << ret << endl;
        size_t *size = new size_t;
        *size = n;
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,size, NULL, 0, NULL, NULL);
        cout << "RUN task: " << ret << endl;
        clEnqueueReadBuffer(command_queue, normbuff, CL_TRUE, 0, sizeof(float), &temp_norm, 0, NULL, NULL);
//        ret = clFinish(command_queue);
//        clFlush(command_queue);
        cout << "temp norm" << temp_norm << endl;
        
        if(sqrt(temp_norm) >= eps)
            needWhile = !false;
        else needWhile = !true;
    }
    while (needWhile);
    
    return x;
}
int main(int argc, const char * argv[]) {
    
    float *out;
    const int n = 3;
    float aa[n][n] = { {22.5, 3.51,-6.84}, {3.51,36.45,-0.45}, {-6.84,-0.45,36.59}};
    float bb[n] = { 12.36,-1.8,-22.38 };
    float **a = new float*[n];
    float *b = new float[n];
    for(int i = 0;i < n;i++)
    {
        a[i] = new float[n];
        b[i] = bb[i];
        for (int j = 0; j < n; j++) {
            a[i][j] = aa[i][j];
        }
    }
    
    a = generateMatrix(n); b = generateRightPart(n);
    clock_t beg_cpu = clock();
    out = seidel(a, b, n);
    cout << "CPU time:" << clock() - beg_cpu << endl;
    
    
    
    
    for(int i = 0;i < n; i++){
        cout << "x" << i+1 << "  : " << out[i] << endl;
    }
    clock_t beg_gpu = clock();
    float *out_gpu = opencl_main(a, b, n);
    cout << "GPU time: " << clock() - beg_gpu << endl;
    for(int i = 0;i < n; i++){
        cout << "x" << i+1 << "  : " << out[i] << endl;
    }
    return 0;
}
