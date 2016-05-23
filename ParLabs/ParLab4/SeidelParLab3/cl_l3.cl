#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define eps 0.1
__kernel void converge(__global float *xk,__global float *xkp,int n,__global float *norm){
    int i = get_global_id(0);
    if(i >= n) return;
    *norm += (xk[i] - xkp[i])*(xk[i] - xkp[i]);
}
bool converge_1(float *xk, float *xkp,int n)
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
__kernel void seidel(__global float *a,__global float *b,__global float *p,__global float *x,int n, __global float *var){
    int i = get_global_id(0);
    if(i >= n) return;
    p[i] = x[i];
    for (int j = 0; j < i; j++)
        *var += (a[i * i + j] * x[j]);
    for (int j = i + 1; j < n; j++)
        *var += (a[i * i + j] * p[j]);
    x[i] = (b[i] - *var) / a[i * i + i];
}
