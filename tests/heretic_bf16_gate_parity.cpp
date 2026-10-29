// Actual publisher BF16 L0 gate: FP32-input MMVF verify/decode and BF16 high+low
// prefill GEMM versus a double-accumulation CPU oracle. Never changes weights.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "strata/kernels/bf16_gemv.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void cb(cublasStatus_t e){if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cublas failed");}
static float f32(uint16_t v){uint32_t u=(uint32_t)v<<16;float f;std::memcpy(&f,&u,4);return f;}
static uint16_t bf16(float f){uint32_t u;std::memcpy(&u,&f,4);u+=0x7fff+((u>>16)&1);return (uint16_t)(u>>16);}
int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("usage: parity dense.bin offset");
 constexpr int N=2560,Z=6144,T=4;std::vector<uint16_t>w((size_t)N*Z),hi(T*N),lo(T*N);
 std::ifstream in(argv[1],std::ios::binary);in.seekg(std::stoull(argv[2]));in.read((char*)w.data(),w.size()*2);
 if(!in)throw std::runtime_error("truncated actual BF16 gate");
 std::mt19937 gen(740129);std::uniform_real_distribution<float>d(-1,1);std::vector<float>x(T*N),y(T*Z),single(T*Z),gemm(T*Z);
 for(int i=0;i<T*N;++i){x[i]=d(gen);hi[i]=bf16(x[i]);lo[i]=bf16(x[i]-f32(hi[i]));}
 uint16_t *dw,*dh,*dl;float *dx,*dy,*ds,*dg;
 ck(cudaSetDevice(0));ck(cudaMalloc(&dw,w.size()*2));ck(cudaMalloc(&dh,hi.size()*2));ck(cudaMalloc(&dl,lo.size()*2));
 ck(cudaMalloc(&dx,x.size()*4));ck(cudaMalloc(&dy,y.size()*4));ck(cudaMalloc(&ds,y.size()*4));ck(cudaMalloc(&dg,y.size()*4));
 ck(cudaMemcpy(dw,w.data(),w.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));
 ck(cudaMemcpy(dh,hi.data(),hi.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(dl,lo.data(),lo.size()*2,cudaMemcpyHostToDevice));
 strata::kernels::bf16_gemv_fp32_mmvf_multi(dx,N,dw,dy,Z,N,Z,T,nullptr);
 for(int t=0;t<T;++t)strata::kernels::bf16_gemv_fp32_mmvf(dx+t*N,dw,ds+t*Z,N,Z,nullptr);
 cublasHandle_t handle;cb(cublasCreate(&handle));const float one=1,zero=0;
 cb(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,Z,T,N,&one,dw,CUDA_R_16BF,N,dh,CUDA_R_16BF,N,&zero,dg,CUDA_R_32F,Z,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
 cb(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,Z,T,N,&one,dw,CUDA_R_16BF,N,dl,CUDA_R_16BF,N,&one,dg,CUDA_R_32F,Z,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
 ck(cudaDeviceSynchronize());ck(cudaMemcpy(y.data(),dy,y.size()*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(single.data(),ds,y.size()*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(gemm.data(),dg,y.size()*4,cudaMemcpyDeviceToHost));
 double e=0,eg=0,r=0,maximum=0,maxg=0;
 for(int t=0;t<T;++t)for(int z=0;z<Z;++z){double ref=0;for(int n=0;n<N;++n)ref+=(double)x[t*N+n]*f32(w[(size_t)z*N+n]);
  const size_t i=(size_t)t*Z+z;if(std::memcmp(&y[i],&single[i],4))throw std::runtime_error("multi/single MMVF are not bitwise equal");
  const double a=y[i]-ref,b=gemm[i]-ref;e+=a*a;eg+=b*b;r+=ref*ref;maximum=std::max(maximum,std::abs(a));maxg=std::max(maxg,std::abs(b));}
 const double rms=std::sqrt(e/std::max(r,1e-30)),rg=std::sqrt(eg/std::max(r,1e-30));
 if(!std::isfinite(rms)||rms>1e-4||!std::isfinite(rg)||rg>1e-4)throw std::runtime_error("BF16 gate CPU oracle tolerance failed");
 std::cout<<"{\"passed\":true,\"actual_gate_shape\":[6144,2560],\"rows\":4,\"mmvf_multi_single_bitwise\":true,\"mmvf_relative_rms\":"<<rms<<",\"prefill_relative_rms\":"<<rg<<",\"mmvf_max_abs\":"<<maximum<<",\"prefill_max_abs\":"<<maxg<<"}\n";
 cublasDestroy(handle);cudaFree(dw);cudaFree(dh);cudaFree(dl);cudaFree(dx);cudaFree(dy);cudaFree(ds);cudaFree(dg);return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
