#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <inttypes.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <cuda.h>

//设置数据块长度16字节（128位）
//设置gpu中每block中thread数量512
#define AES_BLOCK_SIZE 16
#define THREADS_PER_BLOCK 512

// S盒，host，扩展密钥用
uint8_t s_box[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
    0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
    0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
    0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
    0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
    0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
    0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
    0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
    0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
    0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
    0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
    0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
    0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
    0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
    0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
    0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
    0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

// S盒，device，加密用
__constant__ static const uint8_t sbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
    0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
    0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
    0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
    0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
    0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
    0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
    0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
    0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
    0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
    0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
    0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
    0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
    0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
    0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
    0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
    0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};


// 逆S盒，device，解密用
__constant__ static const uint8_t sboxinv[256] = {
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38,
    0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87,
    0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d,
    0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2,
    0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16,
    0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda,
    0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a,
    0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02,
    0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea,
    0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85,
    0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89,
    0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20,
    0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31,
    0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d,
    0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0,
    0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26,
    0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
};


// x-time operation
__device__ uint8_t rj_xtime(uint8_t x){
  return (x & 0x80) ? ((x << 1) ^ 0x1b) : (x << 1);
}


// 字节代换
__device__ void aes_subBytes(uint8_t *buf){
  register uint8_t i, b;
  for (i = 0; i < 16; ++i){
    b = buf[i];
    buf[i] = sbox[b];
  }
} 


// 逆字节代换
__device__ void aes_subBytes_inv(uint8_t *buf){
  register uint8_t i, b;
  for (i = 0; i < 16; ++i){
    b = buf[i];
    buf[i] = sboxinv[b];
  }
} 


// 轮密钥加
__device__ void aes_addRoundKey(uint8_t *buf, uint8_t *key,uint8_t r){
  //register uint8_t i = 16;
  //while (i--){
   buf[16] ^= key[16*r+16];
   buf[15] ^= key[16*r+15];
   buf[14] ^= key[16*r+14];
   buf[13] ^= key[16*r+13];
   buf[12] ^= key[16*r+12];
   buf[11] ^= key[16*r+11];
   buf[10] ^= key[16*r+10];
   buf[9] ^= key[16*r+9];
   buf[8] ^= key[16*r+8];
   buf[7] ^= key[16*r+7];
   buf[6] ^= key[16*r+6];
   buf[5] ^= key[16*r+5];
   buf[4] ^= key[16*r+4];
   buf[3] ^= key[16*r+3];
   buf[2] ^= key[16*r+2];
   buf[1] ^= key[16*r+1];
   buf[0] ^= key[16*r+0];
  //}
} 

//行位移
__device__ void aes_shiftRows(uint8_t *buf){
  register uint8_t i, j; 
  i = buf[1];
  buf[1] = buf[5];
  buf[5] = buf[9];
  buf[9] = buf[13];
  buf[13] = i;
  i = buf[10];
  buf[10] = buf[2];
  buf[2] = i;
  j = buf[3];
  buf[3] = buf[15];
  buf[15] = buf[11];
  buf[11] = buf[7];
  buf[7] = j;
  j = buf[14];
  buf[14] = buf[6];
  buf[6]  = j;
}


// 逆行位移
__device__ void aes_shiftRows_inv(uint8_t *buf){
  register uint8_t i, j; 
  i = buf[1];
  buf[1] = buf[13];
  buf[13] = buf[9];
  buf[9] = buf[5];
  buf[5] = i;
  i = buf[2];
  buf[2] = buf[10];
  buf[10] = i;
  j = buf[3];
  buf[3] = buf[7];
  buf[7] = buf[11];
  buf[11] = buf[15];
  buf[15] = j;
  j = buf[6];
  buf[6] = buf[14];
  buf[14] = j;
} 

// 列混合
__device__ void aes_mixColumns(uint8_t *buf){
  register uint8_t i, a, b, c, d, e;
  for (i = 0; i < 16; i += 4){
    a = buf[i];
    b = buf[i + 1];
    c = buf[i + 2];
    d = buf[i + 3];
    e = a ^ b ^ c ^ d;
    buf[i] ^= e ^ rj_xtime(a^b);
    buf[i+1] ^= e ^ rj_xtime(b^c);
    buf[i+2] ^= e ^ rj_xtime(c^d);
    buf[i+3] ^= e ^ rj_xtime(d^a);
  }
} 


// 逆列混合
__device__ void aes_mixColumns_inv(uint8_t *buf){
  register uint8_t i, a, b, c, d, e, x, y, z;
  for (i = 0; i < 16; i += 4){
    a = buf[i];
    b = buf[i + 1];
    c = buf[i + 2];
    d = buf[i + 3];
    e = a ^ b ^ c ^ d;
    z = rj_xtime(e);
    x = e ^ rj_xtime(rj_xtime(z^a^c));
    y = e ^ rj_xtime(rj_xtime(z^b^d));
    buf[i] ^= x ^ rj_xtime(a^b);
    buf[i+1] ^= y ^ rj_xtime(b^c);
    buf[i+2] ^= x ^ rj_xtime(c^d);
    buf[i+3] ^= y ^ rj_xtime(d^a);
  }
} 

// 字循环 用于密钥扩展
void rot_word(uint8_t *w) {
  uint8_t tmp;
  uint8_t i;
  tmp = w[0];
  for (i = 0; i < 3; i++) {
    w[i] = w[i+1];
  }
  w[3] = tmp;
}

// 字节代换 用于密钥扩展
void sub_word(uint8_t *w) {
  uint8_t i;
  for (i = 0; i < 4; i++) {
    w[i] = s_box[w[i]];
  }
}

// 密钥扩展
void aes_key_expansion(uint8_t *key, uint8_t *w) {
  uint8_t tmp[4];
  uint8_t i;
  int Nb=4;
  int Nr=14;
  int Nk=8;
  uint8_t len = Nb*(Nr+1);
  uint8_t RC[11] = {0x00,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36};

  for (i = 0; i < Nk; i++) {
    w[4*i+0] = key[4*i+0];
    w[4*i+1] = key[4*i+1];
    w[4*i+2] = key[4*i+2];
    w[4*i+3] = key[4*i+3];
  }

  for (i = Nk; i < len; i++) {
    tmp[0] = w[4*(i-1)+0];
    tmp[1] = w[4*(i-1)+1];
    tmp[2] = w[4*(i-1)+2];
    tmp[3] = w[4*(i-1)+3];

    if (i%Nk == 0) {
      rot_word(tmp);
      sub_word(tmp);
      tmp[0] = RC[i/Nk]^tmp[0];
      tmp[1] = 0x00^tmp[1];
      tmp[2] = 0x00^tmp[2];
      tmp[3] = 0x00^tmp[3];
    } else if (Nk > 6 && i%Nk == 4) {
      sub_word(tmp);
    }

    w[4*i+0] = w[4*(i-Nk)+0]^tmp[0];
    w[4*i+1] = w[4*(i-Nk)+1]^tmp[1];
    w[4*i+2] = w[4*(i-Nk)+2]^tmp[2];
    w[4*i+3] = w[4*(i-Nk)+3]^tmp[3];
  }

}


// 使用aes加密算法，对一块（128位）数据进行加密 
__constant__ uint8_t w2[240];
__constant__ uint8_t w3[240];
__global__ void aes256_encrypt_ecb(uint8_t *buf_d, unsigned long numbytes){
  uint8_t *key=w2;
  uint8_t i;
  uint8_t buf_t[AES_BLOCK_SIZE]; // thread buffer
  //计算待加密数据在总数据中的偏移
  unsigned long offset = (blockIdx.x * THREADS_PER_BLOCK * AES_BLOCK_SIZE) + (threadIdx.x * AES_BLOCK_SIZE);
  if (offset >= numbytes) {  return; }
  //拷贝待加密数据至buf_t
  memcpy(buf_t, &buf_d[offset], AES_BLOCK_SIZE);

  //加密，共14轮
  aes_addRoundKey(buf_t, key,0);
  for(i = 1; i < 14; i++){
    aes_subBytes(buf_t);
    aes_shiftRows(buf_t);
    aes_mixColumns(buf_t);
    aes_addRoundKey( buf_t, key,i);
  }
  aes_subBytes(buf_t);
  aes_shiftRows(buf_t);
  aes_addRoundKey(buf_t,key,14);
  //将加密后的buf_t拷贝回总数据
  memcpy(&buf_d[offset], buf_t, AES_BLOCK_SIZE);
  __syncthreads();
} 



// 使用aes解密算法，对一块（128位）数据进行解密
__global__ void aes256_decrypt_ecb(uint8_t *buf_d, unsigned long numbytes){
  uint8_t i;
  uint8_t *key=w3;
  uint8_t buf_t[AES_BLOCK_SIZE];
  //计算待解密数据在总数据中的偏移
  unsigned long offset = (blockIdx.x * THREADS_PER_BLOCK * AES_BLOCK_SIZE) + (threadIdx.x * AES_BLOCK_SIZE);
  if (offset >= numbytes) { return; }
  //拷贝待解密数据至buf_t
  memcpy(buf_t, &buf_d[offset], AES_BLOCK_SIZE);

  //解密，共14轮
  aes_addRoundKey(buf_t, key,14);
  for (i = 1; i < 14; i++){
  	aes_shiftRows_inv(buf_t);
  	aes_subBytes_inv(buf_t);
  	aes_addRoundKey( buf_t,  key,14-i);
    aes_mixColumns_inv(buf_t);
    }
  aes_shiftRows_inv(buf_t);
  aes_subBytes_inv(buf_t);
  aes_addRoundKey( buf_t,  key,0);
  //将解密后的buf_t拷贝回总数据
  memcpy(&buf_d[offset], buf_t, AES_BLOCK_SIZE);
  __syncthreads();
} 



//aes加密
void encryptdemo(uint8_t *key, uint8_t *buf, unsigned long numbytes){
  uint8_t *buf_d;

  uint8_t *w;
  const int nStreams = 16;
  const int ChunkSize = numbytes / nStreams;
  
  printf("\nBeginning encryption\n");

  //记录加密算法开始时间
  cudaEvent_t start1;
  cudaEventCreate(&start1);
  cudaEvent_t stop1;
  cudaEventCreate(&stop1);
  cudaEventRecord(start1);
  //创建工作流 
  cudaStream_t streams[nStreams];
  for (int i = 0; i < nStreams; i++) {
    cudaStreamCreate(&streams[i]);
  }
  //将s盒拷贝到常量内存中，可以加快读取速度。
  cudaMemcpyToSymbol(sbox, sbox, sizeof(uint8_t)*256);
  w = (uint8_t*)malloc(240*sizeof(uint8_t));
  //密钥扩展
  aes_key_expansion(key, w);
  cudaMalloc((void**)&buf_d, numbytes);
  //将扩展后的密钥存储到常量内存中
  cudaMemcpyToSymbol(w2, w, 240*sizeof(uint8_t));
  //计算dimblock数量，需要多除一个工作流的总数
  dim3 dimBlock(ceil((double)numbytes/nStreams / (double)(THREADS_PER_BLOCK * AES_BLOCK_SIZE)));
  dim3 dimGrid(THREADS_PER_BLOCK);
  printf("Creating %d threads over %d blocks\n", dimBlock.x*dimGrid.x, dimBlock.x);

  int nOffset = 0;
  //开始工作流
  for(int i=0; i<nStreams; i++)
  {
    
    nOffset = ChunkSize*i;

    cudaMemcpyAsync(  buf_d+nOffset,
                      buf+nOffset,
                      ChunkSize,
                      cudaMemcpyHostToDevice,
                      streams[i] );


    aes256_encrypt_ecb<<<dimBlock, dimGrid, 0, streams[i]>>>(
                                   buf_d+nOffset, 
                                   ChunkSize);

    cudaMemcpyAsync(  buf+nOffset,
                      buf_d+nOffset,
                      ChunkSize,
                      cudaMemcpyDeviceToHost,
                      streams[i] );
      
  }
  //等待运算完成
  cudaDeviceSynchronize();

  //记录加密算法结束时间，并计算加密速度  
  cudaEventRecord(stop1);
  cudaEventSynchronize(stop1);
  float msecTotal1,total;
  cudaEventElapsedTime(&msecTotal1, start1, stop1);
  total=msecTotal1/1000;
  printf("time:%f\n",total);
  printf("Throughput: %f Gbps\n", numbytes/total/1024/1024/1024*8);

}


// aes解密
void decryptdemo(uint8_t *key, uint8_t *buf, unsigned long numbytes){
  uint8_t *buf_d;

  uint8_t *w;
  const int nStreams = 16;
  const int ChunkSize = numbytes / nStreams;
  
  printf("\nBeginning encryption\n");

  //记录解密算法开始时间
  cudaEvent_t start1;
  cudaEventCreate(&start1);
  cudaEvent_t stop1;
  cudaEventCreate(&stop1);
  cudaEventRecord(start1);
  //创建工作流  
  cudaStream_t streams[nStreams];
  for (int i = 0; i < nStreams; i++) {
    cudaStreamCreate(&streams[i]);
  }
  //将s盒拷贝到常量内存中，可以加快读取速度。
  cudaMemcpyToSymbol(sbox, sbox, sizeof(uint8_t)*256);
  w = (uint8_t*)malloc(240*sizeof(uint8_t));
  //密钥扩展
  aes_key_expansion(key, w);
  cudaMalloc((void**)&buf_d, numbytes);
  //将扩展后的密钥存储到常量内存中
  cudaMemcpyToSymbol(w3, w, 240*sizeof(uint8_t));
  //计算dimblock数量，需要多除一个工作流的总数
  dim3 dimBlock(ceil((double)numbytes/nStreams / (double)(THREADS_PER_BLOCK * AES_BLOCK_SIZE)));
  dim3 dimGrid(THREADS_PER_BLOCK);
  printf("Creating %d threads over %d blocks\n", dimBlock.x*dimGrid.x, dimBlock.x);

  int nOffset = 0;
  //开始工作流
  for(int i=0; i<nStreams; i++)
  {
    
    nOffset = ChunkSize*i;

    cudaMemcpyAsync(  buf_d+nOffset,
                      buf+nOffset,
                      ChunkSize,
                      cudaMemcpyHostToDevice,
                      streams[i] );


    aes256_decrypt_ecb<<<dimBlock, dimGrid, 0, streams[i]>>>(
                                   buf_d+nOffset, 
                                   ChunkSize);

    cudaMemcpyAsync(  buf+nOffset,
                      buf_d+nOffset,
                      ChunkSize,
                      cudaMemcpyDeviceToHost,
                      streams[i] );
      
  }
  //等待运算完成
  cudaDeviceSynchronize();

  //记录解密算法结束时间，并计算解密速度
  cudaEventRecord(stop1);
  cudaEventSynchronize(stop1);
  float msecTotal1,total;
  cudaEventElapsedTime(&msecTotal1, start1, stop1);
  total=msecTotal1/1000;
  printf("time:%f\n",total);
  printf("Throughput: %f Gbps\n", numbytes/total/1024/1024/1024*8);

}

__global__ void GPU_init() { }



int main(int argc,char** argv){

  FILE *file;
  uint8_t *buf,*buf2; 
  unsigned long numbytes;
  char *fname;
  int  i;
  int padding;
 
  uint8_t key[32];
  // 设置gpu
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if (error_id != cudaSuccess){
    printf("Error: %s\n", cudaGetErrorString(error_id));
    printf("Exiting...\n");
    exit(EXIT_FAILURE);
  }
  if (deviceCount == 0){
    printf("There are no available device(s) that support CUDA\n");
    exit(EXIT_FAILURE);
  }


  // 打开待加密文件
  fname = argv[1];  
  file = fopen(fname, "r");
  if (file == NULL) {printf("File %s doesn't exist\n", fname); exit(1); }
  printf("Opened file %s\n", fname);
  fseek(file, 0L, SEEK_END);
  numbytes = ftell(file);
  printf("Size is %lu\n", numbytes);

  // 将待加密数据读取到内存
  fseek(file, 0L, SEEK_SET);
  buf = (uint8_t*)calloc(numbytes, sizeof(uint8_t)); 
  if(buf == NULL) exit(1);
  if (fread(buf, 1, numbytes, file) != numbytes)
  {
    printf("Unable to read all bytes from file %s\n", fname);
    exit(EXIT_FAILURE);
  }
  fclose(file);

  // 补全
  padding = AES_BLOCK_SIZE * 16 - numbytes % (AES_BLOCK_SIZE * 16);
  numbytes += padding;
  printf("Padding file with %d bytes for a new size of %lu\n", padding, numbytes);

  // 生成密钥
  for (i = 0; i < sizeof(key);i++) key[i] = i;
  cudaMallocHost((void**)&buf2, numbytes);
  cudaMemcpy(buf2, buf, numbytes, cudaMemcpyHostToHost);
  // gpu初始化
  GPU_init<<<1, 1>>>();


  // 调用加密算法
  encryptdemo(key, buf2, numbytes);
  // 将加密后的数据写入cipher.txt
  file = fopen("cipher.txt", "w");
  fwrite(buf2, 1, numbytes, file);
  fclose(file);

  // 解密
  decryptdemo(key, buf2, numbytes);
  // 将解密后的数据写回output.txt
  file = fopen("output.txt", "w");
  fwrite(buf2, 1, numbytes - padding, file);
  fclose(file);
  free(buf);
  //free(buf2);

  return EXIT_SUCCESS;
}