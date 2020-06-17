#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

int NX;
int NY;
int BS;
float *data;
float *data_gpu;

/* in microseconds (us) */
long time_diff_us(struct timeval st, struct timeval et) {
    return (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);
}

void init(float *data) {
  int x, y;
  int cx = NX/2, cy = 0; /* center of ink */
  int rad = (NX+NY)/8; /* radius of ink */

  for(y = 0; y < NY; y++) {
    for(x = 0; x < NX; x++) {
      float v = 0.0;
      if (((x-cx)*(x-cx)+(y-cy)*(y-cy)) < rad*rad) {
	v = 1.0;
      }
      data[x+y*NX] = v;
      data[NX*NY+x+y*NX] = v;
    }
  }
  return;
}

/* Calculate for one time step */
/* Input: data[t%2], Output: data[(t+1)%2] */
__global__ void calc(float *data_gpu, int NX, int NY, int nt)
{
  int i, j, t;

  i = blockIdx.x* blockDim.x+ threadIdx.x;
  j = blockIdx.y* blockDim.y+ threadIdx.y;
  // j = blockIdx.x* blockDim.x+ threadIdx.x;
  // i = blockIdx.y* blockDim.y+ threadIdx.y;

  if (i == 0 || j == 0 || i >= NX-1 || j >= NY-1) return;

  for (t = 0; t < nt; t++) {
    int from = t%2;
    int to = (t+1)%2;

  data_gpu[NX*NY*to+i+j*NX] = 0.2 * (data_gpu[NX*NY*from+i+j*NX]
  			+ data_gpu[NX*NY*from+i-1+j*NX]
  			+ data_gpu[NX*NY*from+i+1+j*NX]
  			+ data_gpu[NX*NY*from+i+(j-1)*NX]
  			+ data_gpu[NX*NY*from+i+(j+1)*NX]);
  }
  return;
}

int  main(int argc, char *argv[])
{
  struct timeval tinb, tina, toutb, touta;
  int nt, i;
  cudaError_t rc;
  FILE *fp;
  char filename[50];

  if (argc != 5){
    printf("Specify Grid_nx, Grid_ny, Timesteps_nt, BlockSize_BS\n");
  }else{
    NX = atoi(argv[1]);
    NY = atoi(argv[2]);
    nt = atoi(argv[3]);
    BS = atoi(argv[4]);
    printf("nx=%d, ny=%d, nt=%d, BS=%d \n", NX, NY, nt, BS);
  }

  data = (float *)malloc(sizeof(float)*2*NX*NY);
  init(data);
  rc = cudaMalloc((void **)&data_gpu, sizeof(float)*2*NX*NY);
  if (rc != cudaSuccess) {
    fprintf(stderr, "cudaMalloc, failed\n"); exit(1);
  }

  sprintf(filename, "./output/diffusion_cuda_s_gflop_nx%d_ny%d_nt%d_BS%d.dat", NX, NY, nt, BS);
  fp = fopen(filename,"w");

  for (i = 0; i < 5; i++) {
    cudaDeviceSynchronize();
    gettimeofday(&tinb, NULL);
    rc = cudaMemcpy(data_gpu, data, sizeof(float)*2*NX*NY, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy, input failed\n"); exit(1);
    }
    cudaDeviceSynchronize();
    gettimeofday(&tina, NULL);

    dim3 grid = dim3((NX+BS-1)/BS, ((NY+BS-1)/BS), 1);
    dim3 block = dim3(BS, BS, 1);
    calc<<<grid, block>>>(data_gpu, NX, NY, nt);

    cudaDeviceSynchronize();
    gettimeofday(&toutb, NULL);
    cudaDeviceSynchronize();
    rc = cudaMemcpy(data, data_gpu, sizeof(float)*2*NX*NY, cudaMemcpyDeviceToHost);
    if (rc != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy, output failed\n"); exit(1);
    }
    cudaDeviceSynchronize();
    gettimeofday(&touta, NULL);
    {
        double us_in, us_out, us_calc;
        double gflops;
        int op_per_point = 5; // 4 add & 1 multiply per point

        us_in = time_diff_us(tinb, tina);
        us_in = us_in/1000000.0;
        us_calc = time_diff_us(tina, toutb);
        us_calc = us_calc/1000000.0;
        us_out = time_diff_us(toutb, touta);
        us_out = us_out/1000000.0;
        printf("Elapsed time for input: %.3lf sec\n", us_in);
        printf("Elapsed time for calc: %.3lf sec\n", us_calc);
        printf("Elapsed time for output: %.3lf sec\n", us_out);
        gflops = ((double)NX*NY*nt*op_per_point)/(us_in+us_calc+us_out)/1000000000.0;
        printf("Speed: %.3lf GFlops\n", gflops);
        fwrite(&us_in,8,1,fp);
        fwrite(&us_calc,8,1,fp);
        fwrite(&us_out,8,1,fp);
        fwrite(&gflops,8,1,fp);
    }
    init(data);
  }
  cudaFree(data_gpu);
  free(data);
  fclose(fp);
  return 0;
}
