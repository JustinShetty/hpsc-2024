#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void bucket_sort(int* key, int* bucket, int range) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id < range) {
    bucket[thread_id] = 0;	
  }
  __syncthreads();

  atomicAdd(&bucket[key[thread_id]], 1);
  __syncthreads();

  int lower = 0;
  for(int i=0; i < range; i++){
    int upper = lower + bucket[i];
    if (lower <= thread_id && thread_id < upper) {
      key[thread_id] = i;
    }
    lower = upper;
  }
}

int main() {
  int n = 50;
  int range = 5;
  int* key;
  int* bucket;
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  bucket_sort<<<1,n>>>(key, bucket, range);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
}
