---
title: ECE408-lecture8
date: 2024-06-09 13:56:30
tags: ECE408
---

## Tiled Convolution

与前几节课讲的矩阵乘法类似，卷积算法也可以用**tiling**的方法。如图展示了tiled 1D convolution：

<img src="https://s2.loli.net/2024/06/09/bKCafcjLh4F8mld.png" alt="image-20240609141815260" style="zoom:67%;" />

将需要卷积计算的数组拆分为小块分别运算。

与矩阵乘法类似的，我们也使用Shared Memory，因为其中的元素被多次调用，将其存入Shared Memory可以减少读取时间。

但是我们同时遇到一些问题。如上图，对于每个tile，threads的数量应与拆分后的数组元素相等。但我们还需要与之相邻的几个元素。这就使得，如果我们令threads的数量与数组元素数量（也就是output的数量）相同，那么就无法用每个thread一次读取一个global memory中的数值的方法，一次性读取所有需要存到shared memory中的元素。因为，如上图，还存在halo，也就是相邻的元素。

也就是说，输入所需的thread大于输出所需的thread。如果我们将thread数量设置为与输入所需的相等，又会造成在输出时的资源浪费，因为部分threads没有参与运算。

我们有3个解决方案：

1. 每个block的大小与output相同，用多步完成输入操作：

   <img src="https://s2.loli.net/2024/06/09/Ps2wCkj8uGmB6SU.png" alt="image-20240609142751036" style="zoom:50%;" />

2. 每个block的大小与input相同，在输出时闲置一部分threads：

   <img src="https://s2.loli.net/2024/06/09/4xXguOGpZFR7LlV.png" alt="image-20240609142853809" style="zoom:50%;" />

3. 每个block的大小与output相同，仅读取一次input，剩余的halos在计算输出时读取：

   <img src="https://s2.loli.net/2024/06/09/NMiWHYacEBvwUQ3.png" alt="image-20240609143028944" style="zoom:50%;" />

法一代码：

```c
__global__ void convolution_1D_tiled_kernel(float *N, float *P, int Mask_Width, int Width) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int radius = Mask_Width / 2;
	__shared__ float N_ds[TILE_SIZE + MAX_MASK_WIDTH - 1];
	int halo_index_left = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    // helo_index_left取上一个block中的元素
    // 如果该元素会被当前block的radius覆盖到
	if (threadIdx.x >= (blockDim.x – radius)) {
        // 将其录入当前block的shared memory，即输入左侧halos
        // 如果当前元素在整个数组的头尾，即ghost cell，则输入0
		N_ds[threadIdx.x - (blockDim.x - radius)] = (halo_index_left < 0) ? 0 : N[halo_index_left];
	}
    // 拷贝中间元素
	N_ds[radius + threadIdx.x] = N[i]; // bounds check is needed
    // 拷贝右侧元素
	int halo_index_right = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
	if (threadIdx.x < radius) {
		N_ds[radius + blockDim.x + threadIdx.x] = (halo_index_right >= Width) ? 0 : N[halo_index_right];
	}
    // 等待所有threads完成操作
	__syncthreads();
	float Pvalue = 0;
    // 计算当前点的output
	for(int j = 0; j < Mask_Width; j++) {
		Pvalue += N_ds[threadIdx.x + j]*M[j];
	}
	P[i] = Pvalue;
}
```

这个方法其实可以略作简化，只需要两步。第一次先拷贝左侧元素。第二次将左侧的一部分threads移到右侧，补上空缺即可：

```c
__global__ void convolution_1D_tiled_kernel float *N, float *P, int Width) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int radius = MASK_WIDTH / 2;
	int start = i – radius; 
    //将所有thread左移radius个元素
	__shared__ float N_ds[TILE_SIZE + MASK_WIDTH - 1];
    // 如果不是ghost cell，就读取到对应位置
    // 如果是ghost cell，就输入0
	if (0 <= start && Width > start) { // all threads
		N_ds[threadIdx.x] = N[start];
	else
		N_ds[threadIdx.x] = 0.0f;
    // 将最左边的MASK_WIDTH个threads右移block个单位，覆盖右边的所有元素
	if (MASK_WIDTH – 1 > threadIdx.x) { // some threads
		start += TILE_SIZE;
	if (Width > start) {
		N_ds[threadIdx.x + TILE_SIZE] = N[start];
	else
		N_ds[threadIdx.x + TILE_SIZE] = 0.0f;
	}
	__syncthreads(); 
	float Pvalue = 0.0f;
	for (int j = 0; MASK_WIDTH > j; j++) {
		Pvalue += N_ds[threadIdx.x + j] * Mc[j];
	}
	P[i] = Pvalue;
}
```

但这个方法有一个小问题，就是它的极限在Width = Radius - 1，小于第一个方法的Width = 2*(Radius-1)。不过，鉴于绝大多数情况下，Width的值都远大于Radius，这也并不算问题，且确实简化了程序。

接下来是法三，相对来说在思路上更直接：

```c
__global__ 
void convolution_1D_tiled_cache_kernel(float *N, float *P, int Mask_Width, int Width) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float N_ds[TILE_WIDTH];
	N_ds[threadIdx.x] = N[i];
    // 读取对应元素到shared memory
	__syncthreads();
	int radius = Mask_Width / 2;
    // 下面两个数据用于判断所取的节点是否在当前block对应元素中
	int This_tile_start_point = blockIdx.x * blockDim.x;
	int Next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
	int N_start_point = i - radius;
	float Pvalue = 0;
	for (int j = 0; j < Mask_Width; j ++) {
		int N_index = N_start_point + j;
		if (N_index >= 0 && N_index < Width) {
            // 先判断是不是ghost cell，如果是，直接跳过（等效于取0）
			if ((N_index >= This_tile_start_point) && (N_index < Next_tile_start_point)) 
				// 如果读取的点是shared memory
				Pvalue += N_ds[threadIdx.x-radius+j] * M[j];
			else 
                // 如果不是shared memory
				Pvalue += N[N_index] * M[j];
		}
	}
	P[i] = Pvalue;
}
```

*

然后重点在法二，毕竟力大砖飞，适用面也更广一些。这次我们以二维数组举例。

如图，对于一个tile（或者叫block），input会大于output

<img src="https://s2.loli.net/2024/06/09/h6xV7uUGtev2dRo.png" alt="image-20240609154914498" style="zoom:50%;" />

所以，在设置block时，需要加上MASK_WIDTH - 1：

```c
dim3 dimGrid(ceil(P.width/(1.0*TILE_WIDTH)), ceil(P.height/(1.0*TILE_WIDTH)), 1)
dim3 dimBlock(TILE_WIDTH + (MASK_WIDTH-1), TILE_WIDTH + (MASK_WIDTH-1), 1);
```

在计算output时，我们仍从第一个thread开始使用，而非忽略开头的部分，从中间开始使用，这样代码会相对简单一些：

```c
int tx = threadIdx.x;
int ty = threadIdx.y;
int row_o = blockIdx.y * TILE_WIDTH + ty;
int col_o = blockIdx.x * TILE_WIDTH + tx;
int row_i = row_o - MASK_RADIUS
int col_i = col_o - MASK_RADIUS
```

上面的代码实现将thread对应的元素向右下移动，形象化表达如图：

<img src="https://s2.loli.net/2024/06/09/WjeD7U3ZrsX64VY.png" alt="image-20240609155639712" style="zoom:50%;" />

同时，也要注意越界的点哦~（即ghost cell）

```c
float Pvalue = 0.0f;
if((row_i >= 0) && (row_i < Width) && (col_i >= 0) && (col_i < Width)) {
	N_ds[ty][tx] = N[row_i*Width + col_i];
} 
else {
	N_ds[ty][tx] = 0.0f;
}
__syncthreads (); // wait for tile
```

而且，因为threads数量多于output，部分threads不需要输出：

```c
if(row_o < Width && col_o < Width) P[row_o * Width + col_o] = Pvalue;
```

