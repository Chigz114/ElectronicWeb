---
title: ECE408-lecture5
date: 2024-06-08 14:05:32
tags: ECE408
---

## Locality and Tiled Matrix Multiplication

书接上回，为了简化matrix multiplication，我们使用了tiling。为了使SM被充分调用，我们需要减少使用global memory。在此例中，每个global memory都被多次调用，所以我们可以将global memory拷贝到每个block的shared memory中，减少调用global memory的次数，进而加快并行运算。

***该例中，tiles与blocks含义相同

具体的操作过程如下（以运算block(0, 0)为例）：

1. 先将M和N中block(0, 0)的数据都存储到shared memory中

<img src="https://s2.loli.net/2024/06/08/1wn6VMOLxsPAloB.png" alt="image-20240608142629208" style="zoom:50%;" />

2. 再将N中第一行和M中第一列相关的数据相乘，存储到P中对应的位置

   <img src="https://s2.loli.net/2024/06/08/af8NTyqS3UhLO7c.png" alt="image-20240608143114021" style="zoom:50%;" />

3. 再将N中第二行和M中第二列相关的数据相乘，存储到P中对应的位置（注意是与上一步的数据加和）

   <img src="https://s2.loli.net/2024/06/08/LskPCS4iQftz7bj.png" alt="image-20240608142955203" style="zoom:50%;" />

4. 再将N中的block(1, 0)和M中的block(0, 1)存储到shared memory中，并重复上述步骤

   <img src="https://s2.loli.net/2024/06/08/BJmha7r35RedEbH.png" alt="image-20240608143554847" style="zoom:50%;" />

整体的操作过程如图：

<img src="https://s2.loli.net/2024/06/08/ZcAqXNYavxGnECz.png" alt="image-20240608143700409" style="zoom:67%;" />

对于P中的一个block，M和N中分别有一行和一列blocks与之相关。所以我们就每次从M和N中取一对相关的blocks，将运算结果存储到P中对应的位置，这样就大大减少了对global memory的调用。

同时，注意到M和N都是动态分配地址的，所以只能使用一维地址。因而，对其地址的表达应该是：

```c
M[Row][m*TILE_WIDTH+tx]				// 形象表达
M[Row*Width + q*TILE_WIDTH + tx]	// 实际表达
N[q*TILE_WIDTH+ty][Col]
N[(q*TILE_WIDTH+ty) * Width + Col]
```

*

另一个问题：如何使并行程序同步呢？

设置barrier。在一些位置设置barrier，只有当所有threads都执行完前置的程序后，所有threads才会继续执行后面的程序。在CUDA中，这个函数是`__syncthreads()`。

在该例中的应用：

```c
__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width)
{
	__shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
	__shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x; 
    int by = blockIdx.y;
	int tx = threadIdx.x; 
    int ty = threadIdx.y;
    // Identify the row and column of the P element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;
	// Loop over the M and N tiles required to compute the P element
	// The code assumes that the Width is a multiple of TILE_WIDTH!
	for (int q = 0; q < Width/TILE_WIDTH; ++q) {
		// Collaborative loading of M and N tiles into shared memory
		subTileM[ty][tx] = M[Row*Width + q*TILE_WIDTH+tx];
		subTileN[ty][tx] = N[(q*TILE_WIDTH+ty)*Width+Col];
		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k)
			Pvalue += subTileM[ty][k] * subTileN[k][tx];
		__syncthreads();
	}
	P[Row*Width+Col] = Pvalue;
}
```

这样，我们就大大减少了global memory的使用！
