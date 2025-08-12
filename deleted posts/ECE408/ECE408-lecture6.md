---
title: ECE408-lecture6
date: 2024-06-08 15:16:53
tags: ECE408
---

## Generalized Tiling & DRAM Bandwidth

上一节中，我们用shared memory加速了并行运算，但是我们假设了Width恰好是Width_tile的整数倍。如图，在更一般的情况中，边界blocks可能出现空缺：

<img src="https://s2.loli.net/2024/06/08/OoIcYx6vF5tfBZ1.png" alt="image-20240608160053940" style="zoom:67%;" />

对于一般情况，我们需要处理边界blocks。解决方法就是，在从global memory中拷贝数据时，如果发现越界，就将数据设置为0。对于该例，这一做法是合理的。而且，除此之外，我们无需其它操作。

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
	// 对于非TILE_WIDTH整数倍的情况，得到向下取整的结果，并+1
    // 对于整数倍的情况，因为Width-1，得到的结果小了1，所以在后面+1
	for (int q = 0; q < (Width - 1)/TILE_WIDTH + 1; ++q) {
		// Collaborative loading of M and N tiles into shared memory
        // 对于M，判断当前点是否在界内
        if(Row < Width && m * TILE_WIDTH + tx < Width){
            subTileM[ty][tx] = M[Row*Width + q*TILE_WIDTH+tx];
        }
        else{
            subTileM[ty][tx] = 0;
        }
        // 对于N，判断当前点是否在界内
        if(m * TILE_WIDTH + ty < Width && Col < Width){
            subTileN[ty][tx] = N[(q*TILE_WIDTH+ty)*Width+Col];
        }
        else{
            subTileN[ty][tx] = 0;
        }
		for (int k = 0; k < TILE_WIDTH; ++k)
			Pvalue += subTileM[ty][k] * subTileN[k][tx];
	}
    if(Row < Width && Col < Width){
        P[Row*Width+Col] = Pvalue;
    }	
}
```

修改了q的for循环，增加了拷贝M、N数据时的边界判断，增加了将结果拷贝回global memory时的边界判断。实现更一般情况下的矩阵运算。

*

*

先做了一个概念引入，我们常用的数据缓存，存在DRAM中（Dynamic Random Access Memory）。为了让数据存储更密集，它减小了存储数据的电容，这使得读取数据变得缓慢。DRAM在图中是Memory Cell Core Array：

<img src="https://s2.loli.net/2024/06/08/fVdX1bJpyjMD9G2.png" alt="image-20240608171730460" style="zoom:67%;" />

其中的数据在需要使用时，会分块，通过Sense Amps放大，传输到Column Latches中，再通过Mux选择需要的数据。

但因为从Core Array中读取数据非常缓慢，我们将其分为大量的banks，并行读写，提高效率。同时，从一个bank中，我们一次读取多个**相邻的**数据，再从中选择需要用的。这种一次读取多个相邻数据的方法叫**bursting**。当我们需要读取的数据在Core Array中的位置相邻时，这一方法可以大大提高读取速度。

在矩阵的例子中，我们需要读取的数据就是相邻的，用burst可以大大提高读取速度。

<img src="https://s2.loli.net/2024/06/08/dhGJNFvamxrbgtK.png" alt="image-20240608172556426" style="zoom:67%;" />

但是，在矩阵的例子中，当我们需要把M、N中相邻的行和列的数据读到threads中时，效率的差别很大。

<img src="https://s2.loli.net/2024/06/08/dV5kXJzGZpvLNjh.png" alt="image-20240608172846586" style="zoom:50%;" />

如图，在M中，我们读取的数据是在同一列，不同行的。它们在地址上不相邻，所以读取要花的时间相对长很多。

而在N中，读取的数据在同一行，地址上相邻，所以读取所需的时间相对较短。

具体如下图：

<img src="https://s2.loli.net/2024/06/08/vGZKECR8QwiTHmY.png" alt="image-20240608173114787" style="zoom:67%;" />

<img src="https://s2.loli.net/2024/06/08/HrvGDelIXpVCgyZ.png" alt="image-20240608173135853" style="zoom:67%;" />

