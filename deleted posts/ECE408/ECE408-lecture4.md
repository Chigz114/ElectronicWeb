---
title: ECE408-lecture4
date: 2024-06-08 11:33:42
tags: ECE408
---

## Memory Model

对于thread，读写不同的memory，所需的时间是不同的：

-读写thread对应的registers时，需要约1个时钟周期

-读写block对应的shared memory时，需要约5个时钟周期

-读写grid对应的global memory时，需要约500个时钟周期

-读（不能写）grid对应的constant memory时，需要约5个时钟周期（with caching）

![image-20240608113912597](https://s2.loli.net/2024/06/08/YAOsZdT6lXfzEMq.png)

*

举个例子：矩阵乘法

<img src="https://s2.loli.net/2024/06/08/vAWbwmpOloRyDnF.png" alt="image-20240608130257943" style="zoom:50%;" />

<img src="https://s2.loli.net/2024/06/08/aGKxB12lw8MskgN.png" alt="image-20240608130342961" style="zoom:67%;" />

如图，我们将M中第i行的第k个元素与N中第j列第k个元素相乘，k的取值为0~width-1，将所有k得到的乘积加和，存到P的第i行，第j列的位置。

如果不考虑并行算法，我们可以简单使用如下代码：

```c
void MatrixMul(float *M, float *N, float *P, int Width)
{ 
	for (int i = 0; i < Width; ++i)
		for (int j = 0; j < Width; ++j) {
			float sum = 0;
			for (int k = 0; k < Width; ++k) {
				// 将对应元素相乘，加和，并存到P中的i，j位置
				float a = M[i * Width + k];
				float b = N[k * Width + j];
				sum += a * b;
			}
			P[i * Width + j] = sum;
		}
}
```

如果要对于前两个for循环应用并行计算，是相对简单的。但如果要对for循环也应用并行计算，就困难了。因为，我们在该例中，用到了浮点数。而理论上说，浮点数之间是not associative的。（这里我结合AI，大致的理解是：由于浮点数存在精度的区别，如果改变运算顺序，会因为误差的取舍而改变精度，影响最终结果，所以不能使用并行运算。）

并行的加和parallel sum，称为一种reduction。（reduction也是一个很怪的词，对此我的理解是：在一些多数据的运算中，通过减少每次运算涉及的变量，使之可以并行运算；这种减少涉及变量的做法被称为reduction）

所以现在，我们仅对两个外循环用并行算法，而在内循环用串行。

*

为了对外循环用并行算法，我们把矩阵分为小矩阵，每个小矩阵计算结果的一部分。每个小矩阵被划分为一个block，进而在一个grid中用二维的blocks，并在block中用二维的threads来表达一个矩阵。这种方法被称为tiling，每个block代表一个block。例如：

<img src="https://s2.loli.net/2024/06/08/HXsRJYcUzLuQ4P9.png" alt="image-20240608135503004" style="zoom:67%;" />

内核调用程序：

```c
// TILE_WIDTH is a #define constant
dim3 dimGrid(ceil((1.0*Width)/TILE_WIDTH), ceil((1.0*Width)/TILE_WIDTH), 1);
dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
// Launch the device computation threads!
MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, Width);
```

一个简单的矩阵乘法程序：

```c
__global__ 
void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int Width) 
{
	// Calculate the row index of the d_P element and d_M
	int Row = blockIdx.y*blockDim.y+threadIdx.y;
	// Calculate the column idenx of d_P and d_N
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	if ((Row < Width) && (Col < Width)) {
		float Pvalue = 0;
		// each thread computes one element of the block sub-matrix
		for (int k = 0; k < Width; ++k)
            // 下面两行调用了global memory，会拖慢程序运行
			Pvalue += d_M[Row*Width+k] * d_N[k*Width+Col];
		d_P[Row*Width+Col] = Pvalue;
	}
}
```

因为到global memory的带宽有限，极大影响了每个thread执行程序。所以，我们需要减少访问global memory。（但这块似乎不是这节课的内容）
