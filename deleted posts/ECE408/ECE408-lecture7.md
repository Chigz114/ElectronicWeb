---
title: ECE408-lecture7
date: 2024-06-08 22:47:31
tags: ECE408
---

## Convolution, Constant Memory and Constant Caching

convolution指的是一种数组操作，每个输出值都是与其邻近的输入值的加权和。

如图是一个一维卷积运算的例子：

<img src="https://s2.loli.net/2024/06/08/zUsNqxpAuK682jd.png" alt="image-20240608234757047" style="zoom:67%;" />

图中的权重是数组M，**M中的值是预先设定的**，决定处理N中数据时给定的权重。

其中有两个参数，一个是MASK_WIDTH，表示参与卷积运算的参数个数，该例中是5；另一个是MASK_RADIUS，表示每一边有几个数据参与运算，该例中是2。

如果涉及到边界情况，可能是补充0，也可能是复制边界值，等等。据具体情况而定。此处给出一个一维数组的运算样例，在边界条件填充0。

```c
__global__ void 
convolution_1D_basic_kernel(float *N, float *M, float *P, int Mask_Width, int Width)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	float Pvalue = 0;
	int N_start_point = i - (Mask_Width/2);
	for (int j = 0; j < Mask_Width; j++) {
		if (((N_start_point + j) >= 0) && ((N_start_point + j) < Width)) {
			Pvalue += N[N_start_point + j]*M[j];
            //对于越界的数值，直接忽略，等效于取0
		}
	}
	P[i] = Pvalue;
}
```

但是我们发现，这样计算出来的值会比原数值大很多，所以需要对得到的结果normalize。方法也很简单，通常是**将mask中的所有权重加和，作为除数**。

但在遇到出现越界点的时候，如果仍然用mask中的全部权重作为除数来normalize，可能使值小于它应有的大小。但具体的解决方案需要根据实际需要分析，可能不改，也可能根据具体位置取相应的权重。

对于二维数组，样例如下：

<img src="https://s2.loli.net/2024/06/08/2CaAjyGn5U8dTlv.png" alt="image-20240608235445697" style="zoom:67%;" />

*

*

Constant Memory使用高速缓存（cache），读取速度远高于global memory。当数据被调用后，会存储到cache中。只有当cache存满后，会用新数据覆盖之前的数据。

所以，当我们需要重复调用相同的数据，或者调用地址连续的数据（前提是constant memory），就会从cache中读取，进而提高读取速度。

<img src="https://s2.loli.net/2024/06/09/C8UWnKfDxugH3YT.png" alt="image-20240609103513254" style="zoom:67%;" />

与Shared Memory不同的是，cache中的数据是系统自动存储的。
