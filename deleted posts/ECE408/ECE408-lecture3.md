---
title: ECE408-lecture3
date: 2024-06-08 10:13:49
tags: ECE408
---

举了一个图像处理的例子，从彩色图像变为黑白：

<img src="https://s2.loli.net/2024/06/08/wPpDgM4umSz1Ex5.png" alt="image-20240608102036764" style="zoom:67%;" />

一张76 x 62像素的图片，每16 x 16方格被归为一个block，用二维blocks表示。

```c
__global__ 
void colorToGreyscaleConversion(unsigned char * grayImage, unsigned char * 					rgbImage, int width, int height) 
{
	int Col = threadIdx.x + blockIdx.x * blockDim.x;
	int Row = threadIdx.y + blockIdx.y * blockDim.y;
	if (Col < width && Row < height) {
		// 因为长宽不是16的整数倍，所以需要判断像素点是否在界内
		int greyOffset = Row*width + Col;
		// RGB有黑白图像3倍的columns
		int rgbOffset = 3 * greyOffset;
		unsigned char r = rgbImage[rgbOffset ]; // red value for pixel
		unsigned char g = rgbImage[rgbOffset + 1]; // green value for pixel
		unsigned char b = rgbImage[rgbOffset + 2]; // blue value for pixel
		// 将RGB的值分开存储并通过公式计算对应的灰度值
		grayImage[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
	}
}
```

第二个例子，图像模糊处理。原理：将需要处理的像素点与其相邻的8个像素点的数据值做平均。在边界点处，忽略越界的相邻点。

代码样例：

```c
__global__
void blurKernel(unsigned char * in, unsigned char * out, int w, int h) {
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	if (Col < w && Row < h) {
		int pixVal = 0;
		int pixels = 0;
		for(int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
			for(int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {
                // for循环遍历周围像素点
                // 此处用BLUR_SIZE代替相邻1个像素点，将问题一般化，提高代码可移植性
				int curRow = Row + blurRow;
				int curCol = Col + blurCol;
				if(curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                    // 确定当前像素点在边界内
					pixVal += in[curRow * w + curCol];
					pixels++; // 考虑到边界问题，需要确定当前像素点周围点个数
				}
			}
		}
		// 输出新的像素值
		out[Row * w + Col] = (unsigned char)(pixVal / pixels);
	}
}
```

*

执行任务时，threads以block为单位，被分配到SM（Streaming Multiprocessor）中。而每个SM的处理能力，由GPU的性能决定。如Maxwell，一个SM可以容纳32个blocks，2048个threads。

每一个block在SM中处理时，又以32个threads为单位（通常是32个）被分为wraps。wrap的分配基于thread的序号。如：thread 0~31被分配到wrap 0，thread 32~63被分配到wrap 1。如果3个blocks被分配到1个SM，且每个block有256个threads，那么每个block就被分为256/32 = 8 wraps，该SM中就有24个wraps。

*

注意，因为同一个wrap中的threads都会经过相同的程序，所以在条件判断中，即使一部分thread中的数据在该判断中为false，也会经过当前分支。只是该thread在这段判断中被设定为false，所以不执行。**但若该warp中的所有thread对当前条件分支的判断均为false，就会直接跳过该分支**。

所以，为了使程序执行尽可能快，**尽量把条件判断做成warp大小的整数倍**，这样，在一部分warp中，所有thread都不执行一个条件分支，就会跳过该分支，进而加快程序执行速度。

*

为了充分利用SM的性能，我们要尽可能用到SM中的每一个thread。但我们也知道，需要以block为单位执行程序，所以，当SM中的最大threads数量不是一个block中threads数量的整数倍时，就无法充分利用。

在如下例子中，一个SM最多可以同时接纳1536个threads，8个blocks，而block可以被分配为8 x 8，16 x 16，32 x 32，那么，哪一种block可以使该SM被最充分利用呢？

-对于8 x 8的block，1536/64 = 24blocks，但每个SM只能同时接收8个blocks，只能同时执行512个threads。

-对于16 x 16的block，1536/256 = 6blocks，可以充分利用SM。

-对于32 x 32的block，只能同时执行1024个threads。

所以16 x 16的方案是这三个中最优的。
