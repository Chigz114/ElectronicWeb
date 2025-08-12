---
title: ECE408-lecture9
date: 2024-06-09 16:01:08
tags: ECE408
---

## Tiled Convolution Analysis

很简单的一节数学课，主要就是计算1D和2D的tiling方法对于global memory所需带宽减少比例的分析。

对于一维数组的tiling，给出如下公式：

**所需带宽减少比例 = (TILE_SIZE \* MASK_WIDTH) / (TILE_SIZE + MASK_WIDTH - 1)**

所有元素被用到的总次数：TILE_SIZE \* MASK_WIDTH。因为总共有tile个元素需要计算output，而每个output需要用到mask_width个元素，所以在不使用tiling的情况下，总共需要从global memory中调用TILE_SIZE \* MASK_WIDTH次元素。

而如果每个数据只从global memory中调用一次，那么就只需要调用TILE_SIZE + MASK_WIDTH - 1次，因为相对于从global memory中调用，从shared memory中调用所需的时间可以忽略不计。

把两者相除，得到的就是global memory所需带宽减少比例。

那么如果是边界的tiles呢？

即存在被忽略的边界值，ghost cell。这就没必要给出具体的公式了，结合上面的理解即可轻松得出。在分母减去本应调用ghost cell的次数，分子减去ghost memory的个数，即可得到答案。

下图给出了一张表格，对应在一些TILE_WIDTH和MASK_WIDTH情况下的比值：

<img src="https://s2.loli.net/2024/06/09/RIwUkhq38jmcWYJ.png" alt="image-20240609162633985" style="zoom:67%;" />

*

*

那么对于二维数组呢？

公式也是类似的：

**所需带宽减少比例 = (TILE_WIDTH \* MASK_WIDTH)^2 / (TILE_WIDTH + MASK_WIDTH - 1)^2**

原理与一维相同，不多赘述。

再给出一张表格，展示一些特定情况下的比值：

<img src="https://s2.loli.net/2024/06/09/X5wqyUbT8L6fmvG.png" alt="image-20240609162926849" style="zoom:67%;" />

*

*

再结合前几节课所学，具体考虑我们需要降低多少带宽使用。

在非tiling的卷积算法中，我们计算所需从global memory中读取的数据与所做的浮点数运算的比例。在每一次最小单位的运算中，我们先从global memory中读取数组N中的一个元素，大小为4Byte；再从constant memory中读取数组M的一个元素（因为它存储在cache中，大小可忽略）；然后将两个浮点数相乘，再加到sum中，完成两次浮点数运算。

所以，4B的global  memory对应2次浮点数运算（FLOP），比值为2B/FLOP。

假设GPU的运算速度为1000GFLOP/s（大约是2010年的性能），global memory的带宽为150GB/s，我们可以得到：

**(150GB/s) / (2B/FLOP) = 75GFLOP/s**，大约是GPU性能的7.5%，如果我们想利用GPU的全部性能，需要将数据的复用次数提高到100 / 7.5 = **13.3次**。

2020年，GPU运算速度达到5000GFLOP/s，但带宽仅有192GB/s，非tiling方法仅能达到性能的1.92%。如果想利用GPU的全部性能，需要将数据的复用次数提高到**52.1次**。

这就需要很大的MASK_WIDTH来平衡。
