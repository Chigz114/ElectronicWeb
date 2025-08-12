---
title: ECE408-lecture2
date: 2024-06-05 12:07:34
tags: ECE408
---

CUDA中一个grid由一个三维数组的thread blocks组成；每一个block又由三维数组的threads组成；同一个grid中的每个thread都执行相同的任务。每个thread的输入数据不同，但它们的处理方式相同。（single-program, multiple-data model）

每一个block，都有一个独特的编号blockIdx，由x，y，z三维的数据组成。

而每一个block内的thread，也有一个独特的编号blockDim，也由x，y，z三维数据组成。

***注意，因为是数组，每个下标都从0开始

