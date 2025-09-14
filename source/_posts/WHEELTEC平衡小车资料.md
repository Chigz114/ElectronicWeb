---
title: WHEELTEC平衡小车资料
date: 2025-1-18 22:41:16
categories: 平衡小车
tags: 平衡小车
top_img: https://img.alicdn.com/i4/2214283073185/O1CN01DUaXAl1ZOmI9WZHla_!!2214283073185.jpg
cover: https://img.alicdn.com/i4/2214283073185/O1CN01DUaXAl1ZOmI9WZHla_!!2214283073185.jpg
---

## WHEELTEC平衡小车资料及导读

**百度网盘链接：**https://pan.baidu.com/s/1pY5abM-OU2H3nZD_TStqTQ?pwd=1ubr 提取码: 1ubr

**原理图：**基本只需使用S28A转接板资源分配说明。上面标明了各引脚的功能，在配置STM32的引脚功能时请参照该PDF。

**平衡小车32底层源码：**我写的MP是基于江协科技的思路写的，所以会更贴近卖家给的库函数（精简版）编程。但因为卖家的程序涉及到很多我们不需要的功能函数，所以他们做了一级比较复杂的函数封装，导致初读程序可能找不到重点。并不需要着急，看注释，找到相关函数再跳转函数定义慢慢看即可^_^。



### 资料使用指引（潘枫岚编）

先打开网盘中的WHEELTEC B570平衡小车附送资料”文件夹，找到B570学习指引.txt，打开。

![image-20250228134747196](https://s2.loli.net/2025/02/28/RbiHASNxtPgolWQ.png)

可以根据里面的内容学习。其中开机和模式调节的直观视频，请看：

![image-20250228134812342](https://s2.loli.net/2025/02/28/rb63HATZetEly8f.png)

在总文件夹中找到：

![image-20250228134836927](https://s2.loli.net/2025/02/28/3ueQ1EKTJRNOX5k.png)

其中的1就是详细的演示了。提示：视频中的红色开关和我们的小车不一样，用充电线给电源充电亮绿灯之后记得把这根线插上，作用是给芯片供电。

![image-20250228134900187](https://s2.loli.net/2025/02/28/hipYT5jJaOxsWtL.png)

然后找到这个开关即可。

![image-20250228134916070](https://s2.loli.net/2025/02/28/jXTkvMSQIFLeUKR.png)

还有一个提示，在地上进行平衡模式（普通模式，normal）时，小车被拿起来后轮子会疯狂转，记得在拿起来之前按user按键停止。还有，视频中调好模式按一次user按键之后即启动，但实操中发现要多按一次。
