---
title: MP1：驱动OLED--6针脚改4针脚
date: 2024-11-22 22:15:53
categories: 平衡小车
tags: 平衡小车
top_img: https://img.alicdn.com/i4/2214283073185/O1CN01DUaXAl1ZOmI9WZHla_!!2214283073185.jpg
cover: https://img.alicdn.com/i4/2214283073185/O1CN01DUaXAl1ZOmI9WZHla_!!2214283073185.jpg
---

## MP1：驱动OLED——6针脚改4针脚

　　在任务开始之前，请先下载Wheeltec提供的源码和各种技术支持文件：https://pan.baidu.com/s/1igIu6VU-7f1i702oJFCTrQ

　　OLED是显示单片机目前工作状态最为直观的方式。所以，驱动程序编写的第一步，就是OLED屏幕。基于显示屏，后续的调试都会方便很多。

　　本MP的目标是，将江协科技中4针脚的软件模拟I2C驱动OLED程序，移植到平衡小车的硬件配置中。

　　**在查看下方的Solution之前，请先尝试自行配置。**

　　**当然，你们如果想沿用六针脚OLED屏幕，也可以尝试自行理解源码。**







　　查看S28A底板资源分配（在网盘文件中）不难发现，OLED屏幕用到了PA15，PB3，PB4，PB5，共4个引脚。其中，与4针脚OLED相同的是PB4和PB5，SCL时钟线是PB5，SDA数据线是PB4。查看STM32F103C8T6引脚定义表可以知道，PB4和PB5不是标准的硬件I2C数据线。同时查询源码可知，使用的是软件I2C。这与江协科技的软件I2C代码可以轻松代换。

　　移植过程中的唯一难点：在复位后，PB4默认作为调试接口（JTAG/SWD）的JTDO信号。这意味着，除非明确将JTAG调试功能禁用，或者将引脚重新配置为GPIO，否则PB4会被占用而导致无法用作普通的GPIO引脚。要解决这个问题，需要在OLED_I2C_Init()函数中加入如下两行代码，使能复用时钟功能，禁用JTAG，只保留SWD。

```c
RCC_APB2PeriphClockCmd(RCC_APB2Periph_AFIO, ENABLE); // 使能复用功能时钟
GPIO_PinRemapConfig(GPIO_Remap_SWJ_JTAGDisable, ENABLE); // 禁用 JTAG，只保留 SWD
```

　　另外，请将全部的GPIO_Pin_8和GPIO_Pin_9改为GPIO_Pin_4和GPIO_Pin_5。

　　最后进行屏幕点亮测试。

　　至此，理论上任务完成。
