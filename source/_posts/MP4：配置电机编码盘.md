---
title: MP4：配置电机编码盘
date: 2025-01-21 15:16:08
categories: 平衡小车
tags: 平衡小车
top_img: https://img.alicdn.com/i4/2214283073185/O1CN01DUaXAl1ZOmI9WZHla_!!2214283073185.jpg
cover: https://img.alicdn.com/i4/2214283073185/O1CN01DUaXAl1ZOmI9WZHla_!!2214283073185.jpg
---

## MP4：配置电机编码盘

　　得到卡尔曼滤波之后的角度值之后，需要将其反馈为对应的电机转速。倾斜角度越大，对应电机的转速变化量也应该越大，来提供足够的加速度使小车整体回正。而控制电机转速，首先需要由传感器得到电机的转速。

　　关于编码盘如何得到电机转速，此处不再赘述。如有疑问，请回看江协科技6-7、6-8[[6-7\] TIM编码器接口_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1th411z7sn?spm_id_from=333.788.videopod.episodes&vd_source=21b78eb73d808f842c3ff2485f3933ab&p=19)。

　　本MP的目标是，将江协科技的TIM编码器程序移植到小车上，使之能读取小车电机的转速，并显示在OLED屏幕上。使用的是TIM时钟的编码盘计数功能。

　　如何测试？无需设置程序让电机转动，只需手动转动轮子，即可通过编码盘获取读数。

　　**理论上至此可以自行查阅相关文档完成任务，请先自行尝试，若仍遇到问题，请继续阅读后续内容。**



**整体配置思路：**

①查看小车引脚定义表，找到电机编码盘对应的引脚和时钟。

②调用江协科技`Encoder.c`文件，并根据小车的实际引脚和需求做配置修改。（如时钟分频、GPIO使能等）

③调用时钟寄存器数据采集函数`Read_Encoder()`并根据小车实际使用的TIM时钟做配置修改。

④在main函数中使能TIM编码盘接口功能，并定时循环读取。

**下面是详细操作：**

　　查看S28A底板资源分配可以发现，两个电机的编码盘分别接在了PB6、PB7，PA6、PA7，还贴心地说明了它们分别属于TIM4时钟和TIM3时钟的引脚。因此在写程序时请注意将Encoder_Init函数中的相关配置写对。![image-20250121141959220](https://s2.loli.net/2025/01/21/WXjKqCeVHd2Emgw.png)

　　此外，注意到分别是PA和PB，请注意在使能GPIO时钟时对应使能GPIOA和GPIOB。

　　因为速度读取的精度越高越好，故不建议设置过高的预分频系数。只需在每次读取编码盘数值后将其清零即可。在WHEELTEC给出的源码中，预分频系数设置为0，自动重装值设定为65535（因为STM32F103系列的时钟寄存器是16位的）。

　　对于TIM3和TIM4，需要分别配置Init函数，并使能。

　　对于两个时钟，也需要分别配置Encoder_Get函数。关于这一函数的书写，建议使用江协科技的写法。WHEELTEC给出的写法集成了多个TIM接口的读取，但需要对C语言的结构体、switch case有所了解。

　　接下来需要对两个TIM接口得到的的编码盘数值做定时读取并清零。因为目前仍未配置统一的定时器中断，所以请先在main函数的while循环内使用Delay函数做定时读取功能。建议配置为20ms读取一次（即50Hz）。

　　最后，请不要忘记在main函数中调用两个Encoder_Init函数。

​	



**一些其它发现：**

①对比WHEELTEC和江协的文档可以发现，江协科技在配置时显式配置了Channels：

```c
TIM_ICInitStructure.TIM_Channel = TIM_Channel_1;
TIM_ICInitStructure.TIM_Channel = TIM_Channel_2;
```

但这与编码器接口配置函数`TIM_EncoderInterfaceConfig`似乎是重复的。在WHEELTEC的文档中就没有这一配置。

②WHEELTEC在文件中配置了TIM3和TIM4的`IRQHandler`函数，来处理可能的溢出中断。但这属于对于特殊情况的处理。只要程序运行正常，采样间隔配置合理，通常不会出现溢出情况，且溢出并不会造成严重后果。

