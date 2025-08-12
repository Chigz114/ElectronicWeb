---
title: USART串口外设
date: 2024-06-12 17:01:43
tags: STM32
---

## USART串口外设

• USART（Universal Synchronous/Asynchronous Receiver/Transmitter）通用同步/异步收发器

• USART是STM32内部集成的硬件外设，可根据数据寄存器的一个字节数据自动生成数据帧时序，从TX引脚发送出去，也可自动接收RX引脚的数据帧时序，拼接为一个字节数据，存放在数据寄存器里

• 自带波特率发生器，最高达4.5Mbits/s

• 可配置数据位长度（8/9）、停止位长度（0.5/1/1.5/2）

• 可选校验位（无校验/奇校验/偶校验）

• 支持同步模式、硬件流控制、DMA、智能卡、IrDA、LIN

• STM32F103C8T6 USART资源： USART1（APB2）、 USART2（APB1）、 USART3（APB1）

### USART框图：

<img src="https://s2.loli.net/2024/06/12/HEzNyYQpTkMm1Vg.png" alt="image-20240612172143832" style="zoom:80%;" />

TDR和RDR在硬件中占用同一个地址，都表示为DR。在执行写操作时，使用TDR，执行读操作时，使用RDR。

移位寄存器：于将数据一位一位地写出或读入。向右移位，与“低位先行”的规则一致。

硬件数据流控：防止数据发送过快，出现丢弃或覆盖数据的情况。

唤醒单元：用于实现多设备串口通信，使能/失能通信模块。

USART接口参照引脚复用表，如USART1用PA9he。

### USART基本结构：

<img src="https://s2.loli.net/2024/06/12/dYGDasyzAE29lo5.png" alt="image-20240612173405361" style="zoom:50%;" />
