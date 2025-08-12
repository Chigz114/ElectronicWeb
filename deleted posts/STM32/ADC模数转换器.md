---
title: ADC模数转换器
date: 2024-06-12 14:52:51
tags: STM32
---

## ADC模数转换器

ADC可以将引脚上连续变化的模拟电压转换为内存中存储的数字变量，建立模拟电路到数字电路的桥梁。

12位逐次逼近型ADC，1us转换时间（从AD转换开始到产生结果需要1us，即转换频率为1MHz）。输入电压范围：0~3.3V，转换结果范围：0~4095。18个输入通道，可测量16个外部和2个内部信号源。

STM32F103C8T6 ADC资源：ADC1、ADC2，10个外部输入通道。

![image-20240612150619242](https://s2.loli.net/2024/06/12/UhAxsqvn9bG2M6c.png)

ADC共有IN0~IN15，16个端口。

对于**规则通道**，可以设置最多16个ADC端口，它们将被依次读取并转换为16位的数据（0~4095），存储到规则通道数据寄存器中。但规则通道数据寄存器只有16位，即只有最后一个数据会被存储。所以，需要配合DMA（数据转运）来实现。

对于**注入通道**，读取的最大数据量与寄存器储存的最大数据量相同，即不需要配合DMA，可以存储所有数据。

在ADC转换完成后，就会置转换结束标志位，用于中断输出控制。

下方有“开始触发”控制，确定哪一个标志位用于AD转换的开始触发。

### ADC基本结构：

<img src="https://s2.loli.net/2024/06/12/cl53r79LNoUDyTI.png" alt="image-20240612151543485" style="zoom:67%;" />

### ADC输入通道引脚复用：

<img src="https://s2.loli.net/2024/06/12/GNK65ezlVIHAWps.png" alt="image-20240612151645943" style="zoom:50%;" />

### 转换模式

#### 单次转换，非扫描模式：

<img src="https://s2.loli.net/2024/06/12/MQGXOIp9CWuqeLH.png" alt="image-20240612151929983" style="zoom:50%;" />

单次触发指定通道的ADC转换，将转换得到的数据存储到寄存器中；转换完成后，置EOC标志位，表示转换完成。如果需要再次转换，则需要再次触发。

#### 连续转换，非扫描模式：

<img src="https://s2.loli.net/2024/06/12/ECs2fydOBWqUFXa.png" alt="image-20240612152151218" style="zoom:50%;" />

同样是只扫描一个通道，但不需要再次触发，而是在识别到EOC标志位后就进行下一次转换。

#### 单次转换，扫描模式：

<img src="https://s2.loli.net/2024/06/12/DbhSCnVal2TKOPp.png" alt="image-20240612152323629" style="zoom:50%;" />

序列中的通道可以任意指定，且可以重复。需要设定通道数目，表示需要读取的信号数量。在一轮扫描完后，置EOC信号，结束转换。

注意，因为使用的是扫描模式，所以需要用DMA及时移走寄存器中的数据，防止被覆盖。

#### 连续转换，扫描模式：

<img src="https://s2.loli.net/2024/06/12/ORLldUG93vYSatJ.png" alt="image-20240612152637866" style="zoom:50%;" />

### 数据对齐：

转换得到的电压值是2^12，即12位数据，但数据寄存器是16位的，所以需要数据对齐。

数据右对齐：高位补0。

数据左对齐：低位补0。

通常都用右对齐。

### 转换时间：

AD转换步骤：采样，保持，量化，编码

STM32 ADC总转换时间：T = 采样时间 + 12.5个ADC周期

例如，当ADCCLK = 14MHz时，采样时间为1.5个ADC周期，T = 1.5 + 12.5 = 14个ADC周期

### ADC初始化：

```c
void AD_Init(void)
{
	/*开启时钟*/
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_ADC1, ENABLE);	//开启ADC1的时钟
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);	//开启GPIOA的时钟
	
	/*设置ADC时钟*/
	RCC_ADCCLKConfig(RCC_PCLK2_Div6);						//选择时钟6分频，ADCCLK = 72MHz / 6 = 12MHz
	
	/*GPIO初始化*/
	GPIO_InitTypeDef GPIO_InitStructure;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AIN;
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_0;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOA, &GPIO_InitStructure);					//将PA0引脚初始化为模拟输入
	
	/*规则组通道配置*/
	ADC_RegularChannelConfig(ADC1, ADC_Channel_0, 1, ADC_SampleTime_55Cycles5);		//规则组序列1的位置，配置为通道0
	
	/*ADC初始化*/
	ADC_InitTypeDef ADC_InitStructure;						//定义结构体变量
	ADC_InitStructure.ADC_Mode = ADC_Mode_Independent;		//模式，选择独立模式，即单独使用ADC1
	ADC_InitStructure.ADC_DataAlign = ADC_DataAlign_Right;	//数据对齐，选择右对齐
	ADC_InitStructure.ADC_ExternalTrigConv = ADC_ExternalTrigConv_None;	//外部触发，使用软件触发，不需要外部触发
	ADC_InitStructure.ADC_ContinuousConvMode = DISABLE;		//连续转换，失能，每转换一次规则组序列后停止
	ADC_InitStructure.ADC_ScanConvMode = DISABLE;			//扫描模式，失能，只转换规则组的序列1这一个位置
	ADC_InitStructure.ADC_NbrOfChannel = 1;					//通道数，为1，仅在扫描模式下，才需要指定大于1的数，在非扫描模式下，只能是1
	ADC_Init(ADC1, &ADC_InitStructure);						//将结构体变量交给ADC_Init，配置ADC1
	
	/*ADC使能*/
	ADC_Cmd(ADC1, ENABLE);									//使能ADC1，ADC开始运行
	
	/*ADC校准*/
	ADC_ResetCalibration(ADC1);								//固定流程，内部有电路会自动执行校准
	while (ADC_GetResetCalibrationStatus(ADC1) == SET);
	ADC_StartCalibration(ADC1);
	while (ADC_GetCalibrationStatus(ADC1) == SET);
}
```

