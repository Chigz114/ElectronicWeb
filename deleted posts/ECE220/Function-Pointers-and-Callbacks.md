---
title: Function_Pointers_and_Callbacks
date: 2024-05-21 14:20:32
tags: ECE220
top_img: https://patchwiki.biligame.com/images/blhx/c/cf/6tj18cxvk0h6rh0s3l2nczv8zggspm1.png
cover: https://patchwiki.biligame.com/images/blhx/c/cf/6tj18cxvk0h6rh0s3l2nczv8zggspm1.png
---

函数指针的形式：

```c
int32_t func (double d, char* s);	//the function
int32_t (*)(double, char*);			//the function pointer
//a pointer to a function that takes a double and a char* and returns an int32_t
```

下面是Lumetta给出的一个例子：

```c
int32_t isort(void* base, int32_t n_elts, size_t size, int32_t(*is_smaller)(void* t1, void* t2)){
//在isort函数中调用is_smaller函数
    char* array = base;
    void* current;
    int32_t sorted;
    int32_t index;
	if(NULL == (current = malloc(size))){
		return 0;
	}
    //这段代码实现一种类似插入排序的算法
    //对于每一个内循环，都提取一个数值，通过比较大小，找到它当前合适的位置，插入数组
    //通过对每个元素执行一次循环，最终实现整体的有序排列
	for(sorted = 2; n_elts >= sorted; sorted++){
        memcpy(current, array + (sorted-1) * size, size);
        //copy one element into current
		for(index = sorted - 1; 0 < index; index--){
			if((*is_smaller)(current, array + (index-1) *size, array + (index-1) *size)){
                //if current is smaller
				memcpy(array + index*size, array + (index-1)*size, size);
				//copy array element index-1 over array element index
            }
			else{
				break;
                //otherwise, found the right place
			}
		}	
		memcpy(array + index * size, current, size);
        //copy current to correct position
	}
	free(current);
	return 1;
}
```

一个简单的指针函数用法示例：

```c
int add(int a, int b){return a+b;}
int magic_1(int a, int b){}//函数内容略
int magic_2(int a, int b){}//函数内容略
typedef int(*operation_t)(int, int);//int型，与函数返回值相同
static operation_t(func_arr[3] = {&add, &magic_1, &magic_2});	//定义operation_txing
int func_index = ;//略
(*(func_arr[func_index]))(a, b)	//调用函数
```

