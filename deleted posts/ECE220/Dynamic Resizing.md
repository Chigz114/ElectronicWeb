---
title: Dynamic Resizing
date: 2024-05-19 14:53:00
tags: ECE220
---

​		对于dynamic resize，我们一般不会每次增加固定个数的位置，因为“增加”操作会消耗很多时间。我们一般采用：每次达到储存上限，都将储存位置翻倍。这样，可以在储存位置限制在合理范围内的情况下，尽量减少realloc操作次数。

​		经计算，这样的做法，浪费的空间的期望是2 (ln2 - 0.5) = 38%

​		接下来是对三个Dynamic Resizing函数和free的介绍

```c#
void* malloc (size_t size);		
//新建一个大小为size的内存块，并返回指针
void* calloc (size_t num_elts, size_t elt_size);	
//新建一个大小为num_elts * elt_size的内存块，并返回指针，并将内存块内的值初始化为0
void* realloc (void* ptr, size_t size);
//清除ptr指针指向的内存块，并将其转移到大小为size的新内存块中，并返回新内存块指针
```

​		对于这三个函数，需要注意的是，如果没有找到符合要求的内存块，函数会返回NULL。而且，对于malloc和realloc函数，它们并不会初始化找到的内存块。如果需要初始化，则需要使用calloc或其它方法。

​		对于realloc函数，可能存在新建size小于原内存的情况，这时不会报错，而是会将原内存块前size位的值复制到新内存块。

​		因为函数返回的是无类型指针，在使用前需要进行强制类型转换，如：

```c
int32_t* new_ptr = (int32_t*)realloc(orig_ptr, size);
```

```c#
void free (void* ptr);
//将ptr指向的内存块free
```

​		free函数没啥说法，只是注意对于单个内存块只free一次。如果出现结构体等指针嵌套的情况，需要先free结构体内部指针，再free结构体指针。

​		函数的等价替换：

```c#
malloc(size) → realloc(NULL, size);
free(ptr) → realloc(ptr, 0);
```

​		Lumetta的小程序：

```c#
int32_t player_create(char* n, char* pswd, int32_t p_age, player_t** new_p){
    if(NULL == player_list){	//如果是第一次执行，新建player_list
        player_list = malloc(max_players * sizeof(player_list));
        if(NULL == player_list){	//如果新建不成功
            return 0;
        }
    }
    else{	//不是第一次执行
        if(max_players == num_players){	//达到储存上限
            new_copy = realloc(player_list, 2 * max_players * sizeof(*player_list));
        }
        if(NULL == new_copy){	//扩大内存不成功
            return 0;
        }
        max_player *= 2;
        player_list = new_copy;
    }
    *new_p = &player_list[num_players];
    num_players++;
    return 1;
}
```

程序中有部分没有写，比如new_copy的定义，等等。仅作为帮助理解。
