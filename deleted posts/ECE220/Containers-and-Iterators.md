---
title: Containers_and_Iterators
date: 2024-05-21 15:17:54
tags: ECE220
---

container是一种包含了其它数据结构的数据结构，并对它们有特殊的访问能力。

例如：linked lists，heap，dynamically-sized arrays

Lumetta在PPT里主要讲了一个双向链表，并简单介绍了如何在双向链表的head后面加data structure：

```c
void dl_insert(double_list_t* head, double_list_t* elt){
    elt->next = head->next;
    elt->prev = head;
    head->next->prev = elt;
    head->next = elt;
}
```

<img src="Containers-and-Iterators.assets/image-20240521154353821.png" alt="image-20240521154353821" style="zoom: 67%;" /><img src="Containers-and-Iterators.assets/image-20240521154425708.png" alt="image-20240521154425708" style="zoom: 67%;" />

在构建这样一个data structure的时候，需要把double_list_t放到最前面，也就是图中的prev和next。如果在prev前面还有一部分的“thing”，我们无法计算如何从指向prev的pointer跳转到thing pointer，因为我们不知道这两个指针之间，thing占用的内存大小。

如果我们需要将一个data structure移除，只需要两步：

```
void dl_remove(double_list_t* elt){
	elt->prev->next = elt->next;
	elt->next->prev = elt-prev;
}
```

<img src="Containers-and-Iterators.assets/image-20240521155338375.png" alt="image-20240521155338375" style="zoom:67%;" />

以及，还有找到链表中第一个元素的函数：

```c
void* dl_first(double_list_t* head){
	return(head == head->next ? NULL : head->next);
}
```

接下来就是iteration，其实就是迭代。实现对链表内容的迭代。当然，下面这个代码是不完整的，意会即可：

```c
typedef enum{
	DL_CONTINUE,				//keep going
	DL_STOP_AND_RETURN,			//return this thing
	DL_REMOVE_AND_CONTINUE,		//remove thing and continue
	DL_REMOVE_AND_STOP,			//remove thing and return it
	DL_FREE_AND_CONTINUE,		//free the thing and continue
}dl_execute_response_t;

typedef dl_execute_response_t(*dl_execute_func_t)(void* dl, void* arg);
//定义回调函数
void* dl_execute_on_all(double_list_t* head, dl_execute_func_t func, void* arg){
    double_list_t* dl;
    double_list_t* remove;
    dl_execute_response_t result;
    
    for(dl = head->next; head != dl; dl = dl->next){
        result = (*func)(dl, arg);
        switch(result){
            case DL_REMOVE_AND_STOP: dl_remove(dl);
            case DL_STOP_AND_RETURN: return dl;
            case DL_REMOVE_AND_CONTINUE:
            case DL_FREE_AND_CONTINUE:
                remove = dl;		//copy it to remove
                dl = dl->prev;		//read the next
                dl_remove(remove);	//remove "thing" from list
                if(result == DL_FREE_AND_CONTINUE){
                    free(remove);
                }
                break;
            default: break;
        }
    }
    return NULL;
}
```

