---
title: IO_in_C
date: 2024-05-22 22:03:09
tags: ECE220
---

C有3种默认流，对应3种默认文件描述符：

stdin对应keyboard（输入）

stdout对应display（正常输出）

stderr对应display（错误）

`int fgetc(FILE* stream);			int getc(FILE* stream);			//若失败，返回EOF(-1)`

`int fputc(int c, FILE* stream);		intputc(int c, FILE* stream);`

`getchar()`与LC3的IN指令类似，返回值是排在标准输入流（stdin）头部的ASCII码值。同时，它需要回车键确认输入内容，并将其添加到输入流尾部。

`putchar()`与LC3的OUT指令类似，将一个字符输出到标准输出流中。它需要\n来将字符序列添加到输出流。

读写模式：

“r”：只读

“w”：只写（如果文件名有对应原文件，则先删除原文件）

“a”：在文件末尾添加

“r+”：读+写

“w+”：先删除，再读+写

“a+”：读+写，写在文件末尾

`FILE *fopen(const char* path, const char* mode);`path是文件名，mode是读写模式（如上），如果执行成功，则返回一个新的stream，如果失败，则返回NULL。

`int fclose(FILE* stream);`stream是需要关闭的文件指针，函数成功执行，返回0；函数未成功，返回EOF（-1）。

用fgets，fputs读写字符串：

`char* fgets(char* s, int size, FILE* stream);	//return s, or NULL on failure`

`int fputs(const char* s, FILE *stream);//return non-negative number, or EOF on failure`

```c
int32_t file_reduce(const char* fname){
    FILE* in;		//定义输入文件指针
    FILE* out;		//ding'y
    if(NULL == (in = fopen(fname,"r")) || NULL == (out = fopen("out.txt","w"))){
        if(NULL != in){	//如果是out文件打开失败，则需要关闭in文件
            fclose(in);
        }				//如果是in文件打开失败，
        return 0;
    }
    int last = EOF;	//定义一个int，存储上一个值
    int character;	//存储当前值
    while(EOF != (character = fgetc(in))){	//while不是最后一个字符
        if(last != character){				//如果当前字符与上一个不同，则输出
            fputc(character, out);
            last = character;
        }
    }
    fclose(in);		//记得关闭文件
    return (0 == fclose(out) ? 1 : 0);	//关闭成功，return 1；关闭失败，return 0
}
```

