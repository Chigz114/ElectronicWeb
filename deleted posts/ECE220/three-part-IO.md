---
title: three_part_IO
date: 2024-05-23 12:47:02
tags: ECE220
---

```c
static const
    int32-t max_word_len = 500;
int main(int argc, char* argv[]){
    char buf[max_word_len +1]	//缓冲区域，存储500个字母+1个NUL
    char* write;				//buf中下一个要写入的字母
    int32_t word_len			//当前word的长度
    FILE* in_file;				//输入文件指针
    int32_t a_char;				//从输入文件中读取一个字母
    if(2 != argc){				//调用时必须是两个元素
        fprintf(stderr, "syntax: %s <file name>\n", argv[0]);	//报错传到stderr中
        return EXIT_BAD_ARGS;	//argv[0]代表文件名
    }
    if(NULL == (in_file = fopen(argv[1],"r"))){	//若无法打开输入文件
        perror("open file");
        return EXIT_FAIL;
    }
    write = buf;
    word_len = 0;
    while(EOF != (a_char = getc(in_file))){
        //执行处理，代码略
    }
    if(0 < word_len){	//文件若在缓冲区域仍有一个word时退出
        *write = 0;		//将write指针指向缓冲区第一个位置
        puts(buf);		//输出
    }
    (void)fclose(in_file);	//关闭输入流
    return EXIT_SUCCEED;	
}
```

