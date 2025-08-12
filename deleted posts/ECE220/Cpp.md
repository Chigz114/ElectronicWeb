---
title: Cpp
date: 2024-05-19 22:54:37
tags: ECE220
top_img: https://patchwiki.biligame.com/images/blhx/c/cf/6tj18cxvk0h6rh0s3l2nczv8zggspm1.png
cover: https://patchwiki.biligame.com/images/blhx/c/cf/6tj18cxvk0h6rh0s3l2nczv8zggspm1.png
---

# **Introduction to Cpp**

 

## **References**

引用（Reference）是 C++ 相对于C语言的又一个扩充。引用可以看做是数据的一个别名，通过这个别名和原来的名字都能够找到这份数据。

引用的定义方式类似于指针，只是用`&`代替了`*`，语法格式为：

```c++
type &name = data;
```

type 是被引用的数据的类型，name 是引用的名称，data 是被引用的数据。引用必须在定义的同时初始化，并且以后也要从一而终，不能再引用其它数据，这有点类似于常量（const 变量）。

```c++
#include <iostream>
using namespace std;

int main() {
    int a = 99;
    int &r = a;								//&表示引用
    cout << a << ", " << r << endl;
    cout << &a << ", " << &r << endl;		//&表示取地址

    return 0;
}
//运行结果：
//99，99
//0x28ff44, 0x28ff44
```

本例中，变量 r 就是变量 a 的引用，它们用来指代同一份数据；也可以说变量 r 是变量 a 的另一个名字。从输出结果可以看出，a 和 r 的地址一样，都是`0x28ff44`；或者说地址为`0x28ff44`的内存有两个名字，a 和 r，想要访问该内存上的数据时，使用哪个名字都行。

只有在创建reference的时候需要用到`&`，如果在使用时添加，则表示取地址。

同时，通过r也可以修改原变量a中储存的数据。

如果不希望通过引用变量修改原数据，可以在定义时添加const限制，形如：

```c++
const type &name = value;
type const &name = value;	//二者等价
```

一段展现按引用传参优势的代码：

```c++
#include <iostream>
using namespace std;
void swap1(int a, int b);
void swap2(int *p1, int *p2);
void swap3(int &r1, int &r2);
int main() {
    int num1, num2;
    cout << "Input two integers: ";
    cin >> num1 >> num2;
    swap1(num1, num2);
    cout << num1 << " " << num2 << endl;
    cout << "Input two integers: ";
    cin >> num1 >> num2;
    swap2(&num1, &num2);
    cout << num1 << " " << num2 << endl;
    cout << "Input two integers: ";
    cin >> num1 >> num2;
    swap3(num1, num2);
    cout << num1 << " " << num2 << endl;
    return 0;
}
//直接传递参数内容
void swap1(int a, int b) {		//该写法是错的，下面两种等价
    int temp = a;
    a = b;
    b = temp;
}
//传递指针
void swap2(int *p1, int *p2) {
    int temp = *p1;
    *p1 = *p2;
    *p2 = temp;
}
//按引用传参
void swap3(int &r1, int &r2) {
    int temp = r1;
    r1 = r2;
    r2 = temp;
}
```

引用还可以作为函数的返回值`return r；`

在将引用作为函数返回值时应该注意一个小问题，就是不能返回局部数据（例如局部变量、局部对象、局部数组等）的引用，因为当函数调用完成后局部数据就会被销毁，有可能在下次使用时数据就不存在了，C++ 编译器检测到该行为时也会给出警告。如：

```c++
#include <iostream>
using namespace std;
int &plus10(int &r) {
    int m = r + 10;
    return m;  //返回局部数据的引用
}
int main() {
    int num1 = 10;
    int num2 = plus10(num1);
    cout << num2 << endl;
    int &num3 = plus10(num1);
    int &num4 = plus10(num3);
    cout << num3 << " " << num4 << endl;
    return 0;
}
//预期结果：
//20
//20 30
```

m在使用完后被销毁，该做法得不到预期的答案。

reference & pointer 比较

引用很容易与指针混淆，它们之间有三个主要的不同：

- 不存在空引用。引用必须连接到一块合法的内存。
- 一旦引用被初始化为一个对象，就不能被指向到另一个对象。指针可以在任何时候指向到另一个对象。
- 引用必须在创建时被初始化。指针可以在任何时间被初始化。



鉴于Pavel又说要学参数引用，故贴个例子在这儿，和C的指针区别不大：

如果一个函数收到了一个参数（variable）的引用，它就可以修改这个参数的值。

```c++
// C++ Program to demonstrate
// Passing of references as parameters
#include <iostream>
using namespace std;

// Function having parameters as
// references
void swap(int& first, int& second)
{
    int temp = first;
    first = second;
    second = temp;
}

// Driver function
int main()
{
    // Variables declared
    int a = 2, b = 3;

    // function called
    swap(a, b);

    // changes can be seen
    // printing both variables
    cout << a << " " << b;
    return 0;
}
//输出
//3 2
```

 同时，Pavel还说要学：常引用作为函数变量（constant references as function arguments），啧

引用本身就是一个常指针，意味着它指向的位置是固定的。如果在前面再加上const，就意味着它指向的空间所存的值也不可以被该引用修改。形如`const &a`

不过有意思的是，因为引用和原变量指向的是同一地址空间，所以，虽然常引用无法改变那片地址储存的值，原变量依然可以修改这个值：

```c++
#include<iostream>
using namespace std;

void main(){
    int b = 15;
    const int &a = b;
    b = 11;	//此时a和b的值都会发生改变
    a = 11;	//但这行就会报错，因为引用a是constant   
}
```

更多资料和题目，参考[References in C++ - GeeksforGeeks](https://www.geeksforgeeks.org/references-in-cpp/)



 

## **Class**

​		class是C++中新增的关键字，专门用来定义类。

```c++
class Student{		//类名称：student
public:				//表示类的成员变量或成员函数有“公开”的访问权限
    //成员变量
    char *name;
    int age;
    float score;

    //成员函数
    void say(){
        cout<<name<<"的年龄是"<<age<<"，成绩是"<<score<<endl;
    }
};
```

类只是一个模板，编译后不占用内存空间，所以在定义类时不能对成员变量进行初始化，因为没有地方存储数据。只有在创建对象后才会给成员变量分配内存，此时才能赋值。

在创建class对象时，关键字class可写可不写

```c++
class Student Chi;
Student Chi;		//两者等价，正确
Student allStu[100];//创建allStu数组，每个元素都是class Student的对象
```

与C类似的，可以用"."来访问成员变量，不同的是，class还可以访问成员函数。

```C++
#include <iostream>
using namespace std;

//类通常定义在函数外面
class Student{		//创建类，不占用内存
public:
    //类包含的变量
    char *name;
    int age;
    float score;
    //类包含的函数
    void say(){
        cout<<name<<"的年龄是"<<age<<"，成绩是"<<score<<endl;
    }
};

int main(){
    //创建对象
    Student stu;	//创建对象stu，其类为Student
    stu.name = "小明";
    stu.age = 15;
    stu.score = 92.5f;
    stu.say();

    return 0;
}
```

如果要获取stu的指针，则需要用取地址符“&”得到它的地址：

```c++
Student stu;
Student *pStu = &stu;
```

pStu是一个指针，指向Student类型的数据，也就是通过Student创建出来的对象stu。

当然，也可以在堆上创建对象，这时候需要用到new关键字：

```C++
Student *pStu = new Student;
```

栈创建的对象都有一个名字，如stu，作为索引。所以不必用指针指向它。但通过new创建的，在堆上分配内存的对象，没有名字，只能得到一个指针，所以必须通过一个指针变量接收这个指针。

栈由程序自动管理，无法使用delete；堆由程序员管理，使用完毕后可通过delete删除。

有了对象指针之后，就可以通过箭头"->"来访问对象的成员了。一个完整的例子：

```C++
#include <iostream>
using namespace std;

class Student{
public:
    char *name;
    int age;
    float score;

    void say(){
        cout<<name<<"的年龄是"<<age<<"，成绩是"<<score<<endl;
    }
};

int main(){
    Student *pStu = new Student;
    pStu -> name = "小明";
    pStu -> age = 15;
    pStu -> score = 92.5f;
    pStu -> say();
    delete pStu;  //删除对象

    return 0;
}
```

 

 

## **Public Simple Inheritance** & public and private access modifiers

**继承（Inheritance）**可以理解为一个类从另一个类获取成员变量和成员函数的过程。例如类 B 继承于类 A，那么 B 就拥有 A 的成员变量和成员函数。

C++中，**派生（Derive）**和继承是同一概念。只是解释的角度不同，分从子类和父类解释。被继承的类称为父类或基类，继承的类称为子类或派生类。

继承的使用场景：

1. 当你创建的新类与现有的类相似，只是多出若干成员变量或成员函数时，可以使用继承，这样不但会减少代码量，而且新类会拥有基类的所有功能。

2. 当你需要创建多个类，它们拥有很多相似的成员变量或成员函数时，也可以使用继承。可以将这些类的共同成员提取出来，定义为基类，然后从基类继承，既可以节省代码，也方便后续修改成员。

下面给出一个实例，定义基类People，派生出Student类：

```c++
#include<iostream>
using namespace std;

//基类 Pelple
class People{
public:
    void setname(char *name);
    void setage(int age);
    char *getname();
    int getage();
private:
    char *m_name;
    int m_age;
};
void People::setname(char *name){ m_name = name; }
void People::setage(int age){ m_age = age; }
char* People::getname(){ return m_name; }
int People::getage(){ return m_age;}

//派生类 Student
class Student: public People{	//继承People中的public成员
public:							//同时在Student中新增两个public成员
    void setscore(float score);
    float getscore();
private:
    float m_score;
};
void Student::setscore(float score){ m_score = score; }
float Student::getscore(){ return m_score; }

int main(){
    Student stu;
    stu.setname("小明");
    stu.setage(16);
    stu.setscore(95.5f);
    cout<<stu.getname()<<"的年龄是 "<<stu.getage()<<"，成绩是 "<<stu.getscore()<<endl;

    return 0;
}
//输出
//小明的年龄是 16，成绩是 95.5
```

不同的继承方式会影响基类成员在派生类中的访问权限。若未指定继承方式，默认为private。

**1) public继承方式**

- 基类中所有 public 成员在派生类中为 public 属性；
- 基类中所有 protected 成员在派生类中为 protected 属性；
- 基类中所有 private 成员在派生类中不能使用。


**2) protected继承方式**

- 基类中的所有 public 成员在派生类中为 protected 属性；
- 基类中的所有 protected 成员在派生类中为 protected 属性；
- 基类中的所有 private 成员在派生类中不能使用。


**3) private继承方式**

- 基类中的所有 public 成员在派生类中均为 private 属性；
- 基类中的所有 protected 成员在派生类中均为 private 属性；
- 基类中的所有 private 成员在派生类中不能使用。

声明为public的数据成员和成员函数也可以被其他类和函数访问。类的公共成员可以在程序的任何地方使用直接成员访问操作符`.`对该类的对象进行访问。

声明为private的类成员只能由类内部的成员函数访问。它们不允许被类之外的任何对象或函数直接访问。只有成员函数或友元函数才允许访问类的私有数据成员。

private函数使用方法（对比）：

```c++
// C++ program to demonstrate private
// access modifier

#include<iostream>
using namespace std;

class Circle
{ 
	// private data member
	private: 
		double radius;
	
	// public member function 
	public: 
		double compute_area()
		{ // member function can access private 
			// data member radius
			return 3.14*radius*radius;
		}
	
};

// main function
int main()
{ 
	// creating object of the class
	Circle obj;
	
	// trying to access private data member
	// directly outside the class
	obj.radius = 1.5;		//此处在main函数内调用radius，而radius是private值，会报错
	
	cout << "Area is:" << obj.compute_area();
	return 0;
}
//输出
// In function 'int main()':
//11:16: error: 'double Circle::radius' is private
//         double radius;
//                ^
//31:9: error: within this context
//     obj.radius = 1.5;
//         ^
```

```c++
// C++ program to demonstrate private
// access modifier

#include<iostream>
using namespace std;

class Circle
{ 
	// private data member
	private: 
		double radius;
	
	// public member function 
	public: 
		void compute_area(double r)
		{ // member function can access private 
			// data member radius
			radius = r;
			
			double area = 3.14*radius*radius;
			
			cout << "Radius is: " << radius << endl;
			cout << "Area is: " << area;
		}
	
};

// main function
int main()
{ 
	// creating object of the class
	Circle obj;
	
	// trying to access private data member
	// directly outside the class
	obj.compute_area(1.5);		//类的内部成员函数可以调用private
	
	
	return 0;
}
//输出
//Radius is: 1.5
//Area is: 7.065
```

对比可见，只有类的内部函数才能调用private值。



用using关键字可以改变基类成员在派生类中的访问权限。但using 只能改变基类中 public 和 protected 成员的访问权限，不能改变 private 成员的访问权限，因为基类中 private 成员在派生类中是不可见的，根本不能使用，所以基类中的 private 成员在派生类中无论如何都不能访问。

实例：

```c++
#include<iostream>
using namespace std;

//基类People
class People {
public:
    void show();
protected:
    char *m_name;
    int m_age;
};
void People::show() {
    cout << m_name << "的年龄是" << m_age << endl;
}

//派生类Student
class Student : public People {
public:
    void learning();
public:
    using People::m_name;  //将protected改为public
    using People::m_age;  //将protected改为public
    float m_score;
private:
    using People::show;  //将public改为private
};
void Student::learning() {
    cout << "我是" << m_name << "，今年" << m_age << "岁，这次考了" << m_score << "分！" << endl;
}

int main() {
    Student stu;
    stu.m_name = "小明";
    stu.m_age = 16;
    stu.m_score = 99.5f;
    stu.show();  //compile error，show()是private属性的，会报错
    stu.learning();

    return 0;
}
```





## **`this` Pointer**

this是C++中的一个关键字，也是一个const指针，指向当前对象，通过它可以访问当前对象的所有成员。

因为this是一个指针，所以在访问对象的成员时，需要用`->`来访问。

```c++
#include <iostream>
using namespace std;

class Student{
public:
    void setname(char *name);
    void setage(int age);
    void setscore(float score);
    void show();
private:
    char *name;
    int age;
    float score;
};

void Student::setname(char *name){
    this->name = name;
}
void Student::setage(int age){
    this->age = age;
}
void Student::setscore(float score){
    this->score = score;
}
void Student::show(){
    cout<<this->name<<"的年龄是"<<this->age<<"，成绩是"<<this->score<<endl;
}

int main(){
    Student *pstu = new Student;
    pstu -> setname("李华");
    pstu -> setage(16);
    pstu -> setscore(96.5);
    pstu -> show();

    return 0;
}
```

this用于类的内部，通过this可以访问类的所有成员。包括private、protected、public属性的。

this是const指针，值不可被修改。只有对象创建后this才有意义。

 

## Destructor and Default / Copy / Explicit Constructor

C++中，有一种特殊的成员函数，它的名字和类名相同，没有返回值，不需要用户显式调用（用户也不能调用），而是在创建对象时自动执行。这种特殊的成员函数就是构造函数（Constructor）。它可以在创建对象的同时为成员变量赋初值。

**注意，构造函数与类同名。**

在Class内定义Constructor的语法：

```c++
<class-name> (list-of-parameters)
{
     // constructor definition
}
```

在Class外定义Constructor的语法：

```c++
<class-name>: :<class-name>(list-of-parameters)
{
     // constructor definition
}
```

在Class内定义Constructor的实例：

```c++
// defining the constructor within the class
#include <iostream>
using namespace std;
class student {
	int rno;
	char name[50];
	double fee;

public:
	// 定义constructor
	student()
	{
		cout << "Enter the RollNo:";
		cin >> rno;
		cout << "Enter the Name:";
		cin >> name;
		cout << "Enter the Fee:";
		cin >> fee;
	}

	void display()
	{
		cout << endl << rno << "\t" << name << "\t" << fee;
	}
};

int main()
{
	student s; // constructor gets called automatically when
			// we create the object of the class
	s.display();
	return 0;
}
//输出
//Enter the RollNo:121
//Enter the Name:Geeks
//Enter the Fee:5000
//121     Geeks   5000
```

在Class外定义Constructor的实例：

```c++
// defining the constructor outside the class
#include <iostream>
using namespace std;
class student {
	int rno;
	char name[50];
	double fee;

public:
	// 在定义类时仅声明Constructor，但未给出定义
	student();
	void display();
};

// 在类外定义Constructor
student::student()
{
	cout << "Enter the RollNo:";
	cin >> rno;
	cout << "Enter the Name:";
	cin >> name;
	cout << "Enter the Fee:";
	cin >> fee;
}

void student::display()
{
	cout << endl << rno << "\t" << name << "\t" << fee;
}

// driver code
int main()
{
	student s;
	s.display();
	return 0;
}
//输出
//Enter the RollNo:11
//Enter the Name:Aman
//Enter the Fee:10111
//11      Aman    10111
```

#### Default Constructor

默认构造函数没有变量。它不需要输入，执行一些固定工作或不执行。如果我们不定义构造函数，程序会自动定义一个函数。下面是一个实例：

```c++
// C++ program to illustrate the concept of default
// constructors
#include <iostream>
using namespace std;

class construct {
public:
	int a, b;

	// Default Constructor
	construct()
	{
		a = 10;
		b = 20;
	}
};

int main()
{
	// Default constructor called automatically
	// when the object is created
	construct c;
	cout << "a: " << c.a << endl << "b: " << c.b;
	return 1;
}
//输出
//a: 10
//b: 20
```

另一个实例，如果不定义构造函数：

```c++
// C++ program to demonstrate the implicit default
// constructor
#include <iostream>
using namespace std;

// class
class student {
	int rno;
	char name[50];
	double fee;

public:
};

int main()
{
	// creating object without any parameters
	student s;
	return 0;
}
//无输出
```

#### Copy Constructor

复制构造函数是使用同一类（class）下的另一个对象初始化当前对象的成员函数。它可以引用`&`同“类”的对象作为参数。

语法：

```c++
ClassName (ClassName &obj)
{
  // body_containing_logic
}
```

如果没有定义显式复制构造函数，C++会提供默认的隐式复制构造函数：

```C++
// C++ program to illustrate the use of Implicit copy
// constructor
#include <iostream>
using namespace std;

class Sample {
	int id;

public:		//此处没有给出copy constructor的定义，即为隐式
	// parameterized constructor
	Sample(int x) { id = x; }
	void display() { cout << "ID=" << id; }
};

int main()
{
	Sample obj1(10);		//通过parameterized constructor给obj1的id赋初值
	obj1.display();
	cout << endl;

	// creating an object of type Sample from the obj
	Sample obj2(obj1); //通过隐式复制构造函数给obj2赋值，值来自obj1
	obj2.display();
	return 0;
}
//输出
//ID=10
//ID=10
```

显式复制构造函数，实例：

```c++
// C++ Program to demonstrate how to define the explicit
// copy constructor
#include <iostream>
using namespace std;

class Sample {
	int id;

public:
	//默认构造函数
	Sample() {}

	// parameterized constructor
	Sample(int x) { id = x; }

	//复制构造函数，定义
	Sample(Sample& t) { id = t.id; }		//定义为，复制id值

	void display() { cout << "ID=" << id; }
};

// driver code
int main()
{
	Sample obj1(10);
	obj1.display();
	cout << endl;

	// copy constructor called
	Sample obj2(obj1); //从obj1中复制id值到obj2
	obj2.display();

	return 0;
}
//输出
//ID=10
//ID=10
```

带有参数化构造函数的显式复制构造函数，实例：

```c++
// C++ program to demonstrate copy construction along with
// parameterized constructor
#include <iostream>
#include <string.h>
using namespace std;

// class definition
class student {
	int rno;
	char name[50];
	double fee;

public:
	student(int, char[], double);
	student(student& t) //复制构造函数，定义
	{
		rno = t.rno;			//复制rno参数
		strcpy(name, t.name);	//复制name参数
		fee = t.fee;			//复制fee参数
	}

	void display();
};

student::student(int no, char n[], double f)	//类外，定义参数化构造函数
{
	rno = no;
	strcpy(name, n);
	fee = f;
}

void student::display()
{
	cout << endl << rno << "\t" << name << "\t" << fee;
}

int main()
{
	student s(1001, "Manjeet", 10000);	//定义对象s，输入参数化构造函数的值
	s.display();

	student manjeet(s);	//定义对象manjeet，并通过复制构造函数，从对象s中复制值
	manjeet.display();

	return 0;
}
//输出
//1001    Manjeet    10000
//1001    Manjeet    10000
```

#### 显式构造函数（explicit）

学习显式构造函数，应当先理解隐式（implicit）构造函数。

如果c++类的**其中一个构造函数有一个参数**，那么在编译的时候就会有一个缺省的转换操作：将该构造函数对应数据类型的数据转换为该类对象。这句话可能有点难懂，举个例子：

```c++
class String{
	String(const char* p);	//用字符串p作为初始化值
}
String s1 = "hello";	//这就是一种隐式转换
//这样的转换本身是不合语法的，但通过隐式转换，将“数据”转换为“对象（数据）”，使之合法
//等价于
String s1 = String("hello")
```

但有的时候，可能不需要这种隐式转换：

```c++
class String{
	String(int n);				//本意是预先分配n个字节给字符串
	String(const char* p);		//为字符串p初始化
}
//两种正常写法
String s2(10);				//分配10个字节
String s3 = String(10);		//分配10个字节
//两种不正常写法
String s4 = 10;				//编译通过，分配10个字节的空字符串
String s5 = 'a';			//编译通过，分配int('a')个字节的空字符串
```

为避免这种隐式转换，在构造函数前加上explicit：

```c++
class String{
	explicit String(int n);				//本意是预先分配n个字节给字符串
	String(const char* p);		//为字符串p初始化
}
//两种正常写法仍然正确
String s2(10);				//分配10个字节
String s3 = String(10);		//分配10个字节
//两种不正常写法会报错
String s4 = 10;				//编译不通过
String s5 = 'a';			//编译不通过
```

所以，有些时候，explicit可以有效防止构造函数的隐式转换带来的错误或者误解。

#### 析构函数（Deconstructor）

创建对象时系统会自动调用构造函数进行初始化工作，同样，销毁对象时系统也会自动调用一个函数来进行清理工作，例如释放分配的内存、关闭打开的文件等，这个函数就是析构函数。

析构函数（Destructor）也是一种特殊的成员函数，没有返回值，不需要程序员显式调用（程序员也没法显式调用），而是在销毁对象时自动执行。**构造函数的名字和类名相同，而析构函数的名字是在类名前面加一个`~`符号。**

注意：析构函数没有参数，不能被重载，因此一个类只能有一个析构函数。如果用户没有定义，编译器会自动生成一个默认的析构函数。

下面实例中，通过析构函数释放已经分配的内存（其中大部分代码不需要理解，仅看关于~VLA部分，理解语法即可）：

```c++
#include <iostream>
using namespace std;

class VLA{
public:
    VLA(int len);  //构造函数
    ~VLA();  //析构函数
public:
    void input();  //从控制台输入数组元素
    void show();  //显示数组元素
private:
    int *at(int i);  //获取第i个元素的指针
private:
    const int m_len;  //数组长度
    int *m_arr; //数组指针
    int *m_p;  //指向数组第i个元素的指针
};

VLA::VLA(int len): m_len(len){  //使用初始化列表来给 m_len 赋值
    if(len > 0){ m_arr = new int[len];  /*分配内存*/ }
    else{ m_arr = NULL; }
}
VLA::~VLA(){			//定义析构函数
    delete[] m_arr;  //释放内存
}
void VLA::input(){
    for(int i=0; m_p=at(i); i++){ cin>>*at(i); }
}
void VLA::show(){
    for(int i=0; m_p=at(i); i++){
        if(i == m_len - 1){ cout<<*at(i)<<endl; }
        else{ cout<<*at(i)<<", "; }
    }
}
int * VLA::at(int i){
    if(!m_arr || i<0 || i>=m_len){ return NULL; }
    else{ return m_arr + i; }
}

int main(){
    //创建一个有n个元素的数组（对象）
    int n;
    cout<<"Input array length: ";
    cin>>n;
    VLA *parr = new VLA(n);
    //输入数组元素
    cout<<"Input "<<n<<" numbers: ";
    parr -> input();
    //输出数组元素
    cout<<"Elements: ";
    parr -> show();
    //删除数组（对象）
    delete parr;

    return 0;
}
```

`~VLA()`就是 VLA 类的析构函数，它的唯一作用就是在删除对象（第 53 行代码）后释放已经分配的内存。

C++中的 new 和 delete 分别用来分配和释放内存，它们与C语言中 malloc()、free() 最大的一个不同之处在于：用 new 分配内存时会调用构造函数，用 delete 释放内存时会调用析构函数。

 



## namespace std

C++ 引入了命名空间的概念，计划重新编写库，将类、函数、宏等都统一纳入一个命名空间，这个命名空间的名字就是`std`。std 是 standard的缩写，意思是“标准命名空间”。

#### namespace

一个中大型软件往往由多名程序员共同开发，会使用大量的变量和函数，不可避免地会出现变量或函数的命名冲突。当所有人的代码都测试通过，没有问题时，将它们结合到一起就有可能会出现命名冲突。

例如小李和小韩都参与了一个文件管理系统的开发，它们都定义了一个全局变量 fp，用来指明当前打开的文件，将他们的代码整合在一起编译时，很明显编译器会提示 fp 重复定义（Redefinition）错误。

为了解决合作开发时的命名冲突问题，C++ 引入了**命名空间（Namespace）**的概念。请看下面的例子：

```c++
namespace Li{  //小李的变量定义
    FILE* fp = NULL;
}
namespace Han{  //小韩的变量定义
    FILE* fp = NULL;
}
```

小李与小韩各自定义了以自己姓氏为名的命名空间，此时再将他们的 fp 变量放在一起编译就不会有任何问题。

使用变量、函数时要指明它们所在的命名空间。以上面的 fp 变量为例，可以这样来使用：

```c++
Li::fp = fopen("one.txt", "r");  //使用小李定义的变量 fp
Han::fp = fopen("two.txt", "rb+");  //使用小韩定义的变量 fp
```

`::`是一个新符号，称为域解析操作符，在C++中用来指明要使用的命名空间。

除了直接使用域解析操作符，还可以采用using关键字声明，在代码的开头用`using`声明了 Li::fp，它的意思是，using 声明以后的程序中如果出现了未指明命名空间的 fp，就使用 Li::fp；但是若要使用小韩定义的 fp，仍然需要 Han::fp。

```c++
using Li::fp;
fp = fopen("one.txt", "r");  //使用小李定义的变量 fp
Han :: fp = fopen("two.txt", "rb+");  //使用小韩定义的变量 fp
```

using 声明不仅可以针对命名空间中的一个变量，也可以用于声明整个命名空间，如果命名空间 Li 中还定义了其他的变量，那么同样具有 fp 变量的效果。在 using 声明后，如果有未具体指定命名空间的变量产生了命名冲突，那么默认采用命名空间 Li 中的变量。

```c++
using namespace Li;
fp = fopen("one.txt", "r");  //使用小李定义的变量 fp
Han::fp = fopen("two.txt", "rb+");  //使用小韩定义的变量 fp
```

std具体内容不必了解，std可以大致理解为一个标准库，将可能用到的一些函数等等纳入。

只需要留意`using namespace std;`，它声明了命名空间 std，后续如果有未指定命名空间的符号，那么默认使用 std，代码中的 string、cin、cout 都位于命名空间 std。

```c++
#include <iostream>

void func(){
    //必须重新声明
    using namespace std;
    cout<<"http://c.biancheng.net"<<endl;
}

int main(){
    //声明命名空间std
    using namespace std;
   
    cout<<"C语言中文网"<<endl;
    func();

    return 0;
}
```

如果希望在所有函数中都使用命名空间 std，可以将它声明在全局范围中，例如：

```c++
#include <iostream>
//声明命名空间std
using namespace std;
void func(){
    cout<<"http://c.biancheng.net"<<endl;
}
int main(){
    cout<<"C语言中文网"<<endl;
    func();
    return 0;
}
```

当然，也可以对一个特定函数使用std，但这样写太麻烦了：

```c++
#include <cstdio>
int main(){
    std::printf("http://c.biancheng.net\n");
    return 0;
}
```





## `::`operator

鉴于Pavel老登又说要学`::`，故单独开 一小节。

作用域运算符，简单来说，作用就是从一个集合（一般是class）中取其成员。细分下来有以下几种：

1. 当存在有相同名称的局部变量时，要访问全局变量（不太可能考）：

```c++
#include<iostream>  
using namespace std; 
 
int x;  // Global x 
 
int main() 
{ 
  int x = 10; // Local x 
  cout << "Value of global x is " << ::x; 
  cout << "\nValue of local x is " << x;   
  return 0; 
} 
```

2. 在类以外定义函数（有一定概率考）：

```c++
#include<iostream>  
using namespace std; 
 
class A  
{ 
public:  
 
   // Only declaration 
   void fun(); 
}; 
 
// Definition outside class using :: 
void A::fun() 
{ 
   cout << "fun() called"; 
} 
 
int main() 
{ 
   A a; 
   a.fun(); 
   return 0; 
} 
```

3. 访问一个类的静态变量（有可能考）：

```c++
#include<iostream> 
using namespace std; 
 
class Test 
{ 
    static int x;   
public: 
    static int y;    
 
    // Local parameter 'a' hides class member 
    // 'a', but we can access it using :: 
    void func(int x)   
    {  
       // We can access class's static variable 
       // even if there is a local variable 
       cout << "Value of static x is " << Test::x; 
 
       cout << "\nValue of local x is " << x;   
    } 
}; 
 
// In C++, static members must be explicitly defined  
// like this 
int Test::x = 1; 
int Test::y = 2; 
 
int main() 
{ 
    Test obj; 
    int x = 3 ; 
    obj.func(x); 
 
    cout << "\nTest::y = " << Test::y; 
 
    return 0; 
} 
```

4. 命名空间，如果两个命名空间有相同名称的类，则可以通过该运算符一起使用，而不会发生冲突：

```c++
#include<iostream> 
int main(){ 
    std::cout << "Hello" << std::endl;
} 
```

5. 在一个类中引用另一个类。如果另一个类中存在一个类，我们可以使用嵌套类使用作用域运算符来引用嵌套的类：

```c++
#include<iostream> 
using namespace std; 
 
class outside 
{ 
public: 
      int x; 
      class inside 
      { 
      public: 
            int x; 
            static int y;  
            int foo(); 
 
      }; 
}; 
int outside::inside::y = 5;  
 
int main(){ 
    outside A; 
    outside::inside B; 
 
} 
```



## **`cout`**

预定义的对象 **cout** 是 **iostream** 类的一个实例。cout 对象"连接"到标准输出设备，通常是显示屏。**cout** 是与流插入运算符 << 结合使用的，如下所示：

```c++
#include <iostream>
 
using namespace std;
 
int main( )
{
   char str[] = "Hello C++";
 
   cout << "Value of str is : " << str << endl;
}
//输出：
//Value of str is : Hello C++
```

C++ 编译器根据要输出变量的数据类型，选择合适的流插入运算符来显示值。<< 运算符被重载来输出内置类型（整型、浮点型、double 型、字符串和指针）的数据项。

流插入运算符 << 在一个语句中可以多次使用，如上面实例中所示，**endl** 用于在行末添加一个换行符。

 

 

## **`string` Class**

C++中的string类是一个泛型类，由模板而实例化的一个标准类，本质上不是一个标准数据类型。

鉴于已经介绍了class，仅通过该实例演示string的基础用法：

```c++
#include <iostream>
#include <string>
 
using namespace std;
 
int main ()
{
   string str1 = "runoob";
   string str2 = "google";
   string str3;
   int  len ;
 
   // 复制 str1 到 str3
   str3 = str1;
   cout << "str3 : " << str3 << endl;
 
   // 连接 str1 和 str2
   str3 = str1 + str2;
   cout << "str1 + str2 : " << str3 << endl;
 
   // 连接后，str3 的总长度
   len = str3.size();
   cout << "str3.size() :  " << len << endl;
 
   return 0;
}
//输出：
//str3 : runoob
//str1 + str2 : runoobgoogle
//str3.size() :  12
```

 

 

## **`new` & `delete`**

C语言中，用malloc，free等函数动态分配内存，C++中保留了这些函数。同时，新增了两个**关键字**，new和delete，new用于动态分配内存，delete用于释放内存。

```C++
int *p = new int;		//分配1个int型内存空间
delete p;				//释放内存
```

new操作符会根据后面的数据类型来推断所需空间的大小。

如果希望分配一组连续的数据，可以使用new[]，同时，由new[]分配的内存需要由delete[]释放

```C++
int *p = new int[10];
delete[] p;
```

new和delete应当成对出现，用完即释放。





## Abstraction, Inheritance, Polymorphism, Encapsulation

#### abstraction

数据抽象是c++中面向对象编程最基本、最重要的特性之一。抽象意味着只显示基本信息，隐藏细节。数据抽象指的是只向外界提供有关数据的基本信息，隐藏后台细节或实现。

举个例子。一个人只知道踩油门会增加汽车的速度，或者踩刹车会使汽车停下来，但他不知道踩油门时速度实际上是如何增加的，他不知道汽车的内部机构，也不知道油门、刹车等在汽车中的实施。这就是抽象。

抽象的种类：

1. 数据抽象——这种类型只显示有关数据的必要信息，而隐藏不必要的数据。
2. 控制抽象——这种类型只显示有关实现的必要信息，而隐藏不必要的信息。

抽象的实现：

1. Classes：通过access specifier (public, private) 实现对部分内容的访问，而拒绝对另一部分的访问。
2. Header Files： 通过头文件封装函数。

数据抽象的优点

1. 帮助用户避免编写低级代码

2. 避免代码重复，提高可重用性。

3. 可以在不影响用户的情况下独立更改类的内部实现。

4. 帮助提高应用程序或程序的安全性，因为只向用户提供重要的细节。

5. 它降低了代码的复杂性和冗余性，从而提高了可读性。 

#### Inheritance

该部分主要是public，protected，private三种继承方式，在**Public Simple Inheritance & public and private access modifiers**这一节已经涉及了必要知识，不多赘述。放两个程序作为练习：

```c++
// Example: define member function without argument within
// the class

#include <iostream>
using namespace std;

class Person {
	int id;
	char name[100];

public:
	void set_p()
	{
		cout << "Enter the Id:";
		cin >> id;
		cout << "Enter the Name:";
		cin >> name;
	}

	void display_p()
	{
		cout << endl <<"Id: "<< id << "\nName: " << name <<endl;
	}
};

class Student : private Person {
	char course[50];
	int fee;

public:
	void set_s()
	{
		set_p();
		cout << "Enter the Course Name:";
		cin >> course;
		cout << "Enter the Course Fee:";
		cin >> fee;
	}

	void display_s()
	{
		display_p();
		cout <<"Course: "<< course << "\nFee: " << fee << endl;
	}
};

int main()
{
	Student s;
	s.set_s();
	s.display_s();
	return 0;
}
//输出
//Enter the Id: 101
//Enter the Name: Dev
//Enter the Course Name: GCS
//Enter the Course Fee:70000

//Id: 101
//Name: Dev
//Course: GCS
//Fee: 70000
```

```c++
// Example: define member function without argument outside the class

#include<iostream>
using namespace std;

class Person
{
	int id;
	char name[100];
	
	public:
		void set_p();
		void display_p();
};

void Person::set_p()
{
	cout<<"Enter the Id:";
	cin>>id;
	cout<<"Enter the Name:";
	cin>>name;
}

void Person::display_p()
{
	cout<<endl<<"id: "<< id<<"\nName: "<<name;
}

class Student: private Person
{
	char course[50];
	int fee;
	
	public:
		void set_s();
		void display_s();
};

void Student::set_s()
{
	set_p();
	cout<<"Enter the Course Name:";
	cin>>course;
	cout<<"Enter the Course Fee:";
	cin>>fee;
}

void Student::display_s()
{
	display_p();
	cout<<"\nCourse: "<<course<<"\nFee: "<<fee<<endl;
}

int main()
{
	Student s;
	s.set_s();
	s.display_s();
	return 0;
}
//输出
//Enter the Id: 101
//Enter the Name: Dev
//Enter the Course Name: GCS
//Enter the Course Fee: 70000
//Id: 101
//Name: Dev
//Course: GCS
//Fee: 70000
```

更多内容，请访问[Inheritance in C++ - GeeksforGeeks](https://www.geeksforgeeks.org/inheritance-in-c/?ref=lbp)

#### Polymorphism

多态性 (polymorphism) 这个词的意思是具有多种形式。简而言之，我们可以将多态性定义为消息以多种形式显示的能力。

多态性的一个现实例子是一个人同时可以有不同的特征。一个男人同时是父亲、丈夫和雇员。所以同一个人在不同的情况下表现出不同的行为。这就是所谓的多态性。多态被认为是面向对象编程的重要特征之一。

Types of Polymorphism

- Compile-time Polymorphism
- Runtime Polymorphism
