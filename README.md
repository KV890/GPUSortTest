# 排序PK

本项目主要测试 GPU 对一组字符串进行排序，并对重复的字符串进行去重。我们采用了不同的排序方法，包括基于 Thrust 库的排序方法以及我们自定义的 GPU 排序方法，来对比它们的性能。在此项目中，我们使用了两种负载生成工具：`db_bench` 和 `YCSB-C`，用于生成要排序的字符串。我们还考虑了 CPU 排序，并提供了几个接口供用户实现，以便比较它们的性能。

## 负载类型

项目使用了两种负载生成工具：`db_bench` 和 `YCSB-C`。它们的数据特征如下：

- `db_bench`：db_bench 是 LevelDB 的一个基准测试工具，主要用于模拟不同的数据库操作模式，例如随机读取、顺序读取、随机写入和顺序写入等。在默认情况下，db_bench 会生成按 Zipfian 分布的工作负载。在 Zipfian 分布中，一部分数据项会被频繁地访问，而大部分数据项会较少地被访问。这种分布模拟了许多真实世界的场景，例如网页访问、城市人口分布等。因此，使用 db_bench 生成的负载，可能会有一部分键（key）被多次重复，形成一种 "热点" 数据。

- `YCSB-C`：Yahoo! Cloud Serving Benchmark 的 C++ 版本，可以对存储系统进行性能测试。YCSB-C 的负载可以是均匀分布的，也可以是 Zipfian 分布的。在均匀分布的负载中，每个数据项被访问的概率是相同的。这种分布可以模拟一些场景，例如随机采样。

在项目中，键（key）和值（value）的大小分别为 16B 和 32B，并且是固定的。

## 排序类型

项目中考虑了两种排序类型：GPU 排序和 CPU 排序。

- `GPU 排序`：我们使用了基于 Thrust 库的排序方法作为基准，同时我们留了自定义 GPU 排序方法的接口，供大家实现。

- `CPU 排序`：我们也考虑了在 CPU 上进行排序，以便与 GPU 排序进行对比，并且我们可以直接观察到GPU和CPU的性能差异，来加深我们对GPU并行性和GPU强大并行资源的理解。

我们重点实现GPU排序算法，因为在GPU的加持下，GPU所实现的排序算的性能通常比CPU更高。本项目已提高部分自定义排序算法的接口，供大家实现。

## 正确的排序结果

我们的排序对象是一组字符串，比较对象是两个字符串。

首先，我们要求的排序结果是按照 **字典序** 进行排序的，比如 `adbc` `dbca` `acbd` `adbc` 的排序结果应该是: `acbd` `adbc` `adbc` `dbca`

然后，我们要求我们的集合中不能含有重复的字符串，比如排序结果: `acbd` `adbc` `adbc` `dbca` 这是错误的，我们需要将重复的元素去掉，只保留一个。

最后，我们的排序去重后的结果应该是: `acbd` `adbc` `dbca`

## 环境支持

本项目建议在Linux系统下运行，首先，你需要安装gcc, g++等工具

本项目用CMake工具进行管理，所以你也需要安装CMake工具。

由于使用了CUDA编程，所以需要CUDA环境支持。

## 提供现有的环境支持

环境搭配是一个繁琐的过程，为了方便你集中于思路思考和代码书写，现在提供一个已搭配好的环境，
你需要下载 `ToDesk` 软件，连接设备ID: `173 332 708`, 设备密码: `aaaa2358`，如需要输入密码，输入 `1` 回车即可。非常欢迎提供你的思路。

## 如何使用

你可以修改选项，选择使用哪种负载类型和要排序的 KV 对数量。具体的使用方法如下：

在执行程序时，在程序名后加上 -workload_type 来指定选择那种负载类型，如 `-workload 0` 代表db_bench，1 代表YCSB-C。

加上 -num 来指定要排序的数据量, 如 `-num 1000000` 代表要排序100万个数据量。

在你编译链接完成这个项目之后，会生成一个名为`GPUSortTest`的可执行文件，你可以这样指定你的参数，比如：

`GPUSortTest -workload_type 0 -num 2000000`, 表示用db_bench负载生成200万个KV对进行排序。

### 书写自己的代码

你可以打开 `sort` 文件夹，你可以看到 `cpu_sort.cc` 和 `cpu_sort.h`, 显而易见，用CPU实现的排序算法写在这两个C++文件中。

同理，`gpu_sort.cc`和 `gpu_sort.h` 这两个C++文件用来写GPU实现的排序算法。

写完你的排序算法之后，你需要进行测试、验证和比较。

依然在 `sort` 目录下，找到 `sorting_factory.cc` 和 `sorting_factory.h`, 在 `Sort成员函数` 中调用你的函数，计算你的算法所用时间，最后，不要忘记验证你的算法的正确性。通过 `Asser成员函数` 传入你排序好的结果进行验证。

### 建议

1. 当你测试自己的排序算法时，你应该对 `kvs成员变量` 的副本进行排序，以防原始数据被修改，从而导致后面的排序不正确。
2. 多个排序算法进行比较时，都应该记录它们的执行时长，并且在最后验证算法的正确行。
3. 为了观察数据特征，你可以将要排序的字符串打印出来进行观察，以便设计一个更符合该数据特征的排序算法。

## 贡献

欢迎任何形式的贡献，包括提交问题、改进代码、新增功能等。

比如你想要增加 `自定义 key size` 的功能，欢迎提出这样的需求。


## 许可证

Copyright (c) 2023, IACS, School of Computer Science and Technology, Anhui University
