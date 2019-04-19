it is forked from baidu [DuReader](https://github.com/baidu/DuReader).

## 主要改进
- 改进数据加载方式: 使用yield在使用时才进行数据加载，减少内存使用（穷，没那么大的内存）。之前需要32G以上的内存才能跑起来，现在只需要4G内存就可以跑起来。
- 文件随机读取:采用mmap将文件地址映射到内存地址中，减少一次硬盘到内核空间的数据拷贝，并且能够进行全局的数据shuffle。需要提前使用`create_header.py`创建header文件，每行存储每条记录的（行号，起始位置，长度）

