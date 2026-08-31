# 从零开始学写推理引擎

这是一套完全面向 0 基础读者的学习文章，围绕本仓库的 C++20 推理引擎展开。
每一篇都遵循同样的结构：**为什么需要它 → 核心概念从零讲起 → 手算例子 → 我们的真实代码 → 踩过的坑**。

学习它最好的方式：一边读文章，一边打开对应的源码文件，跑 `make test` 验证你的理解，然后动手改代码看测试会不会红。

## 学习路线（按依赖顺序阅读）

| # | 文章 | 对应源码 | 一句话 |
|---|---|---|---|
| 00 | [总览：一个推理引擎在做什么](00-总览.md) | `src/main.cpp` | 全流程地图，先看整体 |
| 01 | [Tokenizer：文字怎么变成数字](01-Tokenizer.md) | `src/tokenizer.cpp` | 模型只认识数字 |
| 02 | [二进制格式设计：权重文件长什么样](02-二进制格式.md) | `src/weights.cpp` `py/common.py` | 定长头 + 固定顺序 = 零成本校验 |
| 03 | [嵌入与位置编码](03-嵌入与位置编码.md) | `src/model.cpp` | 词向量查表 + 位置偏置 |
| 04 | [RMSNorm：让数值不爆炸](04-RMSNorm.md) | `src/ops.cpp` | 一行公式的归一化 |
| 05 | [RoPE：旋转位置编码](05-RoPE.md) | `src/ops.cpp` | 用旋转把位置写进向量 |
| 06 | [注意力机制：模型怎么"看"上下文](06-注意力.md) | `src/model.cpp` | Q/K/V、softmax、因果 |
| 07 | [KV Cache 与 GQA：解码提速的关键](07-KV-Cache与GQA.md) | `src/kv_cache.h` | 别重复算历史 |
| 08 | [MLP 与 GeGLU：每个词独立加工](08-MLP与GeGLU.md) | `src/model.cpp` | 注意力交换信息，MLP 独立加工 |
| 09 | [INT8 量化：把 1GB 压缩到 250MB](09-INT8量化.md) | `py/common.py` `src/kernel.cpp` | 省内存的原理 |
| 10 | [SIMD 与矩阵乘法：为什么能快](10-SIMD与矩阵乘法.md) | `src/kernel.cpp` | 一条指令算 8 个数 |
| 11 | [采样策略：模型怎么"选词"](11-采样.md) | `src/sampler.cpp` | 贪心、温度、Min-P |
| 12 | [测试策略：怎么证明算对了](12-测试策略.md) | `tests/` `py/generate_golden.py` | NumPy 参考 + 容差 + 真实 bug 回顾 |

## 建议的动手实验（按顺序做）

1. 读完 00-02 后：跑 `./build/gemma --weights_path weights/tiny_weights.bin --dump`，对照文章看懂每一个字段
2. 读完 03-08 后：跑 `make test`，然后故意改错一个地方（比如把 `attention_block` 里的 `kvh = h * nkv / nh` 改成 `0`），看测试怎么红、错误从哪一层冒出来
3. 读完 09-11 后：改 `py/make_test_weights.py` 里的温度/采样参数，观察输出变化；把 `--minp` 去掉对比贪心输出
4. 读完 12 后：给一个新算子（比如 GELU 替代 SiLU）写 golden 测试——这是检验你是否真正掌握的方式

## 前置知识

只需要：会一门编程语言（任意），知道函数和数组是什么。浮点数、SIMD、矩阵都会在文章里从零讲。

## 本引擎的定位（诚实声明）

本引擎实现的是**教学用简化架构**，不是位精确的真实 Gemma 3n。
教学上它足够完整：量化、SIMD、GQA、RoPE、KV Cache、采样、逐层测试——一个推理引擎该有的零件全都有，且总共只有约 1500 行 C++，是你能通读的规模。
