# cpp20gemma

单线程、零外部依赖的 C++20 推理引擎，实现一个 Gemma 风格（RoPE + GQA + GeGLU）的语言模型：

- **INT8 权重量化**（对称 per-tensor scale，省 75% 内存），量化只在 Python 转换脚本做一次
- **手写 SIMD 内核**：arm64 用 **NEON**（int8→int16→int32→f32 展开 + FMLA，对应文章的 AVX2 思路），x86-64 保留文章的 **AVX2** 路径，另有无 SIMD 的标量 fallback
- **完整模型结构**：可学习位置偏置、RMSNorm、RoPE（GPT-NeoX 交错对）、GQA（4 Q 头共享 1 KV 头）、GeGLU、KV Cache（缓存已旋转的 k）、Min-P 采样
- **零依赖**：一个 Makefile，`make` 即可；不链接 BLAS/MKL，无 protobuf
- **逐层单元测试**：C++ 每层输出与 NumPy 参考实现对比；端到端 64 步贪心解码与 Python 参考逐 token 精确一致

## 构建

要求：支持 C++20 的编译器（Apple clang 14+ / GCC 10+）、Python 3 + numpy（仅测试和转换脚本需要）。

```bash
make gemma        # 推理主程序 build/gemma
make unittests    # 单元测试 build/unit_tests
make integration  # 集成测试 build/integration
make test         # 生成测试数据并跑全部测试
```

arm64 上不需要任何特殊 flag（NEON 是 arm64 基线）；x86-64 自动加 `-march=native`。

## 快速开始

```bash
# 1. 生成测试模型（确定性随机权重，无需 HuggingFace 登录）
python3 py/make_test_weights.py --size tiny \
    --out-weights weights/tiny_weights.bin \
    --out-tok weights/tiny_tokenizer.bin

# 2. 推理（Min-P 采样）
./build/gemma --weights_path weights/tiny_weights.bin \
    --tok_path weights/tiny_tokenizer.bin \
    --model_size tiny --n_dec 64 --minp 0.1 --temp 0.7 --seed 42 \
    --prompt "The meaning of life"

# 3. 贪心解码（不给 --minp 即贪心）
./build/gemma --weights_path weights/tiny_weights.bin \
    --tok_path weights/tiny_tokenizer.bin \
    --n_dec 64 --prompt "What is a transformer?"
```

> 测试模型的权重是随机数，输出是"乱码"——它验证的是整条推理管线（量化→加载→前向→采样→解码），不是模型质量。

### CLI 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--weights_path` | 必填 | GMW1 格式权重文件 |
| `--tok_path` | `weights_path + ".tok"` | GMT1 格式 tokenizer |
| `--model_size` | — | 预设名校验：`270m` / `1B` / `tiny` |
| `--prompt` | "What is a transformer?" | 输入提示 |
| `--n_dec` | 250 | 最多生成的 token 数 |
| `--temp` | 0.7 | 温度；≤0 走贪心 |
| `--minp` | 不给=贪心 | Min-P 采样阈值（0~1，越小越保守） |
| `--max_cache_len` | 8192 | KV Cache 分配上限（位置数） |
| `--seed` | 时钟 | 采样随机种子 |
| `--terminate_on_eos` | 1 | 遇到 EOS 停止 |
| `--chat_format` | 0 | 包装 Gemma 对话模板（`<start_of_turn>` 格式） |
| `--dump` | — | 打印模型配置和张量布局后退出 |

## 真实 Gemma 权重（best-effort）

真实 Gemma 3n 是 gated 模型（需要 HF 登录 + 同意协议），本机无法验证。转换脚本已提供，形状会严格校验：

```bash
pip install safetensors sentencepiece
huggingface-cli login
hf download google/gemma-3n-E2B-it tokenizer.model --local-dir weights
hf download google/gemma-3n-E2B-it model.safetensors --local-dir weights

python3 py/convert.py --weights-in weights/model.safetensors \
    --weights-out weights/gemma_i8.bin --model-size 270m
python3 py/convert_tokenizer.py --tok-in weights/tokenizer.model \
    --tok-out weights/gemma_i8.bin.tok
```

注意：本引擎实现的是**文章描述的架构**（简化版），真实 Gemma 3n 有 q_norm/k_norm、滑窗注意力等结构，转换时会明确报错而非静默出错。`embed_positions` 缺失时自动写零（RoPE 已提供位置信息）。

## 架构与实现

```
hidden = embed_tokens[tok] + embed_positions[pos]      # 可学习位置偏置
每层:  h = h + attn(rmsnorm(h));  h = h + mlp(rmsnorm(h))
logits = lm_head(rmsnorm(h))
```

- **量化**：`scale = max(|w|, 1e-9)/127`，`w8 = clip(round(w/scale), -128, 127)`。反量化技巧：`(x·w8) × scale`——点积做完只乘一次 scale
- **GQA**：KV 投影只算 `n_kv_heads × head_dim` 行，KV Cache 内存只有 MHA 的 1/4
- **RoPE**：cos/sin 按位置惰性物化（不做 128K 全量预计算），rope 应用在 k 写入 cache **之前**
- **Min-P**：`max_p × min_p` 动态阈值，比 Top-K/Top-P 更适合小模型；`--seed` 固定 splitmix64，测试可复现
- **权重布局**：所有矩阵按 `[out][in]` 行主序存储，每行输出是连续内存的一段点积——cache 友好的 GEMV
- **加载**：整个文件一次 `aligned_alloc(64)` + 一次 `read`，文件大小由头字段完全决定（形状校验免费获得）

## 二进制格式

### 权重 GMW1 v1（小端）

64 字节定长头：`magic "GMW1" | version=1 | n_layers | dim | n_heads | n_kv_heads | mlp_dim | vocab_size | max_seq_len | head_dim | f32 rope_base | reserved(20, 必须全 0)`

张量区固定顺序（每个 = `N 字节 int8 + 4 字节 f32 scale`）：

1. `embed_tokens [vocab][dim]`，`embed_positions [max_seq_len][dim]`
2. 每层：`q_proj [dim][dim]`、`k_proj/v_proj [n_kv_heads×head_dim][dim]`、`o_proj [dim][dim]`、`gate/up_proj [mlp_dim][dim]`、`down_proj [dim][mlp_dim]`、`gamma_attn/ffn [dim]`
3. `final_norm_gamma [dim]`、`lm_head [vocab][dim]`

### Tokenizer GMT1 v1（小端）

48 字节头：`magic "GMT1" | version | vocab_size | bos_id | eos_id | unk_id | pad_id | reserved`

每条 token：`u32 len | u32 type (0 普通 / 1 byte-fallback / 2 控制) | f32 score | len 字节 UTF-8`

编码：贪心最长前缀匹配 + 逐字节 fallback（`<0xXX>` byte token）；控制 token 解码时跳过。

## 测试

```bash
make test
```

- **13 个单元测试**：加载器（含损坏文件拒绝）、NEON/f32 GEMV、反量化行、RMSNorm、RoPE、SiLU/GeGLU、softmax（含大数稳定性）、逐层前向（embedding/norm/attention/MLP/final/logits 全部与 NumPy 参考对比）、注意力原始分数 + k 的 rope-before-cache 约定、KV Cache、采样器（贪心 + Min-P 采样与 Python 逐 id 精确一致）、tokenizer 往返
- **集成测试**：64 步贪心解码 token 序列与 Python 参考**逐 id 精确相等**，最终 logits 容差 2e-3（实测最大偏差 ~3e-5）；near-tie 时打印两侧 logits 差值诊断
- 容差策略：参考实现使用**反量化后的权重**（与引擎同源），唯一差异是 fp32 累加顺序；matmul/attention/logits `atol 2e-3 / rtol 1e-3`，mlp（幅度最大的中间量）放宽到 `rtol 5e-3`

## 目录结构

```
src/          C++ 引擎：config / weights / kernel / ops / kv_cache / tokenizer / sampler / model / main
py/           Python：测试权重生成、NumPy 参考实现、golden 生成、safetensors/sentencepiece 转换
tests/        单元测试、集成测试、数据生成与运行脚本
```

## 已知限制

- 单线程（文章的设计选择：无锁竞争、cache 命中率高）
- 实现的是文章的简化架构，不是位精确的真实 Gemma 3n（q_norm/k_norm 等不支持）
- 预填充（prompt 处理）未做 GEMM 批量化，长 prompt 在 CPU 上较慢；解码是优化重点
- 128K 上下文的全量 KV 分配不现实（≈10GB），默认 `--max_cache_len 8192`
