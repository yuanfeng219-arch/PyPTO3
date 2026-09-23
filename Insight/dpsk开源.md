具体看 DeepSeek，低成本主要来自四件事叠加：**模型本身每个 token 少算、KV Cache 少占；关键算子自己做到接近硬件上限；MoE 跨卡通信被专门优化；在线服务又通过 Prefill/Decode 解耦和负载均衡把 GPU 利用率拉高。** 这也是为什么“拿到同样的开源权重”不等于“拿到 DeepSeek 的成本结构”。

先看模型层。DeepSeek-V3 总参数是 671B，但每个 token 实际只激活约 37B 参数；配置里是 256 个 routed experts，每个 token 只选 8 个专家。因此它拥有很大的总参数容量，但单 token 不需要像 671B Dense 模型那样把全部参数都算一遍。([GitHub][1]) 这直接降低了每 token 的 FLOPs 和显存带宽压力。

第二个关键是 MLA，也就是 Multi-head Latent Attention。普通 MHA 在长上下文推理时，需要为历史 token 保存较大的 K/V Cache；MLA 把 K/V 表达压缩到较低维 latent representation。DeepSeek-V3 的配置里 `kv_lora_rank=512`，这正是其 KV 压缩结构的一部分。KV Cache 变小之后，同一块 GPU 能容纳更多请求，decode 时读取历史 KV 的内存流量也更低。([GitHub][2])

所以模型设计本身已经在做：

```text
普通大模型
每个 token
    ↓
大量参数计算
+ 大 KV Cache
    ↓
GPU 算力 / HBM 压力高

DeepSeek
    ↓
MoE：671B 中只激活约 37B
MLA：压缩 KV Cache
    ↓
每个 token 少算一些
每个请求少占一些显存
    ↓
同样 GPU 能服务更多 token
```

但这只是第一层。**真正体现 DeepSeek 工程能力的，是它没有停在“模型理论上计算量比较低”。**

比如 GEMM。MoE 最核心的计算最后还是大量矩阵乘法，而且因为每个 expert 收到的 token 数不同，会产生很多特殊 shape 的 GEMM。DeepSeek 自己做了 **DeepGEMM**，专门针对 V3/R1 的 FP8 dense GEMM 和 MoE grouped GEMM 优化；官方明确说 DeepGEMM 就是在支撑 V3/R1 的训练与推理。([GitHub][3])

你可以把这一层理解成：

```text
模型代码：

expert(x)
    ↓
Linear
    ↓
MatMul

普通部署：
PyTorch → 通用 GEMM Kernel → GPU

DeepSeek：
PyTorch
    ↓
针对 DeepSeek shape / FP8 / MoE
专门优化的 DeepGEMM
    ↓
GPU
```

理论上的 FLOPs 没变，但**同样一次 MatMul 在 GPU 上花多少微秒、能吃到多少 Tensor Core 算力，差距可以很大。**

Attention 也一样。MLA 设计得再漂亮，如果没有对应高性能 Kernel，最后也可能跑得很慢。所以 DeepSeek又做了 **FlashMLA**，专门服务 MLA 的 prefill 和 decode。官方公开的早期 H800 benchmark 中，FlashMLA 在相应 workload 下报告了约 3000 GB/s 的 memory-bound 性能以及 580 TFLOPS 的 compute-bound 性能。([GitHub][3])

于是你之前那个公式可以进一步展开：

```text
模型结构
│
├─ MoE
│   └─ 每 token 只激活少量专家
│
├─ MLA
│   └─ KV Cache 压缩
│
↓
理论计算量 / 显存需求已经降低

然后：

MoE GEMM
↓
DeepGEMM
↓
FP8 Tensor Core 利用率提高

MLA
↓
FlashMLA
↓
Attention Kernel 针对模型定制

因此：
理论上的“省”
真正变成硬件上的“省”
```

第三层是 **MoE 通信**，这点尤其重要。

MoE 的问题在于，一个 token 当前在 GPU 0，但它选中的 expert 可能放在 GPU 17，于是需要：

```text
token
↓
Router
↓
发给对应 Expert GPU
        ← dispatch
↓
Expert 计算
↓
结果发回来
        ← combine
```

如果通信做不好，MoE 虽然少算了很多 FLOPs，却把时间全浪费在网络上。

DeepSeek 为此做了 **DeepEP**，专门处理 Expert Parallel 的 `dispatch / combine` All-to-All 通信，并针对两种完全不同的推理阶段做了不同 kernel：

```text
Prefill
大量 token
↓
重点：吞吐
↓
High-throughput DeepEP kernel

Decode
每轮 token 很少
↓
重点：延迟
↓
Low-latency DeepEP kernel
```

而且 DeepEP 支持 NVLink 节点内通信、RDMA 跨节点通信、FP8 dispatch，以及通信和计算 overlap。官方公布的 H800 数据里，在典型 V3/R1 decode workload 下，即使 EP 扩到 64、128、256，仍然针对微秒级 dispatch/combine 延迟进行专门优化。([GitHub][4])

这就到了一个很重要的区别：

> **MoE 让模型“理论上省计算”，DeepEP 才让这个优势在几十乃至上百张 GPU 上没有被通信吃掉。**

第四层其实最接近梁文锋说的“别人很难做到同样成本”：**在线 Serving 架构。**

DeepSeek 官方披露的 V3/R1 在线系统并不是简单：

```text
来一个请求
↓
找几张 GPU
↓
跑完整个推理
```

它把 **Prefill 和 Decode 分开部署**。

因为这两个阶段的硬件特征完全不同：

```text
Prefill
一次处理几千个 prompt token
→ GEMM 大
→ 算力密集
→ 追求吞吐

Decode
一次每个请求只产生约 1 token
→ GEMM 很小
→ KV Cache / 通信 / latency 更重要
→ 追求低延迟
```

所以 DeepSeek 给它们用了不同并行配置。官方披露的一个生产系统配置是：

```text
Prefill
Routed Expert：EP32
MLA / Shared Expert：DP32
4 nodes

Decode
Routed Expert：EP144
MLA / Shared Expert：DP144
18 nodes
```

为什么 decode 反而把 EP 拉到 144？因为 V3 每层有大量专家，而每 token 只选其中少数。把专家铺得更开以后，每张 GPU 只保存、读取少量专家权重，可以降低单 GPU 的 memory-access 压力；同时把大量在线请求聚合起来，让各 expert 获得足够 batch。([GitHub][5])

这里非常能体现 DeepSeek 的系统思路：

```text
                 用户请求
                    │
          ┌─────────┴──────────┐
          ↓                    ↓
       Prefill               Decode
    算 prompt             一个个出 token
          │                    │
        EP32                 EP144
        DP32                 DP144
          │                    │
          └─────────┬──────────┘
                    ↓
            大规模请求统一调度
                    ↓
      GPU / Expert / KV Cache 负载均衡
```

他们甚至专门做了多个 load balancer，例如关注 **KV Cache 在 GPU 间是否平衡、每张 GPU 收到的请求数是否平衡、Expert 的负载是否平衡**。因为线上系统真正浪费钱的一种典型情况就是：

```text
GPU 0   ██████████ 100%
GPU 1   ██████████ 100%
GPU 2   ███         30%
GPU 3   █████       50%
GPU 4   ██████████ 100%

你买了 5 张卡，
有效算力可能只用了 3~4 张卡的水平。
```

DeepSeek 会动态处理这些不均衡问题。([GitHub][5])

所以如果你现在重新理解“DeepSeek 为什么便宜”，我建议不要记成一串技术名词，而是记成下面这条因果链：

```text
① Model Architecture
MoE + MLA
↓
让“一个 token 理论上需要的计算/显存”更少

② Kernel
DeepGEMM + FlashMLA
↓
让这些特殊计算真正高效跑在 GPU 上

③ Distributed
DeepEP + 大规模 Expert Parallel
↓
让 MoE 跨 GPU 的通信代价尽可能被压低 / 隐藏

④ Serving
Prefill-Decode Disaggregation
+ DP / EP
+ Load Balancing
↓
让整组 GPU 更接近持续满载

⑤ Precision
FP8 GEMM + FP8 dispatch 等
↓
进一步减少计算和通信数据量

最终：
同样 1000 张 H800
↓
DeepSeek 每秒可以产出更多有效 token
↓
每百万 token 摊到的 GPU 成本更低
```

这里最后一句才是“低成本部署”的最准确定义：**不是服务器更便宜，而是单位 GPU·小时能生产更多满足 SLA 的 token。**

这也直接解释了梁文锋那句话为什么成立。假设你把 DeepSeek-V3 权重下载下来，你拿到的主要是：

```text
V3 weights
+ model architecture
```

但你想达到他的成本，还需要复现：

```text
V3
↓
MLA optimized implementation
↓
FlashMLA
↓
MoE grouped GEMM
↓
DeepGEMM
↓
Expert Parallel
↓
DeepEP
↓
Prefill / Decode disaggregation
↓
EP / DP topology
↓
KV / Request / Expert load balancing
↓
集群调度与线上容量管理
```

而且其中很多东西虽然现在 DeepSeek 也在陆续开源，**“代码公开”仍然不等于“生产系统已经调到同样水平”**。你还要有相同级别硬件、网络拓扑、请求规模、调参经验、监控体系和长期工程迭代。

这就是梁文锋所谓的那个门槛：**模型可以复制，单个优化也可以复制，但把 Model → Kernel → Communication → Serving → Cluster 共同调到一个全局成本最优点，要困难得多。**

[1]: https://github.com/deepseek-ai/DeepSeek-V3?utm_source=chatgpt.com "GitHub - deepseek-ai/DeepSeek-V3 · GitHub"
[2]: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/configs/config_671B.json?utm_source=chatgpt.com "DeepSeek-V3/inference/configs/config_671B.json at main · deepseek-ai/DeepSeek-V3 · GitHub"
[3]: https://github.com/deepseek-ai/open-infra-index/blob/main/README.md?utm_source=chatgpt.com "open-infra-index/README.md at main · deepseek-ai/open-infra-index · GitHub"
[4]: https://github.com/deepseek-ai/DeepEP/blob/main/README.md?utm_source=chatgpt.com "DeepEP/README.md at main · deepseek-ai/DeepEP · GitHub"
[5]: https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md?plain=1&utm_source=chatgpt.com "open-infra-index/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md at main · deepseek-ai/open-infra-index · GitHub"
