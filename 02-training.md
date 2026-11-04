# 训练方法与技巧

本文记录在一台 GPU、单任务、单数据源、简单 diffusion/flow loss 与最简单训练循环之上，公司代码实际加入的改动。

## 索引

| ID | 方法 | 主要收益 | 状态 |
|---|---|---|---|
| T-001 | proprio anchor 状态注入 | 统一状态条件和 VLM 文本接口 | 启用 |
| T-002 | embodiment 专属动作投影 | 降低跨本体参数冲突 | 已实现/GB0.8启用 |
| T-003 | 动作维和时间维 loss mask | 阻止 padding 与无效维污染 | 启用 |
| T-004 | flow matching 训练目标 | 学连续动作速度场 | 启用 |
| T-005 | TTRTC 随机 clean prefix 训练 | 训练时匹配实际延迟执行 | 已实现 |
| T-006 | multi-flow 同批多噪声采样 | 提升每个视觉样本的动作监督密度 | 已实现 |
| T-007 | 连续 flow 与 FAST token 联合训练 | 结合序列 token 与连续动作监督 | 已实现 |
| T-008 | loss mask 的 NaN-safe 归约 | 避免 `NaN × 0` 污染 batch | 已实现 |
| T-009 | 动作维 loss 权重 | 对不同物理意义维度单独调权 | 已实现 |
| T-010 | DCT 频域动作正则 | 约束动作频率结构 | 已实现/未默认启用 |
| T-011 | 语言交叉熵分块与重计算 | 降低长 prompt token loss 峰值显存 | 已实现 |
| T-012 | 索引式 language logits | 不把全部 token logits 都实体化 | 已实现 |
| T-013 | VLM 到 action expert 的梯度权重 | 控制辅助头对主干的影响 | 已实现 |
| T-014 | 多头辅助目标隔离 | 防止 action suffix 看到自回归 target | 已实现 |
| T-015 | 全模型 activation checkpoint | 以计算换显存 | 启用 |
| T-016 | 指定层 activation checkpoint | 细粒度控制显存和重算代价 | 已实现 |
| T-017 | FSDP2 与分片 EMA | 大模型可训练同时保留稳定评估权重 | 已实现 |
| T-018 | AdamW、fused Adam、CAME 8bit | 不同规模下的优化器与显存折中 | 已实现 |
| T-019 | Warmup + cosine 调度 | 抑制早期大梯度，后期细调 | 启用 |
| T-020 | torch.compile / Inductor 训练 | 算子融合、减少解释开销 | 已实现 |
| T-021 | TF32 和 FP8 | 提升吞吐、控制显存 | 已实现 |
| T-022 | pin memory、persistent workers 与 prefetch | 让 H2D 和数据解码重叠 | 已实现 |
| T-023 | DataLoader 并行数据副本 | 单进程内部多 replica 的负载分配 | 已实现 |
| T-024 | checkpoint 保留策略 | 只保留关键步数并控磁盘 | 已实现 |
| T-025 | 分样本来源记录与 NaN 定位 | 把 loss 异常追溯到数据 | 已实现 |
| T-026 | 稳定且可重现的 flow RNG | rank/call 级独立 seed | 已实现 |
| T-027 | 训练过程分解 profiling | 判断瓶颈在数据、前向、反向还是 EMA | 已实现 |
| T-028 | 无效数据 loss 跳过与失败快速反馈 | 减少坏批次对训练动力学影响 | 已实现 |
| T-029 | 训练、推理前缀契约校验 | 阻止错配 checkpoint 进入训练/部署 | 已实现 |
| T-030 | 辅助目标与主体 policy 的隔离 | 控制目标但不污染通用 policy | 已实现 |

---

## T-001 proprio anchor 状态注入

**基础做法**：把机器人状态直接拼成文本，例如：

```text
State: 0.1, -0.2, ...
```

**公司改进**：在语言 token 序列中预留 `<|propri|>` 位置；连续的 22/32 维 proprioception 经过投影后替换该位置 embedding。通过 `proprio_anchor` 模式，state 保持为数值张量，而文本 token 仍保留完整句法。

**为什么可能有效**：

- 避免把连续高维关节数值量化成文本 digit，减少 token 长度和数值精度损失。
- VLM 的文本接口保持统一，动作专家可以从精确状态 token 读取条件。
- 可以对 state 单独裁剪、mask、分位数归一化，而不改变 prompt。

**代价/风险**：

- 训练、推理和 checkpoint 的 `state_input_mode`、state width、token id 必须一致。
- 状态缺失以及 padding 位置需要明确 mask，不能用零值伪装真实状态。
- 旧 checkpoint 的共享投影和 0.8 的 embodiment 专属投影不可混用。

**代码**：

- `giga-code/giga-models/giga_models/models/vla/giga_brain_0/modeling_giga_brain_0.py`
- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_transforms.py`

**状态**：`启用`。

---

## T-002 embodiment 专属动作投影

**基础做法**：所有机器人共享一个 `action_in_proj` 和 `action_out_proj`。

**公司改进**：`EmbodimentSpecificLinear` 按 embodiment id 选择独立权重和 bias；0.8 还引入 embodiment-specific state projection，使 H01、Agilex、UMI、Ego 等本体拥有各自的状态到 VLM 投影。

**为什么可能有效**：

- 不同本体的关节维度、单位和动作含义相差大。
- 共享投影容量有限时，梯度可能互相冲突。
- 固定 32 维 padding 后，独立投影可以更有效地为每个本体学习单位归一化映射。

**代价/风险**：

- 参数和显存增加。
- 每个 embodiment 训练数据不足时更容易过拟合。
- embodiment id、state projection 权重形状和 checkpoint 兼容升级必须显式处理。
- 这不是“本体越多一定越好”，部署仍要限制在训练过的本体上。

**代码**：

- `giga-code/giga-models/giga_models/models/vla/giga_brain_0/modeling_giga_brain_0.py`
- GB0.8 对比：`gb08-code-compare-20261103/reports/GB08_TRAIN_DATASETS_COMPARE_20261103.md`

**状态**：动作投影 `已实现`；GB0.8 状态专属投影 `启用`，GB0.7 三饮品未启用。

---

## T-003 动作维与时间维 loss mask

**基础做法**：对完整 `[B,H,32]` tensor 直接做 MSE。

**公司改进**：

- 时间维使用 `action_loss_mask` 过滤 padding 尾步。
- 维度使用 `action_dim_loss_mask` 过滤未监督动作维。
- 样本级 `skip_loss_mask` 屏蔽无效图像样本。
- flow loss 先选有效区域，再按有效维度数量归约。

**为什么有效**：

- 某些机器人只使用前 14/22 维，剩余 padding 维覆盖了真实动作。
- episode 尾部的缺失未来动作不应被当成零目标。
- 按有效 dim 数量归一化，可以避免不同 action layout 的量级偏置。

**代价/风险**：

- mask shape、dtype 和广播方向写错时可能静默污染 loss。
- `mask * loss` 与 `torch.where` 在 NaN/Inf 下行为不同。
- 时间 mask 和 dim mask 都要影响 flow noise 的构造，不只是影响最终 loss。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_loss.py`
- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_trainer.py`

**状态**：`启用`。

---

## T-004 flow matching 训练目标

**基础做法**：

- 直接回归绝对动作，或用 DDPM 假设多步随机去噪。
- 简单 MSE 输出动作。

**公司改进**：

- 采样连续 time、noise 和插值状态 `x_t = t*noise + (1-t)*action`。
- 网络预测速度 `v_t = noise - action`，用 MSE 学速度场。
- 推理使用固定数量的 `num_steps=10` 去噪。

**为什么可能有效**：

- 连续动作分布是多峰的，直接回归容易输出多模式平均，动作会发钝。
- flow matching 保留生成式能力，同时比完整 diffusion 采样更简洁。
- 与动作 expert 的双流结构可自然融合视觉前缀和动作 suffix。

**代价/风险**：

- 训练和推理的时间参数约定必须相同。
- 输出必须反归一化和应用 delta/absolute 变换。
- 噪声、时间、mask 和 prefix 条件都属于合约，不能单独修改某一端。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_loss.py`
- `giga-code/giga-models/giga_models/models/vla/giga_brain_0/modeling_giga_brain_0.py`

**状态**：`启用`。

---

## T-005 TTRTC 随机 clean prefix 训练

**基础做法**：训练时未来动作全部加噪；部署时拿到已有动作但只从头重新预测，可能产生拼接跳变。

**公司改进**：训练时随机选择 delay `d ∈ [0,max_delay]`，将动作块前 `d` 步设为 clean 动作、对应 timestep 设为零，只监督后续 postfix；部署时把实际已执行动作作为 hard prefix 送入 proposal。

**为什么有效**：

- 训练直接覆盖“先有一部分已经执行的动作，再预测未来”的推理场景。
- 减少重规划时前后 chunk 起点不一致。
- prefix 与实际控制延迟关联，降低实时队列切换的突跳。

**代价/风险**：

- 必须保证 postfix supervision mask 不包含prefix。
- 部署中 prefix 长度不能超过训练 max delay 和实际队列长度。
- 若推理时间明显超过 prefix 覆盖时长，best-effort 会执行到未 conditioned 区，需要明确失败或 splice 策略。

**代码**：

- `giga-code/giga-models/giga_models/models/vla/giga_brain_0/ttrtc.py`
- `zhangzhong_wbc/server/src/model.py`

**状态**：`已实现`，多数 WBC 部署配置启用。

---

## T-006 multi-flow 同批多噪声采样

**基础做法**：每个视觉样本只采一个 flow-noise/time，做一次 flow loss。

**公司改进**：同一视觉 prefix 复制 `K` 份，用独立噪声和时间预测 `K` 个 action suffix；对共享 prefix 的 KV cache 可复用，并将动作侧的 VLM gradient 权重按配置控制。

**为什么可能有效**：

- 图像前向和 VLM prefix 是昂贵部分；多动作噪声共用 prefix 提高有效监督。
- 多个 flow samples 对同一决策的多模式动作分布覆盖更好。
- 可以降低每步 GPU 上的数据加载需求相对于监督信号的比例。

**代价/风险**：

- action batch 扩大 `K` 倍，显存和 backward 成本增加。
- 输出 loss 的统计口径与单个 flow 不一致，不能只看 loss 数值和旧实验比。
- 共享 prefix 的梯度处理不当会让动作 loss 过度改写 VLM。
- RGB 预训练数据一般不需要这项，通常只用于动作预训练。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_trainer.py`
- `giga-code/giga-models/giga_models/models/vla/giga_brain_0/modeling_giga_brain_0.py`

**状态**：GB0.8 预训练 `启用`；当前三饮品后训练日志为 `flow_matching_batch_mul=1`，未启用。

---

## T-007 连续 flow 与 FAST token 联合训练

**基础做法**：只训练连续动作，或只把动作转成离散 token 做自回归。

**公司改进**：同一个 VLM prefix 同时输出：

- flow action expert 的连续速度场 `v_t`；
- 从动作 chunk 生成 FAST token 后的语言 `cross_entropy`。

两者通过 mask 和 attention 分离，避免 flow suffix 从 autoregressive target 里偷看答案。

**为什么可能有效**：

- FAST 目标擅长学习动作序列结构，连续 flow 目标保留高频和连续控制精度。
- 两者共享视觉语义表征，有互补监督。
- 最终部署可继续只走 flow 动作，因此不必须承担自回归推理延迟。

**代价/风险**：

- total loss 中 FAST 项可能远大于 flow 项，不能直接拿 total 和纯 flow 训练比较。
- 两个目标的最优梯度权重不同，需要监控 `diffusion_loss` 与 `fast_token_llm_loss` 分开曲线。
- tokenizer、输出词表、special ids 和 checkpoint 词表尺寸全部要匹配。
- teacher forcing 下的 token loss 改善，不能自动推断 autoregressive rollout 也变好。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_loss.py`
- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_trainer.py`
- 对比证据：`gb08-code-compare-20261103/reports/GB08_VS_GB07_3DRINKS_LOSS_20261103.md`

**状态**：`已实现`。GB0.8 三饮品启用；GB0.7 当前任务关闭。

---

## T-008 loss mask 的 NaN-safe 归约

**基础做法**：

```python
loss = (per_elem_loss * mask).sum() / mask.sum()
```

**公司改进**：先在有效位置取张量，再归约；无效区域使用 `torch.where` 置零，而非乘零。动作 dimension mask 和 token mask 都做同样处理。

**为什么有效**：

- `NaN * 0` 仍是 NaN，一个异常无效 token 可污染整个 batch。
- 先选有效位置后归约，不会让未监督位置的值参与梯度或反向传播。
- 对混合 fp16/bf16 下偶发 Inf 更稳。

**代价/风险**：

- 选位置会造成额外 gather 和临时内存。
- mask 必须精确，否则会无声丢掉有监督样本。
- NaN-safe 不等于根因修复：如果有效 action 本身 NaN，应进一步过滤并记录来源。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_loss.py`

**状态**：0.8 分支对此有明确改进；0.7 主线仍需按当前版本判断。

---

## T-009 动作维 loss 权重

**基础做法**：所有动作维度等权 MSE。

**公司改进**：按 embodiment 对指定动作维降低或提高 loss 权重，例如对双手机器人的旋转和夹爪维度设置不同权重。

**为什么可能有效**：

- 不同关节的数值范围、噪声和任务重要性不同。
- 旋转和夹爪在低层坐标中更容易产生大误差，但并不等价于控制重要性。
- 允许同一模型保持统一 loss 框架，仅调整每维监督比例。

**代价/风险**：

- 权重过度手工调参会导致某些维欠拟合。
- 训练和验证的 mse 只能作辅助，必须看真机维度误差和成功率。
- 权重必须绑定正确的 embodiment 和 dim 顺序。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_loss.py`
- GB0.8 预训练配置中的 `flow_action_dim_loss_weight_cfg`

**状态**：`已实现`，按配置启用。

---

## T-010 DCT 频域动作正则

**基础做法**：只在时域动作上算 MSE。

**公司改进**：在完全有效的动作候选区内，将动作序列转 DCT，对低频和高频分别设权，再计算频域相似度；权重可配置。

**为什么可能有效**：

- 时域点误差可能忽略平滑度和频率结构。
- 对高频施加更大权重可以抑制抖动；赋予低频不同权重可控制整体轨迹。
- 只在完整有效序列上启用，避免 padding 造成频谱假信号。

**代价/风险**：

- DCT 额外计算和临时内存。
- 权重过大可能牺牲快速动作。
- 仅在 padding 少的序列上生效，数据 padding 比例高时作用有限。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_loss.py`

**状态**：`已实现`，默认主配置未必启用。

---

## T-011 语言交叉熵分块与重计算

**基础做法**：一次把所有语言 token hidden states 投到 vocab 维，生成 `[N_tokens, vocab]` logits。

**公司改进**：将需要监督的语言 token 先 gather，按 `language_loss_chunk_size` 分块计算 CE；对每个 chunk 可选非重入 activation checkpoint。

**为什么可能有效**：

- VLM 的 vocab head 很大，长 prompt 下完整 logits 是显存峰值。
- chunk 后每次只保留一部分 logits。
- gather 后只计算实际 loss 位置，避免为视觉、padding 或不需要监督的位置算语言 head。

**代价/风险**：

- 分块循环增加 Python 和调度开销。
- checkpoint 增加前向重算时间。
- token mask、target shift 和 position mapping 必须完全正确。

**代码**：

- `giga-code/giga-models/giga_models/models/vla/giga_brain_0/modeling_giga_brain_0.py`

**状态**：GB0.8 `language_loss_chunk_size=1024`；GB0.7 的基础路径多为 0。

---

## T-012 索引式 language logits

**基础做法**：对 `prefix_out[:, :-1, :]` 全序列算 vocab logits，再用 mask 选择。

**公司改进**：先用 `lang_loss_masks[:,1:]` 选出需要计算 loss 的 hidden states，再只对选中 hidden 做投影，并返回对应 mask。

**为什么可能有效**：

- 视觉 token、instruction-only token 和 padding 都不需要语言 head。
- 减少 vocab projection 的输出大小和临时显存。
- 更易于实现 chunk 化 CE。

**代价/风险**：

- 需要把 compact logits 与原 token index 对齐。
- 一旦 shift 错一位，语言 loss 仍能训练但学到错误 token。
- 某些任务需要全部 logits 输出时，应保留非 compact 路径。

**代码**：

- `giga-code/giga-models/giga_models/models/vla/giga_brain_0/modeling_giga_brain_0.py`

**状态**：`已实现`，由 `compact_language_logits` 控制。

---

## T-013 VLM 到 action expert 的梯度权重

**基础做法**：action loss 对 VLM 的梯度全量回传，或完全截断。

**公司改进**：在 action expert 读取 prefix KV 时使用自定义 autograd 权重，对 key/value 的跨分支梯度进行缩放。GB0.8 还允许 weight 和 FAST 分支共同塑造 VLM。

**为什么可能有效**：

- 全量回传时，动作 expert 可能有很大梯度，破坏视觉语言预训练表征。
- 全截断则动作任务无法调整视觉表征来区分任务相关物体。
- 小权重可保留部分任务适配能力，减少 catastrophic drift。

**代价/风险**：

- 梯度权重不是普通 loss 权重，不能根据 total loss 大小直接调。
- 过小可能使 VLM 对细粒度机器人动作不敏感。
- 必须检查 prefix KV 和 action suffix 的梯度流是否出现 NaN。

**代码**：

- `giga-code/giga-models/giga_models/models/vla/giga_brain_0/paligemma2_with_expert.py`
- `giga-code/giga-models/giga_models/models/vla/giga_brain_0/modeling_giga_brain_0.py`

**状态**：`已实现`，由 `action_loss_vlm_gradient_weight` 控制。

---

## T-014 多头辅助目标隔离

**基础做法**：所有 token、图像和动作共享普通 causal mask。

**公司改进**：

- 连续 flow suffix 不能 attend 到 FAST 预测 target；
- FAST token 行不能 attend 到预测的 subtask token；
- 计算 action suffix position ids 时排除被 mask 的 target token；
- 对 prefix-only mask 构造成只计算 query rows，避免全矩阵重建。

**为什么可能有效**：

- 模型不能通过看见目标 token embedding 降低训练 loss。
- 多目标共存时 attention 语义明确：输入条件可以共享，预测答案不能共享。
- 真实位置编号不因为被屏蔽的占位 token 错位。

**代价/风险**：

- 训练端与推理端 attention mask 必须一致，否则训练-推理偏移。
- 复杂的 block-causal mask 很难靠肉眼确认，必须单测。
- 被 mask 的 target 不能让 `NaN` 或 padding 值参与计算。

**代码**：

- `giga-code/giga-models/giga_models/models/vla/giga_brain_0/modeling_giga_brain_0.py`

**状态**：`已实现`，多目标训练时启用。

---

## T-015 全模型 activation checkpoint

**基础做法**：保留所有层前向激活供 backward 使用。

**公司改进**：对视觉层和 action expert 的全部或指定层使用 checkpoint，只保存边界状态，backward 重新计算层内激活。

**为什么有效**：

- 大 VLM 和 action expert 训练的主要峰值来自激活而非权重。
- 允许更大 batch、更长文本或更高精度训练。
- 与 FSDP 配合可明显降低每卡显存。

**代价/风险**：

- 增加前向重算时间，训练更慢。
- 对含随机性和原地写操作的层，必须使用非重入 checkpoint。
- 所有层都 checkpoint 不一定是吞吐最优；只包最关键层可能更好。

**代码**：

- `giga-code/giga-train/giga_train/strategies/activation_checkpointing.py`
- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_trainer.py`

**状态**：GB0.8 三饮品配置启用，当前配置 checkpoint 全部 `Gemma2DecoderLayerWithExpert`。

---

## T-016 指定层 activation checkpoint

**基础做法**：只能是全开或全关。

**公司改进**：支持 `ClassName` 和 `ClassName__##__N` 这种带数量限制的 wrap 规则；只对前 N 个同类 layer 启用 checkpoint。

**为什么有效**：

- 显存瓶颈集中在大型 transformer layer，而小型 visual embedding 层重算成本可能不划算。
- 每个任务可找到显存与吞吐的折中。
- 训练框架与模型源码无需改类结构即可局部启用。

**代价/风险**：

- module 计数依赖模型结构，升级模型后可能偏移。
- 复杂 VLM 的 class name 变化会让规则静默不匹配。
- 生产恢复时需验证 checkpoint 配置。

**代码**：

- `giga-code/giga-train/giga_train/strategies/activation_checkpointing.py`

**状态**：`已实现`。

---

## T-017 FSDP2 与分片 EMA

**基础做法**：每张卡放完整模型；EMA 在每张卡复制完整权重。

**公司改进**：

- 使用 FSDP2 shard 参数；
- 分片 EMA buffer 在每 rank 只保存对应分片；
- 支持从 sharded model 重建 EMA；
- 训练时 `get_model_state_dict(full_state_dict=True)` 用于 EMA 更新后按原模分片写回。

**为什么可能有效**：

- 让单机无法容纳的大模型可训练。
- 不必额外为每张卡复制一整份 EMA 模型。
- 评估用的 EMA 权重来自完整模型而非某 rank 残片。

**代价/风险**：

- 全量 state dict 的 gather 成本很高，可能占训练时间的显著比例。
- optimizer、EMA 和随机状态各自大小不同，checkpoint 空间很大。
- FSDP 版本、state dict 类型和 checkpoint 兼容规则不能乱改。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/fsdp2_sharded_ema.py`
- `giga-code/giga-train/giga_train/strategies/ema.py`
- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_trainer.py`

**状态**：`已实现`；是否分片 EMA 依配置而定。

---

## T-018 AdamW、fused Adam、CAME 8bit

**基础做法**：PyTorch 普通 AdamW。

**公司改进**：

- 框架暴露标准 AdamW 配置；
- 可选 fused Adam 降低逐参数更新开销；
- 可选 CAME 8bit 降低 optimizer state。

**为什么可能有效**：

- 大模型 optimizer state 可占数倍显存。
- fused kernel 对大量小张量更新可减少 Python/CUDA 调度。
- 8bit 优化器允许更大 batch 或更少 offload。

**代价/风险**：

- 8bit 数值行为需要可靠性验证。
- fused 与 FSDP/CPU offload/平台兼容性不一定全覆盖。
- 更换 optimizer 属于训练动力学变化，不只是性能开关。

**代码**：

- `giga-code/giga-train/giga_train/optimizers/build.py`
- `giga-code/giga-train/giga_train/optimizers/fused_adam.py`
- `giga-code/giga-train/giga_train/optimizers/came_8bit.py`

**状态**：`已实现`，按配置选择。

---

## T-019 Warmup + cosine 调度

**基础做法**：固定 learning rate。

**公司改进**：以 step 为单位的 warmup，然后 cosine 衰减到配置的最小比例。支持按 epoch 或 step 两种基准计算。

**为什么有效**：

- 训练初期 AdamW 二阶矩尚未稳定，大学习率易出现 loss spike。
- cosine 长出尾适合后训练细调，避免训练末端持续大幅扰动。
- step-based 调度更适合流式无限数据和早停。

**代价/风险**：

- `warmup_steps`/`decay_steps` 必须与总 step 匹配。
- 以小数据集和大 batch 调出的 schedule 迁移到大混合数据不一定合适。
- 恢复训练时 scheduler step 序号必须同步，不能只恢复模型。

**代码**：

- `giga-code/giga-train/giga_train/schedulers/schedulers.py`

**状态**：`启用`。

---

## T-020 torch.compile / Inductor

**基础做法**：逐算子 eager 执行。

**公司改进**：可选 `torch.compile`，支持设置 backend、mode 与 full_compile；服务端可选择只编译 denoise step 或整个 pipeline。

**为什么可能有效**：

- 融合 elementwise、归一化和 attention 辅助算子。
- 减少 Python 与 kernel launch overhead。
- 对固定 shape 推理收益尤其明显。

**代价/风险**：

- 首次编译时延明显。
- dynamic shape、data-dependent control flow 和随机数会触发重编译。
- 训练端 compile 与 checkpoint 恢复没有直接关系，但会影响性能和故障模式。
- 必须通过 smoke test 和数值一致性验证，不能用“能开始跑”代替。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_trainer.py`
- `zhangzhong_wbc/server/src/model.py`
- `zhangzhong_wbc/server/script/start_server_wbc.sh`

**状态**：报告中的 GB0.8 训练设置 `dynamo.backend=inductor`；服务端 WBC 配置 `backend=inductor`。

---

## T-021 TF32 与 FP8

**基础做法**：所有训练均用 FP32。

**公司改进**：

- 默认允许 TF32；
- 可配置 FP16/BF16/FP8；
- FP8 支持 Transformer Engine recipe 路径；
- 训练框架可绕过 accelerate 默认转换，自定义 FP8 autocast wrapper。

**为什么可能有效**：

- Ampere/Hopper 上 TF32 提高矩阵乘吞吐。
- BF16 比 FP16 更少溢出。
- FP8 降低显存并提高 tensor core 利用率。

**代价/风险**：

- 精度与模型任务高度相关，不能只看吞吐。
- FP8 的 scaler、norm、attention 和 reduce 容易出现 NaN。
- 必须保存完整模型 dtype，避免推理端误用训练低精度状态。
- CPU 端 state dict 与模型 `register_to_config` 的精度字段也不能漂移。

**代码**：

- `giga-code/giga-train/giga_train/trainers/trainer.py`
- `giga-code/giga-train/giga_train/trainers/trainer.py`

**状态**：TF32 `已实现`；FP8 `已实现`，按任务启用。

---

## T-022 pin memory、persistent workers 与 prefetch

**基础做法**：DataLoader 默认 `num_workers=0` 或每次 epoch 结束 kill workers。

**公司改进**：主 GB0 训练路径显式 `pin_memory=True`，可选 `persistent_workers=True` 和 `prefetch_factor`。

**为什么有效**：

- pinned CPU memory 让 H2D copy 可异步。
- persistent workers 避免每个 epoch 重新打开数据文件和解码器。
- prefetch 将 CPU 数据准备与 GPU 计算重叠。

**代价/风险**：

- pinned memory 占用系统内存，worker 多时可能挤压 page cache。
- persistent worker 状态可能残留 episode cache。
- prefetch factor 过大时更多样本提前解码，内存峰值上升。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_trainer.py`
- GB0.7 分支的 `pin_memory` 兼容补丁

**状态**：主 GB0 训练路径显式支持并常见启用。

---

## T-023 DataLoader 并行数据副本

**基础做法**：一个 batch 只交给一个模型 replica。

**公司改进**：`ParallelBatchSampler` 将一个数据 batch 重复 `data_parallel_size` 次，供进程内多个模型 replica 消费。

**为什么可能有效**：

- 支持单进程内多 replica 的实验结构。
- 不需要由外部启动多个完全独立训练进程即可做并行数据。
- 与 shared prefix 多 flow 的情况不同，它复制的是数据批本身。

**代价/风险**：

- 相同 batch 被复制会给模型带来重复样本。
- 如果希望每个 replica 看到不同数据，这个策略不适合。
- 使用前应确认 trainer 是否真的具备匹配的模型分片和合并逻辑。

**代码**：

- `giga-code/giga-train/giga_train/samplers/parallel_batch_sampler.py`

**状态**：`已实现`，仅 `data_parallel_size>1` 时启用。

---

## T-024 checkpoint 保留策略

**基础做法**：每 N 步保存所有 optimizer、EMA 和模型，或只留最新。

**公司改进**：可分别控制 checkpoint 连续保留数、固定 step 保留列表、是否保存 optimizer；按 interval 保存；记录随机状态；对 FSDP2 支持分片模型写入。

**为什么有效**：

- 模型 14–16GB 级别，optimizer 可超过 20GB，全量 checkpoint 很快耗尽磁盘。
- 固定保留 milestone 便于做 scaling 或 eval 对比。
- 各 rank 随机状态可提高断点恢复一致性。

**代价/风险**：

- 最终评估模型和可断点训练的模型不是同一个东西。
- 如果 checkpoint 写盘过慢，会阻塞训练和造成 scheduler 假停顿。
- 保存策略变化不应改变模型训练本身。

**代码**：

- `giga-code/giga-train/giga_train/trainers/trainer.py`
- `checkpoint_keeps`、`checkpoint_total_limit`、`checkpoint_save_optimizer`

**状态**：`已实现`。

---

## T-025 分样本来源记录与 NaN 定位

**基础做法**：只记录 batch loss；发现 NaN 时不知道是哪条数据。

**公司改进**：在 transform 里附加 `debug_repo_id`、`episode_index`、`frame_index` 等 Python 标量，不进入 GPU 前向；在出现 NaN loss 时辅助定位样本来源。

**为什么可能有效**：

- 大数据集 NaN 往往来自少数损坏或异常 action。
- 可快速定位到数据集、episode 和 frame，而不必二分训练配置。
- 额外元数据不进模型 tensor、不占 GPU。

**代价/风险**：

- debug 字段必须由 collator 正确透传，不能误 stack 成 tensor。
- 来源字段名称和 dataset 映射要保持稳定。
- 只定位错误，不会自动修正原数据。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_transforms.py`
- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_trainer.py`

**状态**：`已实现`。

---

## T-026 稳定且可重现的 flow RNG

**基础做法**：直接使用全局 `torch.randn()`；多 rank 和 resume 后难以重现具体 flow noise。

**公司改进**：可选 `flow_rng_seed`，按 base seed、rank、call index 派生独立 generator 和 seed；用 SplitMix 风格混合避免简单拼接碰撞。

**为什么可能有效**：

- 每 rank 的 flow noise 可独立重现。
- 同一次训练配置下可复现 NaN 或 loss 轨迹。
- 可以对照两种实现是否产生同样噪声。

**代价/风险**：

- 与全局 RNG 相比，噪声序列不同，不能沿用旧实验的 loss 对比。
- 如果未同步 call index，resume 后仍可能漂移。
- 这是可重现性优化，不是模型质量提升本身。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_loss.py`

**状态**：`已实现`，由配置启用。

---

## T-027 训练过程分解 profiling

**基础做法**：只报告每秒 step 数。

**公司改进**：按窗口统计 `data_next`、`data_getitem`、图片解码、transform、collate、forward、backward、optimizer、scheduler、EMA、checkpoint 保存等阶段；使用 CUDA event 收集前向/反向，并报告均值、p95、最大和占比。

**为什么有效**：

- 总 step 时间慢时，可以判断是算法计算、数据 I/O、checkpoint 还是日志。
- 发现 GPU 空转时能区分数据 worker 和主进程瓶颈。
- 支持对每个 camera 的预处理单独定位。

**代价/风险**：

- profiling 本身有少量开销，不宜长期高频率启用。
- 时间区间可能嵌套，统计要避免重复归因。
- 每步数据 shape 变化会影响解释。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_trainer.py`

**状态**：`已实现`。GB0.8 训练日志有 `Pipeline profile window_step` 证据。

---

## T-028 无效数据 loss 跳过与失败快速反馈

**基础做法**：坏样本进入 loss；launcher 吞掉子进程错误后继续等待或表现假成功。

**公司改进**：

- 数据层提供 skip/invalid mask；
- trainer 对 NaN loss 做持续计数，避免无限浪费计算；
- launcher 显式保留子进程退出状态；
- 训练框架对部分不支持的字段或接口尽早给明确报错。

**为什么有效**：

- 大规模训练中最怕“进程仍在跑但不训练”或静默漏掉异常。
- 快速失败可保留 GPU 预算，而不是等待超时。
- 数据级跳过可在坏点比例低时继续训练。

**代价/风险**：

- 过度快速失败可能被偶发 I/O 噪声放大。
- 跳过样本会导致统计分布变化，需要查看 skipped 计数。
- launcher 修正后，必须保证 wrapper 不吞 exit code。

**代码**：

- `giga-code/giga-train/giga_train/trainers/trainer.py`
- `giga-code/giga-train/giga_train/distributed/launch.py`
- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_transforms.py`

**状态**：`已实现`。

---

## T-029 训练、推理前缀契约校验

**基础做法**：训练配置只写模型超参，部署从另一个配置文件重新写一遍。错一个 action horizon、max delay 或 mask 仍可能“能启动”。

**公司改进**：

- 服务端加载 checkpoint 时校验 `enable_ttrtc`、`ttrtc_max_delay`、`n_action_steps`、`max_action_dim`、state mode、VLM type；
- server 启动前 sidecar 检查训练相机；
- client 与 server 校验 metadata 和窗口长度。

**为什么可能有效**：

- 模型可能可以在错误 horizon 上推理，但结果会静默变坏。
- 前缀、动作宽度、state mode 和相机顺序属于模型输入契约，不应依赖人工记忆。
- 可以在真机执行前失败。

**代价/风险**：

- 校验过严可能阻断兼容旧检查点的合法实验；需要明确兼容规则。
- sidecar/config/checkpoint 三方不一致时排查路径要清楚。
- 校验只能防止契约错配，不能证明模型质量。

**代码**：

- `zhangzhong_wbc/server/src/model.py`
- `zhangzhong_wbc/server/src/server.py`
- `zhangzhong_wbc/server/script/start_server_wbc.sh`

**状态**：`启用`。

---

## T-030 辅助目标与主体 policy 的隔离

**基础做法**：所有 embedding、所有 token 都共享同一目标，动作 expert 可以看到或间接依赖辅助预测。

**公司改进**：

- action suffix 不 attend 自回归 target；
- subtask 目标不反过来决定 flow action 的输入；
- action expert 的 VLM 梯度权重独立配置；
- FAST 词表扩展只作用于对应 VLM/checkpoint 合约。

**为什么可能有效**：

- 防止推理时不可用的 token target 泄漏到动作表示。
- 让辅助监督改善共享视觉语义，但不把最终动作路径绑定到自回归延迟。
- 独立开关和权重使各任务可以选择不同组合。

**代价/风险**：

- 多目标耦合难以从总 loss 判断优劣，需要分开监控。
- 目标隔离本身可能限制 VLM 共享，需实测。
- 训练后若改用 FAST 推理，必须重新验证自回归生成路径，而不能只用 flow rollout 结论。

**代码**：

- `giga-code/giga-models/giga_models/models/vla/giga_brain_0/modeling_giga_brain_0.py`
- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_loss.py`

**状态**：`已实现`。
