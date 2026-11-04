# 覆盖范围、证据与待审清单

更新时间：2026-11-04。

这份文件用于防止“笔记写了多少”与“代码审了多少”混淆。第一版覆盖主动画链和关键控制路径，但不宣称已经穷尽整个仓库。

## 1. 当前覆盖

### 数据

已审：

- `giga-datasets/giga_datasets/datasets/base_dataset.py`
- `dataset.py`
- `lerobot_dataset.py`
- `default_collator.py`
- `default_sampler.py`
- `special_sampler.py`
- `weighted_sampler.py`
- `aspect_ratio_sampler.py`
- `bucket_sampler.py`
- `bucket_batch_sampler.py`
- `giga_brain_0_transforms.py` 中的归一化、action mask、action delta、无效数据和图像输入路径
- `lerobot_open_cache.py`
- `lightweight_shard_index.py` 的接口和主要分片路径

第一版已记录 20 条数据方法。SFT 与 RGB 相关的高阶采样、VLM/机器人混合策略和完整物化分片实现尚未逐行展开。

### 训练

已审：

- `giga-train/giga_train/trainers/trainer.py`
- `strategies/activation_checkpointing.py`
- `strategies/ema.py`
- `optimizers/*`
- `schedulers/*`
- `samplers/parallel_batch_sampler.py`
- `distributed/launch.py`
- `giga-models` 的 `giga_brain_0_loss.py`
- `modeling_giga_brain_0.py` 的 prefix/suffix、action projection、multi-flow、language loss 和 TTRTC
- `giga_brain_0_trainer.py` 的数据加载、checkpoint、EMA、completion、loss 和 profile 入口
- `fsdp2_sharded_ema.py`

第一版已记录 30 条训练方法。仍需展开：

- Qwen3.5/robomanip expert 所有分支
- Gemma3/Gemma4/其他 VLM 后端
- 全部 FSDP/DeepSpeed/FP8 边界情况
- sharded EMA 的详细一致性证明
- 各种 modality 与机器人本体专属训练配置

### 部署

已审：

- `zhangzhong_wbc/server/configs/base.py`
- `server/src/model.py`
- `model_pipeline.py`
- `server.py`
- `start_server_wbc.sh`
- `manage_server.sh`
- `run_client.sh`
- 任务 launcher 与部署文档
- resident server 与 MCP 分层文档中的实际调用链

第一版已记录 20 条部署方法。仍需展开：

- AgentOS/MCP 的完整执行状态机
- 多 client、多 server 的租约和并发策略
- 全部服务管理与系统级故障恢复
- 配置备份和真实回滚自动化

### 推理

已审：

- WBC server 编译、warmup、prefix 和模型推理
- TTRTC 训练/推理 helper
- smooth/temporal H01 和 AgileX client
- async inference、adaptive horizon、alignment、low-pass、continuity blend
- resident review 得出的 completion、teleop/pause 和安全检查机制
- client 风险点与协议约束

第一版已记录 21 条推理方法。仍需展开：

- 完整 ROS 控制路径与所有机器人发布 topic
- 全部动作重采样和夹爪校准
- Pico/teleop 介入的真实控制所有权
- 真实模型在不同任务上的延迟、overrun 和失败模式统计数据

## 2. 当前状态

### 已验证的证据

| 范围 | 证据 |
|---|---|
| GB0.7 三饮品 loss 与构成 | `gb08-code-compare-20261103/reports/GB08_VS_GB07_3DRINKS_LOSS_20261103.md` |
| GB0.8 与 GB0.7 数据/训练代码差异 | `gb08-code-compare-20261103/reports/GB08_TRAIN_DATASETS_COMPARE_20261103.md` |
| 训练调用链 | `TRAINING_CODE_MAP.md` |
| 奶茶部署调用链 | `MILKTEA_DEPLOYMENT_FLOW.md` |
| 部署资产校验 | `MILKTEA_HEAD_DEPLOYMENT_20261028.md` |

### 未做完整验证的方法

以下已有实现或单次运行证据，但尚不应当当作已完全验证：

- `multi-flow` 在目标任务上的收益和代价。
- FAST token 辅助 loss 对最终连续动作和成功率的影响。
- DCT 正则及动作维 loss 权重是否优于基础 MSE。
- 物化分片与虚拟多根的 I/O 和正确性收益。
- bucket sampler 在混合分辨率数据上的实际 GPU 吞吐收益。
- low-pass、adaptive horizon、continuity blend 的独立消融。
- torch.compile、FP8、attention 后端在不同硬件上的数值一致性。
- 多帧历史在真机时钟偏差和网络丢帧下的稳定性。

## 3. 下一批审查

优先级从高到低：

1. **训练目标与消融**
   - DCT loss
   - action dimension loss weights
   - multi-flow sample 数量
   - FAST token loss 与 flow loss 的独立消融
   - TTRTC delay 分布与执行频率的对应关系

2. **数据加载与采样**
   - `WeightedConcatDataset` 和 `ListWeightedSampler` 的完整实现
   - 多机器人分组和组内/组间权重
   - LeRobot 旧版与 v3.0 在 repack、video、advantage 上的差异
   - virtual multi-root 的真实样本语义和误差边界

3. **模型架构**
   - Qwen3.5 / Gemma3 / Gemma4 / robomanip expert
   - 时间注意力和 history pool
   - language loss chunking 对吞吐和显存的影响
   - attention backend 数值稳定性和推理延迟

4. **部署和生产控制**
   - agent/MCP 的真实状态机
   - 服务启动、握手、模型 lease 与 resident prepare 的超时边界
   - 每次真实失败的根因、复现和修复状态
   - GPU server 的多任务分片与资源预算

5. **推理控制**
   - 动作块自适应长度和触发阈值的稳定性
   - 平滑器和连续性偏置的相位滞后
   - teleop 接管触发与重新武装的完整状态
   - completion 阈值与任务真正完成的偏差
   - 不同发布频率下的延迟预算

## 4. 后续记录要求

新增或修订条目时，按以下格式追加：

```markdown
## X-NNN 方法名

**基础做法**：...
**公司改进**：...
**为什么可能有效**：...
**代价/风险**：...
**适用场景**：...
**代码**：...
**状态**：`启用|已实现|实验|已验证`
**证据**：日期、提交、日志、实验或现场结果。
```

若一个方法只存在于实验脚本，必须标记 `实验`。若一个方法已被主配置引用但没有独立消融，应标记 `已实现` 或 `启用`，不要写“已验证”。

## 5. 状态修订记录

| 日期 | 变更 |
|---|---|
| 2026-11-04 | 创建首版索引和四份分册；记录 20 条数据、30 条训练、20 条部署、21 条推理方法 |
