# 数据方法与技巧

本文只记录在“逐个文件、逐帧解码、普通随机打乱、固定目录拼接”这一基础做法之上的改进。

## 索引

| ID | 方法 | 主要收益 | 状态 |
|---|---|---|---|
| D-001 | 延迟打开数据文件 | 启动轻、worker 故障隔离 | 启用 |
| D-002 | worker 级有限句柄 LRU | 控制内存和文件句柄 | 已实现 |
| D-003 | 混合大小相机或文件的桶采样 | 减少 padding、提高有效吞吐 | 已实现 |
| D-004 | 按数据集大小的幂律重采样 | 抑制超大数据集垄断 | 已实现 |
| D-005 | 显式采样权重与分组均衡 | 控制任务、机器人、来源比例 | 已实现 |
| D-006 | 数据集边界内采样 | 避免 batch 被单一来源独占 | 已实现 |
| D-007 | 无限采样并确定性切 epoch | 流式训练稳定、可复现 | 启用 |
| D-008 | 场景图与多数据流对齐 | 一个样本组合图像、本体、标签 | 已实现 |
| D-009 | 相机键标准化 repack | 不同数据集统一到模型契约 | 已实现 |
| D-010 | resize-with-pad | 避免相机宽高比变化 | 启用 |
| D-011 | 轻量图像增强 | 提高视觉泛化 | 已实现 |
| D-012 | 时间偏移构造 action chunk | 从单帧样本生成动作序列 | 启用 |
| D-013 | delta action 与维度 mask | 统一绝对/增量控制并屏蔽无效维度 | 启用 |
| D-014 | 分位数归一化 | 降低离群值支配 | 启用 |
| D-015 | 无效动作块过滤 | 阻止坏数据污染监督 | 已实现 |
| D-016 | FAST 动作 token 目标 | 提供离散动作辅助监督 | 已实现 |
| D-017 | 时间戳对齐的多帧观测 | 真实历史而非“推理次数历史” | 已实现 |
| D-018 | episode 起始重置历史缓存 | 防止跨 episode 污染 | 已实现 |
| D-019 | 虚拟多根、物化分片与结果等价分片 | 大数据量下降低随机 I/O 和打开成本 | 实验/已实现 |
| D-020 | 无效图像样本跳过 loss | 避免损坏帧产生伪监督 | 已实现 |

---

## D-001 延迟打开数据文件

**基础做法**：构造 `Dataset` 时立即打开 parquet、视频解码器、元数据和文件句柄。

**公司改进**：`BaseDataset` 把构造和 `open()` 分开。对象只保存路径，第一次访问时才进行重量级初始化。各 worker 独立打开自己的资源。

**为什么有效**：

- 主进程构造数据集时不复制大量文件句柄。
- 多 worker 启动后各自打开，避免资源属于错误进程。
- 某个 worker 的文件状态不会被其他 worker 共享破坏。

**代价/风险**：

- 首次 `__getitem__` 更慢。
- 必须正确实现 `open()` 的并发保护和 `close()`，否则容易泄漏或重复打开。

**适用场景**：LeRobot、视频、LMDB、远程文件等重量级数据源。

**代码**：

- `giga-code/giga-datasets/giga_datasets/datasets/base_dataset.py`
- `giga-code/giga-datasets/giga_datasets/datasets/lerobot_dataset.py`

**状态**：`启用`。LeRobot 的 `self.dataset` 初始为 `None`，在 `open()` 时创建。

---

## D-002 worker 级有限句柄 LRU

**基础做法**：缓存所有打开过的子数据集；或者每次访问都关闭再重开。

**公司改进**：每个 worker 只保留最多 `max_open_per_worker` 个打开的 LeRobot 子数据集，采用 LRU 淘汰，定期 `gc.collect()`。被淘汰时只释放重的内部 `FastLeRobotDataset`，不破坏上层路径配置。

**为什么可能有效**：

- 跨机器人、跨来源的 `ConcatDataset` 可能引用几十到上百个根目录。
- 每个 LeRobot 根都可能持有 Arrow 缓存、视频解码器和元数据。
- 全缓存会推高 RSS 和句柄数；全不缓存会反复打开，拖慢随机读取。

**代价/风险**：

- LRU 容量太小会抖动，频繁重开。
- 视频解码器释放可能带来短时延迟尖峰。
- 当前实现依赖 `child.dataset = None` 的 LeRobot 内部状态，升级数据层时要回归验证。

**适用场景**：多数据集混合、跨任务联合训练、单机大内存但 worker 很多的训练。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/lerobot_open_cache.py`
- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_trainer.py`

**状态**：`已实现`。是否启用取决于 `dataloaders.train.cache.lerobot_open_cache`。

---

## D-003 混合大小数据桶采样

**基础做法**：全数据随机打乱后统一 pad 到最大尺寸。

**公司改进**：根据文件图像宽高比或预计算 `bucket_index`，在每个桶内打乱，再对桶整体打乱；支持不同桶使用不同 batch 大小。

**为什么可能有效**：

- 混合分辨率视觉数据如果随机组 batch，每个 batch 都被最大分辨率支配。
- 将近似尺寸样本放在一起可减少 padding 像素和视觉编码计算。
- 桶内保留随机性，全局再随机切换桶，降低固定长度顺序带来的偏差。

**代价/风险**：

- 建立桶索引要扫描样本属性。
- 若采样顺序在时间或任务上高度相关，需要另行混入来源均衡。
- 动态分辨率并不是免费得到；训练步数对应的样本覆盖可能变化。

**代码**：

- `giga-code/giga-datasets/giga_datasets/samplers/aspect_ratio_sampler.py`
- `giga-code/giga-datasets/giga_datasets/samplers/bucket_sampler.py`
- `giga-code/giga-datasets/giga_datasets/samplers/bucket_batch_sampler.py`

**状态**：`已实现`，当前基础任务不必启用。

---

## D-004 按数据集大小的幂律重采样

**基础做法**：`ConcatDataset` 按样本数自然采样，越大的数据集被看到的频率越高。

**公司改进**：对每个子数据集，以 `size ** exponent` 计算权重，在总样本量近似保持不变的前提下重新分配采样数。默认指数约 `0.43`。

**为什么可能有效**：

- 原始比例下，少数大数据集会主导梯度。
- 简单等权又会过度重复小数据集。
- 幂律折中：大数据仍更有代表性，但优势被压缩。

**代价/风险**：

- 指数是超参数，会改变任务和数据来源的决策逻辑。
- 小数据集重复过多时可能过拟合。
- 不应与“显式业务权重”无审计地叠加。

**代码**：

- `giga-code/giga-datasets/giga_datasets/samplers/weighted_sampler.py`

**状态**：`已实现`。在配置主路径中可被 `WeightedSampler` 选择。

---

## D-005 显式采样权重与分组均衡

**基础做法**：所有来源按样本总量拼接。

**公司改进**：

- 对机器人、UMI、Ego、VLM 等来源显式指定采样权重。
- `WeightedConcatDataset` 保留子数据集和权重元数据。
- 按组、机器人类型和数据集边界控制采样规模，避免一个大目录吞掉其他目录。
- 支持设置 `weight=0` 排除单个数据源。

**为什么有效**：

- 训练目标不只是“总帧数”，任务类别、本体和场景分布同样重要。
- 对稀少但有价值的机器人任务给较高权重，比拷贝重复数据更容易审计。
- 更容易保证每个 batch 或每个 epoch 包含所需类别。

**代价/风险**：

- 权重调整会直接改变有效训练集，需配合 norm 和验证集。
- 多级权重容易叠加错误：外层任务权重、内层来源权重和 sampler 权重必须分清。
- 处理不当会让采样分布与部署分布偏差过大。

**代码**：

- 配置示例：`giga-code/giga-models/projects/vla/giga-brain-0/configs/giga_brain_1_2wh.py`
- 当前 GB0.8 辅助：`.deployment/gb08_3drinks_h01_posttrain.py`
- 数据集定义：`giga-code/giga-datasets/giga_datasets/datasets/dataset.py`

**状态**：`已实现`。GB0.8 三饮品配置使用分组采样。

---

## D-006 保留子数据集边界的采样

**基础做法**：对大拼接集做完全全局随机打乱。

**公司改进**：`SpecialDatasetSampler` 先在每个子数据集内部构造并打乱，再按 batch block 组织，最后全局混合。

**为什么有效**：

- 可以控制每个 batch 中不同来源的出现结构。
- 避免局部打乱失效时出现长时间连续访问同一数据源。
- 便于配合有限句柄缓存，减少来回切换根目录。

**代价/风险**：

- 边界结构和全局随机性会互相影响。
- 如果学习率、batch norm 或采样比例敏感，需要做 A/B。

**代码**：

- `giga-code/giga-datasets/giga_datasets/samplers/special_sampler.py`

**状态**：`已实现`。

---

## D-007 无限采样与确定性 epoch

**基础做法**：每次 DataLoader 重启都从头产生随机顺序。

**公司改进**：sampler 可无限迭代；`set_epoch()` 使用 `seed + epoch` 确定性重置随机数生成器；训练以 step 为主，而不是强依赖有限 epoch 结束。

**为什么有效**：

- 长训练任务不会因为一个 epoch 结束而停 DataLoader。
- 断点恢复时，epoch 和步数可以更紧密对齐。
- 相同种子下采样顺序可重建。

**代价/风险**：

- 全局 `np.random.seed()` 会影响同进程其他随机行为，扩展时要谨慎。
- 断点恢复只保证相同配置下可重建，不等价于完整状态快照。
- 无限 sampler 一旦配置错，循环逻辑可能在耗尽数据后仍然重复，需要专门测试。

**代码**：

- `giga-code/giga-datasets/giga_datasets/samplers/default_sampler.py`
- `giga-code/giga-datasets/giga_datasets/samplers/special_sampler.py`

**状态**：`启用`。配置以 `max_steps` 驱动训练。

---

## D-008 场景图与多数据流对齐

**基础做法**：单文件里存一条联合样本。

**公司改进**：`Dataset` 按 `data_index` 对齐多个子数据集，按样本索引组合字段；`sort_paths()` 保证带对齐键的数据优先，支持图像、标签、状态等分拆存储。

**为什么有效**：

- 同一观测的不同模态可独立读写、压缩和扩容。
- 新增字段时无需重写全部图像或视频。
- 对多来源采集工具更友好。

**代价/风险**：

- 对齐键必须唯一、类型稳定。
- 字段缺失和索引偏移是高风险问题，必须校验。
- 来自不同采样率和时间戳的流，不能仅靠线性 index 就假设同步。

**代码**：

- `giga-code/giga-datasets/giga_datasets/datasets/dataset.py`
- `giga-code/giga-datasets/giga_datasets/datasets/base_dataset.py`

**状态**：`已实现`。

---

## D-009 相机键标准化 repack

**基础做法**：不同数据集的相机名原样喂给模型。

**公司改进**：在读取时把物理相机键重命名为模型统一键，例如：

```text
observation.images.cam_left_wrist_up
→ observation.images.cam_left_wrist

observation.images.cam_right_wrist_up
→ observation.images.cam_right_wrist
```

**为什么有效**：

- 不同机器人可共享一个 VLM 视觉接口。
- 训练、部署和侧车配置可以围绕统一键做校验。
- 相机键顺序显式声明，避免 dict 顺序或采集命名漂移。

**代价/风险**：

- 映射错误会把错误视角喂给模型。
- 物理相机名称和训练语义必须一一核对。
- 某些数据集多一个下视相机，不能按名称碰撞。

**代码**：

- `giga-code/giga-datasets/giga_datasets/datasets/lerobot_dataset.py`
- GB0.8 配置示例：`.deployment/gb08_3drinks_h01_posttrain.py`

**状态**：`已实现`，GB0.8 三饮品使用。

---

## D-010 resize-with-pad

**基础做法**：直接 resize 到 `224×224`。

**公司改进**：先按同比例缩放，再用固定填充值 pad 到目标分辨率，同时返回原图尺寸、缩放比例、padding 偏移等元信息。

**为什么有效**：

- 不改变物体几何比例，尤其利于小物体和接触操作。
- 训练和推理端使用相同几何变换，减少域差异。
- 返回变换参数，可支持像素位置到原图坐标的逆映射。

**代价/风险**：

- 有效像素区域变小。
- padding 值、训练和推理是否一致必须核对。
- 宽高比相差很大的数据可能产生大量无效像素。

**代码**：

- `giga-code/giga-models/giga_models/pipelines/vla/giga_brain_0/giga_brain_0_utils.py`

**状态**：`启用`，默认三相机视觉路径使用。

---

## D-011 轻量图像增强

**基础做法**：只用原始帧。

**公司改进**：

- 所有相机支持颜色扰动：brightness、contrast、saturation。
- 运行时锚点相机支持轻微随机 crop 和 ±5° rotation。
- 训练时才开启，验证路径使用非增强配置。
- 支持可选前景/背景增强和历史帧 dropout 的代码路径。

**为什么可能有效**：

- 对光照、相机安装偏移和轻微姿态变化更稳。
- 只增强主视图，避免破坏腕相机接触区域。
- 少量几何扰动比大范围失真更贴近真机部署误差。

**代价/风险**：

- 增强强度过大时会破坏精确接触和夹取几何。
- 不同相机使用不同增强策略，若训练和推理图变换不一致会出现域差。
- 前景/背景增强依赖额外 mask，不适用于没有 mask 的数据。

**代码**：

- `giga-code/giga-models/giga_models/pipelines/vla/giga_brain_0/giga_brain_0_utils.py`
- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_transforms.py`

**状态**：图像基本增强 `已启用`；前景背景增强等高级路径 `已实现`，不是所有任务启用。

---

## D-012 时间偏移构造 action chunk

**基础做法**：一个样本只给一个动作。

**公司改进**：在 LeRobot 元数据中声明 `delta_info={'action': 50}`，从当前帧读取未来 50 个时间戳，自动形成动作块；对 `model_prediction` 等辅助字段同步应用相同窗口。

**为什么有效**：

- 把从“单步模仿”变成“短时动作轨迹建模”。
- 动作块长度与模型的输出 horizon 一致。
- 辅助预测字段能和 action 对齐，避免形状和时间语义错位。

**代价/风险**：

- episode 尾部必须正确产生 padding mask。
- 低帧率、不同 fps 和丢帧会让窗口含义发生变化。
- 时间戳和采样率不匹配时，不能只看数组长度。

**代码**：

- `giga-code/giga-datasets/giga_datasets/datasets/lerobot_dataset.py`

**状态**：`启用`，主线任务常见 50 步 action horizon。

---

## D-013 delta action 与维度 mask

**基础做法**：所有动作统一绝对位置，或统一全部转成 delta。

**公司改进**：

- 对每个机器人或 embodiment 提供不同的 delta mask。
- 关节位置可走 delta，夹爪等维度保留绝对语义。
- 无效或不受模型监督的动作维度不进入噪声、diffusion loss 或 FAST 目标。
- 物理动作宽度补齐到 32 维时，padding 维度不污染监督。

**为什么有效**：

- 对绝对关节位置和夹爪命令使用不同数值语义更自然。
- 数据实际尺寸不同，可共享固定 32 维模型张量。
- 屏蔽无效维度可避免零值 padding 被当作真实目标。

**代价/风险**：

- mask 顺序必须与数据集 action layout 完全一致。
- 训练和部署的 delta 解释必须一致，否则会出现动作反解错误。
- 夹爪范围变化和 offset 需要单独验证。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_transforms.py`
- `giga-code/giga-models/giga_models/pipelines/vla/giga_brain_0/giga_brain_0_utils.py`
- H01 配置：`zhangzhong_wbc/server/configs/microwave_wbc.py`

**状态**：`启用`。

---

## D-014 分位数归一化

**基础做法**：`(x - mean) / std`。

**公司改进**：默认支持 `q01` 和 `q99`，将区间映射到 `[-1, 1]`。对动作目标默认不做 clamp，对状态输入可选 clamp。

**为什么有效**：

- 机器人动作里常有短时异常尖峰，mean/std 容易被离群值拉偏。
- 1%/99% 分位区间更能稳定覆盖正常动作范围。
- 动作作为监督目标时不 clamp，可避免直接修改真值；状态输入可 clamp 限制推理时的极端输入。

**代价/风险**：

- 超出 q01/q99 的正常轨迹会被压缩范围或产生分布外值。
- 不同机器人必须使用对应统计文件。
- 训练和推理必须使用同一 norm 文件及同样 key 选择逻辑。

**代码**：

- `giga-code/giga-models/giga_models/pipelines/vla/giga_brain_0/giga_brain_0_utils.py`
- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_transforms.py`

**状态**：`启用`。主线 config 普遍设置 `use_quantiles=True`。

---

## D-015 无效动作块过滤

**基础做法**：所有训练样本直接计算 loss。

**公司改进**：在 action 数据上检查有限值、归一化后幅度、连续帧稳定性等条件；一旦发现异常，可屏蔽该 chunk 的 action 维监督和时间维监督。

**为什么有效**：

- 数据采集、单位换算或局部断帧可能产生极端动作。
- 单个巨大动作会拉高 loss 和梯度，导致后续训练不稳定。
- 在样本层禁监督，比整个 batch 丢弃更保留图像语言信号。

**代价/风险**：

- 阈值太严会误伤快速但有效的动作。
- 只屏蔽 action 监督时，要确认 FAST token 或子任务目标是否也应同步屏蔽。
- 过滤逻辑要能定位源数据集和 frame，不能静默吞坏数据。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_transforms.py`
- 入口参数：`invalid_action_chunk_filter_cfg`

**状态**：`已实现`，需检查具体任务配置是否开启。

---

## D-016 FAST 动作 token 目标

**基础做法**：只用连续动作回归或 flow matching。

**公司改进**：动作块通过 FAST tokenizer 压成离散 token；在语言序列中保留指示 mask，让 VLM 学习 token 级动作序列。连续 flow 动作和 FAST target 共享 VLM 表征，但 attention 中将两侧预测目标隔开。

**为什么可能有效**：

- 离散序列监督强调长期结构和离散模式，与连续 flow loss 互补。
- VLM 的 token 预测能力可以从文本扩展到动作领域。
- teacher-forcing 训练稳定，不要求训练时依赖不成熟的自回归采样。

**代价/风险**：

- FAST token loss 很容易在总 loss 中占主导，需要正确权重。
- tokenizer 词表和 token id 映射必须与 checkpoint 完全一致。
- 若部署只使用连续 flow，FAST 分支属于辅助监督，需通过实验确认是否真正提升策略，而不是只改善 token 指标。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_loss.py`
- `giga-code/giga-models/giga_models/models/vla/giga_brain_0/modeling_giga_brain_0.py`

**状态**：`已实现`。GB0.8 三饮品启用，GB0.7 当前三饮品关闭。

---

## D-017 时间戳对齐的多帧观测

**基础做法**：每推理一次就把新观测当一个历史帧，忽略推理周期变化。

**公司改进**：按相机 ROS timestamps 保存环形缓存；请求时根据 `observation_memory_offsets` 取固定时间间隔的帧，例如 `[-60,-30,0]`；对帧时间差设置最大 gap，缺帧时返回 padding mask。

**为什么有效**：

- 推理耗时波动不会改变“历史帧”的物理时间距离。
- 训练时的帧偏移和部署时的实际帧偏移保持一致。
- 可以用 mask 表达少量缺帧，而不是拿错位帧冒充历史。

**代价/风险**：

- 相机时钟不同步会直接破坏历史语义。
- 高帧率缓存占用更高。
- episode 切换、服务重连和任务切换时必须清理历史。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/scripts/inference/h01_client/nvidia_client_smooth_temporal.py`
- `giga-code/giga-models/projects/vla/giga-brain-0/scripts/inference/inference_agilex_client_unified_smooth_temporal.py`

**状态**：`已实现`。

---

## D-018 episode 起始重置历史缓存

**基础做法**：服务常驻，历史缓存跨任务或跨 episode 保留。

**公司改进**：检测 episode id 变化后清空本地时间历史，并在下一次请求中发送 `reset_observation_memory=True` 给 server。

**为什么有效**：

- 防止上一个任务末帧被误认为新任务历史。
- 时间超时、任务切换和重连后能恢复干净的上下文。
- 本地和 server 两端一起清，避免只清一端仍保留过期缓存。

**代价/风险**：

- 首个请求的上下文为空或全 padding，可能暂时降低预测质量。
- 应确认 server 支持 reset 字段且幂等。
- 不要按客户端进程启动次数推断 episode 边界，应该用明确任务或 episode id。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/scripts/inference/h01_client/nvidia_client_smooth_temporal.py`
- `giga-code/giga-models/projects/vla/giga-brain-0/scripts/inference/inference_agilex_client_unified_smooth_temporal.py`

**状态**：`已实现`。

---

## D-019 虚拟多根、物化分片与结果等价分片

**基础做法**：一个 `LeRobotDataset` 对应一个物理目录，随机读时跨目录反复打开。

**公司改进**：

- `VirtualMultiRootLeRobotDataset` 用轻量索引表现多个根目录。
- 将多个小根整理成物理分片，减少随机 I/O。
- 分片读取保持样本语义不变，并提供结果等价性约束。
- 支持 Arrow cache warmup，避免首个训练 step 同时重打开所有大文件。

**为什么可能有效**：

- 大规模多机器人训练常遇到“很多小数据集、每个几十 MB”的目录结构。
- 随机批次频繁切换目录时，打开元数据、Arrow 和视频的成本可能远高于模型读取。
- 分片可以让一组小数据共享打开成本。

**代价/风险**：

- 分片索引或映射错误会产生不易发现的样本错位。
- 物化过程复制、占用额外存储。
- 分片后 chunk 随机性、episode 边界和样本顺序必须由测试守住。
- 不能把“结果等价”只当成文件内容相同，还应验证 sample index 和 transform 前语义。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/lightweight_shard_index.py`
- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_trainer.py`

**状态**：`已实现/实验`。常见小任务不必开，GB0.8 大数据和虚拟多根路径更相关。

---

## D-020 无效图像样本跳过 loss

**基础做法**：所有读到的图像直接计算视觉、语言和动作 loss。

**公司改进**：检测全零、异常解码或显式标记的坏图像，并在样本里提供 `_skip_loss_for_invalid_images`；转换层将其变为 `skip_loss_mask`，训练层跳过对应样本的损失贡献。

**为什么可能有效**：

- 部分发布器或磁盘读错误会产生全零帧。
- 全零图像会让视觉编码器输出错误语义，模型可能学成“看见黑屏就产生某动作”。
- 只跳过 loss 比停训练更容忍少量坏帧。

**代价/风险**：

- 全零判断可能误伤真正的黑背景场景。
- 如果坏帧和真实动作有关联，简单跳过会产生选择偏差。
- 应跟踪 skip 计数和来源，坏数据比例突增时告警。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_transforms.py`
- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_trainer.py`

**状态**：`已实现`。
