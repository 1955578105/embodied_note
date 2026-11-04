# 推理与控制方法与技巧

本文记录在“收到一帧后完整前向一次，取一组动作，逐帧直接发布 ROS”这一基础做法之上的改进。

## 索引

| ID | 方法 | 主要收益 | 状态 |
|---|---|---|---|
| I-001 | 只编译 denoise step | 把优化集中在重复最频繁的路径 | 已实现 |
| I-002 | Tensor 预热与固定 shape | 消除首任务编译抖动 | 启用 |
| I-003 | FlashAttention 或 SDPA 路径优化 | 减少 attention 显存和时间 | 已实现 |
| I-004 | TTRTC hard prefix | 让重规划接上已执行轨迹 | 启用 |
| I-005 | best-effort / strict 调度 | 延迟下选择连续执行或拒绝过期结果 | 已实现 |
| I-006 | 触发剩余动作阈值 | 提前发起推理 | 已实现 |
| I-007 | 自适应执行 horizon | 轨迹发散时缩短动作块 | 已实现 |
| I-008 | 关节位姿就近对齐 chunk | 消除线程时延导致的重放/跳步 | 已实现 |
| I-009 | Butterworth 低通平滑 | 抑制关节高频抖动 | 已实现 |
| I-010 | 拼接连续性偏置 | 消除 low-pass 后残余阶跃 | 已实现 |
| I-011 | 夹爪与底盘特殊处理 | 不只按全部关节统一滤波 | 已实现 |
| I-012 | 异步推理与主线程定时发布 | 推理等待和控制发布解耦 | 已实现 |
| I-013 | 动作缓冲区与不重复消费 | 每个动作只执行一次 | 启用 |
| I-014 | 动作命令行限与夹爪边界 | 拦截明显异常的大跳 | 启用 |
| I-015 | VR teleop 接管暂停 | 人工操控优先，避免模型抢命令 | 启用 |
| I-016 | 推理断线后安全停止 | server 断开不继续发布旧动作 | 启用 |
| I-017 | 时间戳历史与 episode reset | 防止历史错位 | 已实现 |
| I-018 | perception delay 与 server prefix | 让条件与真实执行进度一致 | 已实现 |
| I-019 | completion/progress 连续确认 | 减少误完成任务 | 已实现 |
| I-020 | 只发送必要相机与压缩图像 | 降低 client 编码与网络延迟 | 启用 |
| I-021 | 客户端侧不盲目调大步长门限 | 防真实模型大跳直接下发 | 启用 |

---

## I-001 只编译 denoise step

**基础做法**：完全不编译，或把整个 pipeline 一次性编译。

**公司改进**：WBC server 默认编译 `policy.denoise_step`，保留图像解码、prompt tokenize 和输出后处理为普通 Python；可选 `full_compile` 编译全 pipeline。

**为什么可能有效**：

- flow sampling 在固定 shape 下重复执行多步，最值得做 kernel 融合。
- 图像预处理、JPEG 编码和 I/O 会触发大量 Python 控制流，整图编译收益低且容易重编译。
- 只编译热点能缩短首次启动。

**代价/风险**：

- 预处理和网络 I/O 仍占明显延迟，不能期待 10 倍加速。
- 改 TTRTC 分支或 action horizon 时可能触发重编译。
- compile 失败时 error 发生在服务启动，而不是第一次用户任务。

**代码**：

- `zhangzhong_wbc/server/src/model.py`
- `zhangzhong_wbc/server/configs/base.py`

**状态**：WBC 配置 `backend=inductor`、`full_compile=False`。

---

## I-002 Tensor 预热与固定 shape

**基础做法**：服务启动后等第一个真实请求才初始化 CUDA kernel 和编译。

**公司改进**：在 READY 前完成多 delay 的 dummy inference，并使用固定 image shape、state/action width 和 flow horizon；服务打印每个 delay 的耗时。

**为什么可能有效**：

- 固定 shape 避免动态 shape fallback。
- 首轮 CUDA allocator、cuDNN/cuBLAS 和 Inductor 的初始化被移出真实控制路径。
- 通过 warmup 可以提前识别 TTRTC 分支是否未准备。

**代价/风险**：

- 固定 shape 对实际输入变化不宽容，正式摄像头分辨率错误会失败或重编译。
- warmup 必须使用和实机相同的 image mode 和相机数量。
- 预热完成不代表传感器在线，仍需要 client 的 sensor readiness。

**代码**：

- `zhangzhong_wbc/server/src/server.py`
- `zhangzhong_wbc/server/configs/base.py`

**状态**：`启用`。

---

## I-003 FlashAttention 或 SDPA 路径优化

**基础做法**：所有 attention 都使用完整 attention matrix。

**公司改进**：

- 对部分 PaliGemma 路径提供 `enable_flash_attention_optimization`，裁掉右侧 prompt padding，让 eager/Inductor prefill 使用更密集的 KV；
- 在专家 attention 中按路径选择 SDPA Math、Flash 或 fused 后端；
- 对 prefix query rows 只构造需要的 mask 行，避免生成完整二维矩阵。

**为什么可能有效**：

- 视觉和文本 prefix 是序列长度主要来源，padding 会直接占用 attention 计算。
- 用高效 attention 后端可减少 HBM 和中间张量。
- query row 局部 mask 支持把 action expert 的 suffix 计算压到短路径。

**代价/风险**：

- 不同 attention 后端数值有微小差异；长前缀上可能放大。
- 使用错误 mask 会造成训练-推理不一致。
- 裁 padding 后 position ids、KV cache 和生成模式必须维持一致长度定义。

**代码**：

- `giga-code/giga-models/giga_models/models/vla/giga_brain_0/modeling_giga_brain_0.py`
- `giga-code/giga-models/giga_models/models/vla/giga_brain_0/paligemma2_with_expert.py`

**状态**：代码 `已实现`；当前 WBC 配置的 `enable_flash_attention_optimization=False`。

---

## I-004 TTRTC hard prefix

**基础做法**：每次推理直接生成新的 50 步动作；client 自行选择执行其中前若干步。

**公司改进**：在动作 chunk 的头部接受已经执行过或已承诺的 clean prefix；模型只生成 postfix，客户端校验模型输出前缀未改变；在喷发前把前缀映射回实际模型空间。

**为什么可能有效**：

- 将推理时已有动作作为强约束，减少 chunk 之间的不连续。
- 新推理不会重新改写已执行轨迹。
- 可以把模型延迟和队列状态连接成一个明确协议。

**代价/风险**：

- prefix 过长会减少新生成的有效动作。
- prefix 在设备端、client 和 server 之间需要一致的 normalization 与 delta 处理。
- 如果队里动作已过期，hard prefix 会把旧决策锁进新结果；必须有 strict/best-effort 策略。

**代码**：

- `giga-code/giga-models/giga_models/models/vla/giga_brain_0/ttrtc.py`
- `zhangzhong_wbc/server/src/model.py`

**状态**：`启用`，WBC server 的 `max_delay=10` 等配置生效。

---

## I-005 best-effort 与 strict 调度

**基础做法**：推理未返回时继续重复旧动作；或者超时直接停掉。

**公司改进**：

- `strict` 要求实际消费步数不超过 conditioned prefix，并在超出时停止；
- `best-effort` 在 prefix 不足时仍 splice，同时计数 `PREFIX_OVERRUN`；
- client 可以配置 queue、prefetch、execute horizon 和报告字段，明确哪些步骤是旧动作延续。

**为什么可能有效**：

- 真实网络/模型延迟并非固定，严格模式可确保不执行未 conditioned 段，但更容易卡顿。
- best-effort 保持连续性，适合交互 demo，但必须对超界次数告警。
- 两种模式都有显式可审计指标，而不是隐藏行为。

**代价/风险**：

- best-effort 实际上会让机器执行外推或旧动作，安全上不能当作等价 strict。
- 高频率、低 prefix 时间预算会把 overrun 当作常态，掩盖延迟问题。
- strict 需要 prefix 足够长；频率和推理时间要配套设计。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/scripts/inference/h01_client/nvidia_client_smooth.py`
- `zhangzhong_wbc/server/src/model.py`

**状态**：WBC 常用 best-effort；日志有显式 `PREFIX_OVERRUN`。

---

## I-006 触发剩余动作阈值

**基础做法**：等动作队列耗尽后再推理，机器人可能出现停顿。

**公司改进**：当当前 chunk 剩余动作数降到 `inference_trigger_remaining` 时，后台线程提前发起下一次推理；旧队列继续发布，直到新结果满足应用条件。

**为什么可能有效**：

- 将推理延迟隐藏在旧 chunk 仍在执行的期间。
- 通过剩余阈值控制启动时刻，避免推太早导致旧结果失效，推太晚则停顿。
- 可与自适应 horizon 结合，在长动作块尾部更频繁地重新规划。

**代价/风险**：

- 阈值必须大于推理的实际耗时对应的动作步数。
- 旧线程和新线程如果没有互斥，可能产生重复请求。
- 未经 conditioned 的旧动作数量必须监控。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/scripts/inference/h01_client/nvidia_client_smooth.py`

**状态**：`已实现`。

---

## I-007 自适应执行 horizon

**基础做法**：每个预测 chunk 都固定执行 35/40/50 步。

**公司改进**：沿候选 checkpoint 计算动作与首帧的 L2 距离；一旦超过阈值，就缩短 horizon 并提前重推理。gripper 等维在距离计算中单独处理。

**为什么有效**：

- 模型短时预测比长时预测更可靠。
- 一旦轨迹在预测中明显走向不同区域，继续执行末端错误动作不如尽早重规划。
- 动作块总长度固定，但实际执行的有效安全 horizon 随状态变化。

**代价/风险**：

- 距离阈值是任务相关的，不同机器人关节尺度和速度差异大。
- 过短 horizon 会让推理频率大幅提高，反而加剧网络压力。
- 该启发式不能替代动作限幅和最终校验。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/scripts/inference/inference_agilex_client_unified_smooth.py`

**状态**：`已实现`。

---

## I-008 关节位姿就近对齐 chunk

**基础做法**：根据推理耗时估计新 chunk 应执行的第 `N` 步，直接从固定索引 `N` 开始。

**公司改进**：以估计起点为中心开一个小窗口，在预测动作中找与最近已发布/执行关节位置最近的帧作为新 chunk 起点；按真实机器人状态对齐，不只按时间估计。

**为什么有效**：

- 线程调度、相机时间戳和网络延迟会让纯时间索引误差。
- 起点偏小会重放已走过轨迹，起点偏大则跳过轨迹，都会产生可见抖动。
- 关节距离比 action chunk 的绝对时钟更能反映当前物理状态。

**代价/风险**：

- 只对齐手臂关节，夹爪和底盘需要专门处理。
- 动作空间中不同关节尺度不同，距离度量应做归一化或加权。
- 状态反馈有延迟时，对齐目标本身会滞后。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/scripts/inference/inference_agilex_client_unified_smooth.py`

**状态**：`已实现`。

---

## I-009 低通平滑动作序列

**基础做法**：将模型动作直接逐帧发布。

**公司改进**：对左右臂各 7 个关节使用 Butterworth low-pass；夹爪和 14 维后字段不做同样平滑；在拼接窗口内先平滑，再丢弃 historical context。

**为什么可能有效**：

- VLA chunk 的逐帧预测会含高频噪声。
- 高频抖动会传到电机控制器，造成机械冲击和视觉抖动。
- 对夹爪单独处理可以避免开合动作被平滑得过慢。

**代价/风险**：

- low-pass 带来相位滞后，过强会削弱快速避障和接触反应。
- 滤波参数与发布频率绑定，20/30/40Hz 不能默认为同一组。
- 只靠低通不能修复 chunk 拼接造成的直流位置跳变。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/scripts/inference/inference_agilex_client_unified_smooth.py`

**状态**：`已实现`。

---

## I-010 拼接连续性偏置

**基础做法**：直接 low-pass 拼接旧动作与新 chunk。

**公司改进**：强制新段首帧对齐最近已执行动作，在 `continuity_blend_steps` 步内线性衰减 offset，即让新段起点的位置连续性优先，然后逐渐回归模型输出。

**为什么有效**：

- low-pass 对一个位置阶跃只会平滑过渡，仍可能残留位置误差。
- 线性 offset 更直接消除拼接处跳变。
- 衰减过程保持最终跟随模型新轨迹。

**代价/风险**：

- offset 过大意味着训练/推理分布差很大，不能长期只靠偏置掩盖问题。
- 夹爪与底盘不应统一加 offset。
- 衰减步数太少会再次形成大速度尖峰。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/scripts/inference/inference_agilex_client_unified_smooth.py`

**状态**：`已实现`。

---

## I-011 夹爪和底盘特殊处理

**基础做法**：所有动作维统一 post-process。

**公司改进**：

- smoothing 对夹爪列保留原值；
- 连续性偏置只作用于手臂，不作用于夹爪和底盘；
- 客户端可配置 gripper rescale、offset、dead zone；
- 动作维度由 layout 显式切片。

**为什么可能有效**：

- 夹爪是离散或接近离散的夹取状态，不应像连续关节一样平滑。
- 底盘或腰部有不同运动学与安全范围。
- 明确 layout 可防止把 head 与 waist 的 16:22 列发到错误 ROS topic。

**代价/风险**：

- rescale、offset 和 dead zone 是部署侧修改动作，必须与训练数据语义一致。
- 错一列就是控制不同关节，最危险的是维度互换。
- 关节名称的顺序和训练侧顺序要分别检查。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/scripts/inference/inference_agilex_client_unified_smooth.py`
- `giga-code/giga-models/projects/vla/giga-brain-0/scripts/inference/h01_client/nvidia_client.py`

**状态**：`已实现`。

---

## I-012 异步推理与主线程定时发布

**基础做法**：主控制循环里同步等待模型生成，推理时机器人停住。

**公司改进**：一个后台线程负责图像构建、网络推理和新 chunk 处理；主线程按固定 `publish_rate` 从动作缓冲区取动作发布，并处理缓存和触发。

**为什么可能有效**：

- 控制发布不会被单次推理延迟阻塞。
- 旧动作队列在推理期间继续提供平滑轨迹。
- 推理延迟与执行周期解耦，客户端可以稳定控制频率。

**代价/风险**：

- 线程间共享缓冲区需要锁和 generation/epoch 防护。
- 如果推理过慢，队列会耗尽，需有明确等待与安全停止。
- 任务切换时旧线程可能还在飞，必须取消并防止回写。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/scripts/inference/h01_client/nvidia_client_smooth.py`
- `giga-code/giga-models/projects/vla/giga-brain-0/scripts/inference/h01_client/nvidia_client.py`

**状态**：`已实现`。

---

## I-013 动作缓冲与不重复消费

**基础做法**：每个控制 tick 都从最新或首项动作读取，可能同一个动作被执行多次。

**公司改进**：每个 queue item 被 `consume()` 恰好一次；新旧 chunk 拼接时保留旧 buffer 的剩余段而非重新播放；report 记录 dispatched/candidates/consumer index。

**为什么可能有效**：

- 防止低发布频率下对同一 action 反复执行。
- 防止 TTRTC 重规划时旧动作被重新完整播放。
- 审计每步动作来源，便于回放和调试。

**代价/风险**：

- 索引和锁错误会导致空 buffer 或 action 丢失。
- 不重复消费不等于不重复发出；ROS 层仍按 tick 发布。
- 任务切换时必须 reset buffer，不能保留旧任务动作。

**代码**：

- client 主循环与 scheduler
- resident WBC session

**状态**：`启用`。

---

## I-014 动作命令行限与夹爪边界

**基础做法**：模型输出直接转换成 JointState 发布。

**公司改进**：

- 对选定连续关节限制相邻命令的绝对变化量；
- 对夹爪命令设置配置上下界；
- 失败时暂停推理并清空 action history，等待重新武装。

**为什么有效**：

- 模型或 chunk splice 偶发大跳时，先停止比把异常直接下发安全。
- 夹爪的数值单位和开合行程有限，明显越界通常是归一化或 layout 错误。
- reset command history 避免暂停后恢复时用旧 reference 重新判断。

**代价/风险**：

- 阈值过小会误停真实的快速动作。
- 门限只是异常拦截，不是速度或加速度安全控制器。
- 恢复时若没有重新确认传感器/状态，可能复用旧视觉。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/scripts/inference/h01_client/nvidia_client.py`
- `zhangzhong_wbc/server/src/core.py`

**状态**：`启用`。

---

## I-015 VR 或人工 teleop 接管暂停

**基础做法**：模型和人工同时向 joint_command 发布。

**公司改进**：监视外部 joint command publisher；检测人工命令突发后立即暂停模型；人工停止且安静一段时间后，要求用户明确重新恢复。

**为什么有效**：

- 避免模型在人工夹取或示教时继续抢控制权。
- 释放后不自动恢复可防止机器人突然动。
- 操作员可在真机危险处理中迅速接管。

**代价/风险**：

- 只监控 topic 名可能把其他正常控制器误判为人工接管。
- 接管和恢复时动作 reference 必须重置。
- 外部 publisher 存在但无命令流时，不能仅凭 publisher 名称就认定正在接管。

**代码**：

- client/robot adapter 的 takeover gate
- `MILKTEA_DEPLOYMENT_FLOW.md`

**状态**：`已实现`。

---

## I-016 推理断线后安全停止

**基础做法**：网络断了仍继续沿用旧 action buffer。

**公司改进**：推理或 WebSocket I/O 故障触发 pause，清 action history；resident stop verification 检查关节在停止后短窗口内静止；必要时保持 IDLE。

**为什么有效**：

- 在视觉条件已过期后继续播放动作可能撞墙。
- 旧 buffer 越走到末端越不可信。
- 用关节位置差确认没有残留命令仍在运动。

**代价/风险**：

- 静止证明的窗口和容差是启发式，不能保证物理急停。
- 保留多长旧动作存在安全取舍；宁可停下也不放大旧 buffer。
- 检查通过的静止状态不意味着任务完成。

**代码/文档**：

- client 主循环
- resident stop verification

**状态**：`已实现`。

---

## I-017 多帧时间戳缓存与 episode reset

**基础做法**：只有最新一帧，或按推理次数拼接历史。

**公司改进**：按 ROS header timestamp 保存每个相机的短历史，以秒/帧 offset 选取时间对齐清晰帧；缺帧用 pad mask；episode 起始时清本地和 server 历史。

**为什么可能有效**：

- 多帧模型依赖真实时间跨度，推理频率变化会破坏“每几次推理一帧”的 naive 历史。
- 启动和任务切换的第一帧不能被上一个任务的旧图像污染。
- 返回 mask 让模型区分“确实过去 0.5 秒”和“没有帧”。

**代价/风险**：

- 相机时钟和采集 timestamp 必须同步可信。
- 时间偏差和网络抖动会造成 frame gap。
- 多帧缓存增加内存和图像编码开销。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/scripts/inference/h01_client/nvidia_client_smooth_temporal.py`

**状态**：`已实现`。

---

## I-018 perception delay 与 server prefix

**基础做法**：请求里不告诉 server 距离观测已经过了几个控制周期。

**公司改进**：客户端记录本次推理实际消费/前置的动作长度，server 计算 runtime prefix steps；请求携带 prefix 和 inference delay；服务端重算输出并校验前 `delay` 行是否一致。

**为什么可能有效**：

- 模型看到的是过去时刻的观测，当前机器人可能已经执行若干步。
- 将真实 delay 显式确认为条件，比模型自己猜时间更准确。
- hard prefix 强制新 chunk 与实际已执行轨迹一致。

**代价/风险**：

- delay 不能超出训练 max delay。
- prefix 归一化和模型空间转换必须对称。
- 推理耗时超过 prefix 覆盖时，部分旧动作属于 unconditioned extrapolation，需监控。

**代码**：

- `zhangzhong_wbc/server/src/model.py`
- `zhangzhong_wbc/server/src/core.py`

**状态**：`已实现`。

---

## I-019 completion/progress 连续确认

**基础做法**：模型一帧给出高 completion 就停止。

**公司改进**：completion 和 progress 均过阈值，且连续多帧满足才停止；只在观测下降后建立 start-of-execution，避免 warmup 的项目高值误触发。

**为什么有效**：

- 单帧 status 有噪声。
- 任务初始状态可能偶然被模型判为完成。
- 与视觉变化同步可阻止旧状态缓存导致立即结束。

**代价/风险**：

- 连续帧阈值增加任务结束延迟。
- 若 status 是模型猜的，连续误判仍会结束，需 eval 和 timeout。
- 任务开始阈值逻辑要单独测试。

**代码**：

- client scheduler
- resident session

**状态**：`已实现`。

---

## I-020 只发送必要相机与压缩图像

**基础做法**：发送所有相机和原始分辨率。

**公司改进**：server 配置显式 `camera_names` 和顺序；client 只发送这些相机；图像先压缩，server 用准确的尺寸解码；只有任务训练所需相机进入视觉编码。

**为什么有效**：

- 发送多余相机增加编码、网络和 server 解码，却不会增加模型输入。
- 正确的相机子集降低端到端延迟。
- byte-size bounds 阻止异常大图或空图。

**代价/风险**：

- 选错相机子集会损失关键接触视图。
- 相机顺序不能只看集合相同。
- 压缩质量和去畸变与训练相机处理必须一致。

**代码**：

- server/client 配置及 `core.py`

**状态**：`启用`。

---

## I-021 客户端侧不盲目调大动作门限

**基础做法**：为“顺畅执行”把动作 gate 设得非常大。

**公司改进**：保留可配置的单步上限和夹爪上下界，异常时暂停并人工重新武装，不自动取消保护。

**为什么可能有效**：

- 一次实际大跳可能是模型输出、单位或 chunk splice 的真实故障。
- 连续执行并不等于正确执行。
- 安全暂停给现场人员判断和重置相机/机械臂状态的机会。

**代价/风险**：

- 门限与发布频率、动作语义绑定，需任务级标定。
- 误停会浪费执行窗口，应记录具体越界维度。
- 如果真实动作速度本身超过门限，应修模型/轨迹而不是静默放宽。

**代码**：

- client config `action_max_step`、`action_bounds`
- `zhangzhong_wbc/server/src/model.py`

**状态**：`启用`。
