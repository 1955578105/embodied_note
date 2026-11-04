# 部署方法与技巧

本文记录在“每台机器手工起一个脚本，客户端直接连一个模型端口，返回动作后直接发 ROS”这一基础做法之上的改进。

## 索引

| ID | 方法 | 主要收益 | 状态 |
|---|---|---|---|
| P-001 | Agent / MCP / client / server 分层 | 决策、编排、控制和模型解耦 | 启用 |
| P-002 | server 只推理，client 独占 ROS 发布 | 故障边界清晰 | 启用 |
| P-003 | 显式 wire protocol 与 metadata | 在真机动作前发现不匹配 | 启用 |
| P-004 | 多任务固定端口路由 | 单 client 可切换任务而不用重连 Agent | 启用 |
| P-005 | server 单 client lease | 防两个控制源同时占用模型 | 启用 |
| P-006 | 启动后 warmup 全 delay 分支 | 把编译成本移到任务启动前 | 启用 |
| P-007 | warmup 可复现与 action checksum | 比较不同 adapter/后端输出 | 已实现 |
| P-008 | 训练相机顺序预检 | 避免 raw/rect 或相机错序 | 启用 |
| P-009 | checkpoint、norm、tokenizer 全路径检查 | 避免加载错资产 | 启用 |
| P-010 | 单一 image mode 开关跨两端同步 | 防训练部署图像域错配 | 启用 |
| P-011 | 远端启动只转发必要环境 | 统一主机和机器人配置 | 启用 |
| P-012 | lock 与 owned PID | 防重复 client/server | 启用 |
| P-013 | 优雅停止与升级后强制结束 | 避免残留服务占用端口 | 启用 |
| P-014 | 多 server 预启动与复用 | 降低 Agent 等待和冷启动 | 已实现 |
| P-015 | Agent actions catalog 与 server 路由分离 | 面向用户命名与实现解耦 | 启用 |
| P-016 | task status 完成信号 | 自动结束控制而不依赖固定时长 | 已实现 |
| P-017 | 压缩后二进制图像协议 | 降低网络带宽和编码开销 | 启用 |
| P-018 | 可恢复配置备份 | 快速回滚生产配置 | 已采用 |
| P-019 | 静态端口和服务器分区 | 便于排障与资源规划 | 启用 |
| P-020 | health check 先做只读探测再执行 | 降低启动竞态风险 | 已实现 |

---

## P-001 Agent / MCP / client / server 分层

**基础做法**：语音或规划程序同时负责理解、加载模型、读相机和发布 ROS。

**公司改进**：分层为：

```text
Agent/LLM
  → MCP tool
  → robot client
  → VLA server
  → ROS joint commands
```

**为什么有效**：

- Agent 只决定“做什么”，无需持有机器人控制细节。
- MCP 把自然语言任务转换为受控 action/pipeline task。
- robot client 负责传感器、队列、安全门和 ROS。
- VLA server 纯推理，可独立部署在 GPU 服务器。

**代价/风险**：

- 多一层就多一种协议和生命周期问题。
- 必须清楚同名概念：AgentOS 的 `client` 并不是 robot client。
- task status 从 server 到 client 到 Agent 必须端到端透传。

**适用场景**：多任务厨房、需要语音控制和大模型远程推理的机器人。

**代码/文档**：

- `MILKTEA_DEPLOYMENT_FLOW.md`
- `zhangzhong_wbc/server/script/start_server_wbc.sh`
- `zhangzhong_wbc/z_run_client_sh/run_client.sh`

**状态**：`启用`。

---

## P-002 server 只推理，client 独占 ROS 发布

**基础做法**：推理服务器直接连接机器人 ROS 或由多服务共同发布关节。

**公司改进**：server 仅处理模型输入并返回动作；robot client 负责读 ROS、编码图像、队列调度、动作上限检查和安全暂停。

**为什么可能有效**：

- GPU 服务崩溃时不会直接发布半截 ROS 命令。
- 控制频率、命令历史和停机验证在机器人本机完成。
- 网络连接断开时客户端能安全停止，而不是留一个远端发布器失控。

**代价/风险**：

- client 和服务端的动作维度、归一化、前缀协议必须严格匹配。
- 网络往返时间进入控制闭环。
- resident daemon、独立 client 和外部 teleop 的发布所有权必须协调。

**代码**：

- `zhangzhong_wbc/server/src/model.py`
- `giga-code/giga-models/projects/vla/giga-brain-0/scripts/inference/h01_client/nvidia_client.py`
- `zhangzhong_wbc/server/src/server.py`

**状态**：`启用`。

---

## P-003 显式 wire protocol 与 metadata

**基础做法**：服务端只有裸 TCP/JSON，字段缺失时用默认值补齐。

**公司改进**：

- 连接后 server 先返回 metadata；
- client 验证 protocol、state/action width、action horizon、camera 顺序、prompt、TTRTC、task status；
- 每个请求验证 request id、generation、start step、prefix 长度、state shape、image key 集合和单张图字节范围；
- 服务端校验 hard prefix 是否被模型保持。

**为什么有效**：

- 大量错误配置仍可产出“形状正确”的错误动作；metadata 会在执行前拦截。
- request id/generation 防止旧任务结果被新任务消费。
- 动作和前缀校验阻止陈旧 chunk 拼接。

**代价/风险**：

- 增加协议维护面。
- metadata 校验过严会增加升级兼容成本。
- 仅校验字段存在不代表语义正确，仍需端到端数值测试。

**代码**：

- `zhangzhong_wbc/server/src/core.py`
- `zhangzhong_wbc/server/src/server.py`
- `zhangzhong_wbc/z_run_client_sh/run_client.sh`

**状态**：`启用`。

---

## P-004 多任务固定端口路由

**基础做法**：一个模型一个部署脚本，每换任务重新 SSH、加载模型和启动 client。

**公司改进**：MCP 的 actions catalog 把动作名映射到 pipeline task；任务路由再映射到固定 host/port。常驻 client 可在任务之间切换，server 保持连接。

**为什么有效**：

- 避免每个任务都承担模型冷启动。
- Agent 只看到业务动作名，不必知道 server 地址。
- 排障时端口和任务可直接对应。

**代价/风险**：

- 端口表与 actions catalog 可能双写漂移。
- 多 server 共享 GPU 时要有显存预算。
- 若支持“单 client 多 server”，必须明确模型切换、队列清空和 reset 语义。

**代码/文档**：

- 现场路径：机器人 `robot_mcp/vla/actions.json`（AgentOS 部署代码，不在当前工作区）
- 现场路径：机器人现场环境中的 `agentos_vla_resident/resident_routes.py`
- 工作区链路说明：`MILKTEA_DEPLOYMENT_FLOW.md`
- 工作区部署记录：`MILKTEA_HEAD_DEPLOYMENT_20261028.md`

**状态**：`启用`。

---

## P-005 server 单 client lease

**基础做法**：一个 WebSocket server 接受多个 client，并允许它们并发推理。

**公司改进**：server 使用一个 async lease；有 client 持有时，新连接立即用关闭码 `1013` 拒绝。

**为什么可能有效**：

- 防止两条控制流交错写入同一条动作队列。
- 模型内部有推理状态和 TTRTC prefix，天然不适合多个独立 session 并发。
- 单 client 故障时能快速释放，后续 client 可独占。

**代价/风险**：

- 旧连接异常残留时，新连接会被拒。
- 启动、重连和 MCP task 超时之间需要正确释放 lease。
- 单一 lease 限制横向共享同一 server；多机器人必须多端口/多实例。

**代码**：

- `zhangzhong_wbc/server/src/server.py`

**状态**：`启用`。

---

## P-006 启动后 warmup 全 delay 分支

**基础做法**：服务开始监听后等第一个真实请求触发编译。

**公司改进**：启动后先构造 dummy observation；从 delay=0 得到初始动作，再把前若干动作作为 prefix，依次 warmup delay 6–10；只有全部完成后才进入 READY 和 WebSocket监听。

**为什么可能有效**：

- TTRTC 的 prefix 长度会改变模型图和数据路径，只 warmup delay=0 不代表其他分支已编译。
- 把编译和首次初始化成本从用户任务中移走，降低首动作超时。
- 提前暴露输出 shape、NaN、prefix 保持错误。

**代价/风险**：

- 启动时间显著增长。
- dummy 输入若与原任务图像/相机分辨率不一致，warmup 分支仍可能在正式任务重编译。
- warmup 产生的动作不能用于控制，只用于初始化和一致性比较。

**代码**：

- `zhangzhong_wbc/server/src/server.py`
- `zhangzhong_wbc/server/configs/base.py`

**状态**：`启用`。典型 delay `(0,6,7,8,9,10)`。

---

## P-007 warmup 可复现与 action checksum

**基础做法**：只看 warmup 没有报错。

**公司改进**：给 warmup flow noise 固定 seed；打印每次模型耗时、输出 status 和 action SHA-256；支持多 adapter 比较脚本。

**为什么有效**：

- 不同优化后端、版本或 codec 可输出 shape 正确但数值错误。
- checksum 和相同 seed 可用于跨机器、跨后端回归。
- 比人工看前几个浮点数更容易发现微小耦合错误。

**代价/风险**：

- checksum 对硬件非确定性和合理优化差异很敏感，需设容差或比较指标。
- dummy input 不是业务分布，checksum 相同不代表真机行为相同。
- warmup 记录也不应替代端到端测试。

**代码**：

- `zhangzhong_wbc/server/src/server.py`
- `zhangzhong_wbc/server/script/compare_model_adapters.py`

**状态**：`已实现`。

---

## P-008 训练相机顺序预检

**基础做法**：模型加载后才知道相机 key 不匹配。

**公司改进**：启动 server 前只读取 checkpoint 的 `inference_config.json`，比较 `present_img_keys` 和 server 配置的相机顺序；不加载模型、不占 GPU。

**为什么有效**：

- raw 和 rect 模型使用不同相机名，错误组合会导致动作质量悄悄下降。
- 相机顺序错误比形状错误更隐蔽，metadata 也可能只比较集合。
- 预检快速失败，避免先占 GPU 再报错。

**代价/风险**：

- sidecar 必须与真实权重重同步，不能只改 `config.json`。
- 预检只检查命名，不检查相机物理安装是否正确。
- repack 逻辑可在数据端改变名称，部署端仍要独立核对。

**代码**：

- `zhangzhong_wbc/server/script/start_server_wbc.sh`
- `zhangzhong_wbc/server/src/model.py`

**状态**：`启用`。

---

## P-009 checkpoint、norm、tokenizer 全路径检查

**基础做法**：只检查 checkpoint 是否存在。

**公司改进**：启动前检查 checkpoint 目录、权重、norm、普通 tokenizer 和 FAST tokenizer；加载后检查 checkpoint config 与 service config 的 TTRTC、horizon、action width、state mode、VLM type 和 task-status head。

**为什么有效**：

- 微调模型可能结构正确，但 norm 文件有误，导致动作数值完全错位。
- tokenizer 缺失或不同词表会影响语言和 FAST 动作 token。
- 把错误推迟到推理中途会消耗大量冷启动时间。

**代价/风险**：

- 符号链接和离线模型目录必须稳定。
- 文件存在不等于 SHA 正确；关键发布仍应做哈希验证。
- 多版本 norm 与 checkpoint 应一同版本化。

**代码**：

- `zhangzhong_wbc/server/src/model.py`
- `zhangzhong_wbc/server/configs/base.py`
- `zhangzhong_wbc/manage_server.sh`

**状态**：`启用`。

---

## P-010 单一 image mode 开关跨两端同步

**基础做法**：server 与 robot client 各自手工改相机名和 rect 参数。

**公司改进**：任务 launcher 的 `IMAGE_MODE` 作为唯一配置源；server 启动时 source 同一文件并导出 image mode 与标定参数，再转发给远端 client。

**为什么有效**：

- raw/rect 必须和训练模型一致；两端不同会得到看似正常但错位的输入。
- 相机标定只维护一处，避免 server 和 client 拷贝不同。
- 配置名错误在 shell 层直接拒绝。

**代价/风险**：

- 环境变量和配置文件仍然有多种作用域，易误以为改一处有效。
- 修改后必须重启两端，运行中不会热更新。
- rect 标定图像尺寸、K/D 和 target FOV 需绑定训练数据。

**代码**：

- `zhangzhong_wbc/z_run_client_sh/run_microwave_client.sh`
- `zhangzhong_wbc/server/script/start_server_wbc.sh`
- `zhangzhong_wbc/z_run_client_sh/run_client.sh`

**状态**：`启用`。

---

## P-011 远端启动只转发必要环境

**基础做法**：SSH 到机器人后在 shell 里再维护一套参数。

**公司改进**：主机脚本以 shell-quoted environment 前缀向远端传递 host、port、client config、image mode 和标定 JSON，然后直接 exec 机器人上的 launcher。

**为什么有效**：

- 主机是唯一任务入口，避免在两端手工同步。
- 显式 quoting 避免路径空格和特殊字符被再次解释。
- SSH BatchMode、ConnectTimeout、keepalive 提前约束远端不可达。

**代价/风险**：

- 远端和主机的代码/配置版本仍需同步。
- 长标定 JSON 放进环境变量可能增加审查和调试成本。
- 两端 SSH key 和 host alias 状态会影响启动。

**代码**：

- `zhangzhong_wbc/z_run_client_sh/run_client.sh`
- 本地源码：`kitchen_demo_launcher/start.sh`

**状态**：`启用`。

---

## P-012 lock 与 owned PID

**基础做法**：用 PID 文件认为服务在运行；每次启动新的进程。

**公司改进**：

- client 用文件锁阻止同一机器人上的多 client 并发执行；
- server 通过 PID 文件检查进程是否仍属于预期 launch；
- 过期 PID 文件被视为 stale 并清理；
- 启动后快速确认进程存活，失败即打印 tail。

**为什么有效**：

- 旧进程退出后 PID 文件可能残留，单看 PID 会误判。
- 多 client 同时下发 ROS 命令非常危险。
- 避免重复 server 抢占内存和端口。

**代价/风险**：

- 锁忘了释放会导致“another client owns the lock”。
- 对 PID 复用、不同用户和容器 namespace 需额外识别。
- 强制 kill 是最后手段，正常应先走 graceful stop。

**代码**：

- `zhangzhong_wbc/manage_server.sh`
- client lock path 配置
- resident stop verification

**状态**：`启用`。

---

## P-013 优雅停止与强制结束

**基础做法**：使用 `Ctrl+C`，或直接 kill 进程。

**公司改进**：先发送终止信号并等待一段时间；超时才 `SIGKILL`；resident client 在停止后检查关节话题静止；checkpoint/launcher 显式传播子进程退出码。

**为什么有效**：

- graceful shutdown 允许关闭 WebSocket、flush 日志、释放 GPU。
- 直接强杀会留下拉取中的 lease 和残留连接。
- 静止验证区分“进程杀掉”和“机器人确实没有命令输出”。

**代价/风险**：

- 等待时间会和用户感知的“退出慢”冲突。
- 话题静止检查只能证明短窗口无运动，不能替代急停。
- 子系统自身可能忽略 SIGTERM，需要明确升级策略。

**代码**：

- `zhangzhong_wbc/manage_server.sh`
- resident worker stop verification

**状态**：`启用/已实现`。

---

## P-014 多 server 预启动与复用

**基础做法**：Agent 收到动作后才 SSH 启动模型 server。

**公司改进**：在演示或任务前，用总启动器预启动多个 server 并完成 warmup；Agent 执行时只选择已就绪端口。

**为什么有效**：

- 避免用户在语音任务中等待数分钟模型加载和编译。
- 多个任务可共享已编译的 runtime 资源。
- 任务失败不再混入模型冷启动问题。

**代价/风险**：

- 每台 GPU 上多个 server 持续占用显存。
- 预启动的模型、配置或哈希若过期，会造成假就绪。
- 总启动器失败时必须逐个端口验证，不要仅根据脚本返回 0。

**代码/文档**：

- 本地工作区：`kitchen_demo_launcher/`
- `MILKTEA_DEPLOYMENT_FLOW.md`

**状态**：`已实现`，实际部署按任务现场使用。

---

## P-015 actions catalog 与 server 路由分离

**基础做法**：Agent 直接调用固定模型 URL 或函数。

**公司改进**：

- actions catalog 定义公开动作名、描述、pipeline task 或 replay；
- VLA server 定义端口、模型、prompt；
- Agent 按描述选中动作，MCP 再映射到具体执行程序。

**为什么有效**：

- 用户语言和实现细节解耦，便于重命名或换模型。
- 可以限制公开能力面，不让 Agent 任意拼服务器 URL。
- replay 类和 VLA 类动作可在同一接口下表达。

**代价/风险**：

- 修改 action 名可能破坏 Agent prompt、old task state 或测试。
- 描述和真实 pipeline prompt 不一致会造成错误路由。
- 多 server 模式需要 MCP 支持按任务动态选择目标，而非固定一个 host。

**代码**：

- 现场路径：机器人 `robot_mcp/vla/actions.json`（AgentOS 部署代码，不在当前工作区）
- 现场路径：VLA MCP worker/adapter
- 工作区链路说明：`MILKTEA_DEPLOYMENT_FLOW.md`

**状态**：`启用`。

---

## P-016 task status 完成信号

**基础做法**：执行固定步数或固定时间后停止。

**公司改进**：模型可选输出 `completion` 和 `progress`；client 在两者达到阈值且连续多帧满足时才结束，并设置 completion required。

**为什么有效**：

- 不同任务的数据长度差异很大，固定时长会早停或拖尾。
- 连续多帧阈值减少偶发高值导致误结束。
- 可从 status 曲线和保存的样本判断为什么结束。

**代价/风险**：

- status head 必须真正训练并使用匹配 norm/head 权重。
- 完成 token/head 与基础 action head 是否冻结或训练需明确。
- 完成判定依赖模型质量，检测不到完成时会持续运动，需要上层 timeout。

**代码**：

- `giga-code/giga-models/projects/vla/giga-brain-0/giga_brain_0/giga_brain_0_loss.py`
- server config 的 `enable_task_status`
- 客户端 completion/streak 配置

**状态**：`已实现`，按 raw/rect 模型能力启用。

---

## P-017 压缩后二进制图像协议

**基础做法**：传原始 numpy tensor 或 JSON 数组。

**公司改进**：robot client 将多路相机编码为 JPEG，服务端在推理前解码；请求使用 MessagePack 二进制；每张图设置字节上下限，限制最大消息尺寸。

**为什么有效**：

- 原始 1920×1536×YUV 图像远大于 JPEG。
- MessagePack 二进制比 JSON 浮点数组和 base64 图片更紧凑。
- 服务端可以独立限制单张图和整条消息，防止异常帧耗尽内存。

**代价/风险**：

- JPEG 有损压缩会改变视觉输入，训练与部署必须匹配质量。
- 编解码进入控制延迟，尤其在 CPU 受限的机器人侧。
- 字节上限过小会拒绝合法高细节图像；过大又失去防护意义。

**代码**：

- `zhangzhong_wbc/server/src/server.py`
- `zhangzhong_wbc/server/src/model.py`
- `zhangzhong_wbc/server/configs/base.py`

**状态**：`启用`，典型 `jpeg_quality=80`。

---

## P-018 可恢复配置备份

**基础做法**：原地修改文件，出错后手工回滚。

**公司改进**：修改生产配置前复制时间命名备份；部署文档记录源/目标路径和哈希；使用可复现的 launcher 和 staging 目录。

**为什么有效**：

- 可以快速恢复服务配置，而不依赖记忆。
- 权重、norm、client config 的 SHA 校验让部署证据可复核。
- 时间戳或 transaction ID 让配置文件之间有对应关系。

**代价/风险**：

- 备份必须能还原，不只是存在。
- 动态生成或环境覆盖的配置不一定随源文件一起备份。
- 备份日志不得包含明文凭据。

**代码/文档**：

- `MILKTEA_FROZEN_DEPLOYMENT_20261029.md`
- `MILKTEA_HEAD_DEPLOYMENT_20261028.md`

**状态**：`已采用`。

---

## P-019 静态端口与服务器分区

**基础做法**：每次动态找可用端口，由环境变量自行传递。

**公司改进**：常用任务使用固定端口和固定机器分区，例如 8027 微波炉、8028 清桌、8029 洗碗机、8030 奶茶，并记录 hostname 与端口映射。

**为什么有效**：

- 启动与排障可直接定位 server 和日志。
- Agent、MCP、client 与运维文档可以共享同一个路由表。
- 端口固定使 netstat、日志和 GPU 进程容易对应。

**代价/风险**：

- 端口冲突后若误以为旧进程仍在，会连到错误服务。
- 多 client 或多 server 情况下，端口不能代表真正 task 合约。
- 必须同时检查 host、port、metadata 和 PID，而不是只检查 TCP open。

**状态**：`启用`。

---

## P-020 health check 先做只读探测再执行

**基础做法**：收到任务直接连接模型并开始发布。

**公司改进**：resident 先查询 robot client 状态、检查 daemon 是否运行、是否存在 active clients，再 prepare；prepare 后必须收到 READY，然后再 activation；完成后做静止验证。

**为什么有效**：

- 模型还没有准备完成时不启动控制发布。
- active client 或外部采集存在时拒绝介入。
- 任务终态与“进程存活”分开验证。

**代价/风险**：

- 多级状态机增加超时点和 traceback，要定位是哪一步失败。
- health output 必须足够详细，否则“握手超时”仍无法区分连接、metadata、传感器和模型准备。
- 状态查询本身也会因网络或进程权限失败。

**代码**：

- resident worker/probe
- `MILKTEA_DEPLOYMENT_FLOW.md`

**状态**：`已实现`。
