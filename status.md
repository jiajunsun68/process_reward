# 实验状态

最后维护：2026-10-28 11:57:20 +08:00（Asia/Shanghai）。
当前阶段：G0/G1执行协议已准备，等待服务器环境核验。
本项目在本会话中已登记的服务器提交数：0。服务器队列未查询；没有可验证的Job ID或服务器PID。不能据此推断服务器全局没有排队任务。

执行边界：默认仅G0/G1。G2及后续为计划，未启动。协调者是status.md唯一writer；脚本和子agent不直接修改此表。详细规则见README第12节和AGENT_PROMPT.md。

## A. 已完成实验与工程核验

| RUN_ID | 状态 | 完成位置/日期 | 工作与结果 | 证据 | 范围限制 |
|---|---|---|---|---|---|
| CPU-01 | SUCCEEDED | 本地CPU / 2026-10-27 | 机制反例与2,772配置枚举；严格偏好反转436/2772；终端全可见控制价值差0 | results/probe_summary.json；results/grid.csv | 人工有限状态实验，不是LLM失败率 |
| CPU-02 | SUCCEEDED | 本地CPU / 2026-10-27 | 800配置；两组各4,768匹配pair，train1830/validation330/test2608 | data/manifest.json；results/dataset_summary.json | 同一数学模型；尚非三个机制audit数据 |
| CPU-03 | SUCCEEDED | 本地CPU / 2026-10-27 | 11项单元验证通过；动作解析和scenario读取检查通过 | results/verification.json | 不含GPU训练/模型推理 |
| ENG-01 | SUCCEEDED | 本地静态/CPU / 2026-10-28 | train默认不保存模型；显式final adapter；512MiB上限；内存语法、CLI与默认参数检查通过 | train_dpo.py；results/verification.json | 真实GPU训练、safetensors导出和adapter回读尚未验证 |

## B. 服务器排队 运行及待核验作业

本节只放实际发起提交或已有运行证据的attempt；计划放C节。
最近一次服务器成功核验：未发生。
当前观察来源：无服务器会话。当前查询错误：未发起查询。

| RUN_ID / ATTEMPT_ID | 状态 / 原始调度状态 | Job ID及cluster或主机/PID/启动时间 | GPU | 提交/开始/结束时间 | 最后核验时间 | 进度/排队原因 | 预算/证据/下一步 |
|---|---|---|---:|---|---|---|---|
| — | 未查询（非作业状态） | — | — | — | — | 当前没有可核实的本项目服务器作业记录 | 接入服务器后先核对；禁止将C节直接复制为QUEUED |

## C. 待执行计划

表中PLANNED和BLOCKED均不表示已经排队；不预创建输出目录。
每次实际执行使用固定逻辑RUN_ID加a01；修复重试最多a02，原attempt记录保留。

| RUN_ID | 阶段 / 目的 | 当前状态 | 依赖 | 资源与规模上限 | 计划保存 | 下一步 |
|---|---|---|---|---|---|---|
| G0-ENV | 服务器、权限、驱动、缓存、磁盘及队列核验 | BLOCKED | 服务器会话未提供 | CPU只读检查 | status内摘要；必要时单一log | 在实际服务器确认环境，不猜Job ID |
| G0-SMOKE | 已分配GPU的BF16检查 | PLANNED | G0-ENV | 1 GPU，短作业 | 单一log；0模型 | 运行gpu_smoke.py |
| G0-PROFILE | 独立训练显存/吞吐profile | PLANNED | G0-SMOKE、固定模型commit | 1 GPU-hour，20steps，256train/64val | run.json + metrics.jsonl + log；0模型 | 仅评工程指标，不触发研究gate |
| G1-DATA | 三机制case、posterior与固定π gold | PLANNED | G0-ENV、CPU gold定义 | 360根case；72dev/288audit | data/audit/cases.jsonl与manifest.json | 实现audit_cases.py；不复用换皮充当多机制 |
| G1-DEV | 解析/格式/吞吐、最多一次提示修订 | PLANNED | G1-DATA、模型准备 | 先24dev后72dev，最多两版提示 | 单一run记录+predictions+log | 固定模型、格式、预算 |
| G1-FREEZE | 冻结数据、提示、指标及72-case对照子集 | PLANNED | G1-DEV | CPU | 更新同一manifest/status | audit开始前冻结hash |
| G1-H | 历史教师audit | PLANNED | G1-FREEZE | 288次短调用；1 GPU | 单一run记录+predictions+log | 不提供隐藏信息 |
| G1-P | 隐状态教师audit | PLANNED | G1-FREEZE | 768次短调用；1 GPU | 同上 | 同case各世界posterior加权 |
| G1-PR | 隐状态+参考动作audit | PLANNED | G1-FREEZE | 768次短调用；1 GPU | 同上 | 保留参考评分语义 |
| G1-PRIC | 增加信息可实施约束audit | PLANNED | G1-FREEZE | 768次短调用；1 GPU | 同上 | 主对比为PR→PRIC |
| G1-FULLREF | 参考完备性对照 | PLANNED | G1-FREEZE | 72case，共384次短调用 | 同上 | 不借用history gold补答案 |
| G1-MC | 相同history-only π的MC | PLANNED | G1-FREEZE | 已知π优先CPU；可选LLM最多2592次生成 | 单一run记录+predictions；必要时log | 不把4rollout当gold |
| G1-ANALYZE | 后验聚合、case bootstrap与gate | PLANNED | G1各固定条件完成或预算终止 | 2000次bootstrap，seed23 | 单一汇总写run.json；status写结论 | 允许no-go或证据不足 |
| G2-MATRIX | 4标签×3seed偏好学习 | PLANNED（后续范围） | G1证据与残留缺口、另立训练任务 | 最多12正式训练run；预算待profile | 每run至多1个final adapter | 本次不提交 |
| G3-G4 | PRM/在线RL/独立工具环境 | PLANNED（后续范围） | G2及明确新预算 | 未开放 | 未开放 | 本次不提交 |

## D. 预算 阻塞与阶段决策

### 当前账目

| 项目 | 协议上限 | 当前已登记使用/预留 | 执行与核验说明 |
|---|---:|---:|---|
| G0/G1 GPU时间 | 36 GPU-hours | 本会话提交0，因此已登记0/预留0 | 服务器既有任务未查询；接入后先核对 |
| 并发GPU任务 | 2 | 已登记0 | 每个任务1 GPU，遵守调度器分配 |
| 真正QUEUED任务 | 4 | 无可核验记录 | 不能把“未查询”写成服务器全局0 |
| 教师调用 | 4272 | 0 | 含第二版dev及全部attempt；首轮+fullref为3624 |
| 可选LLM-MC生成 | 2592 | 0 | 同π CPU-MC另记，不占GPU |
| G0/G1模型保存 | 0 | 0 | profile不得传save-final-adapter |
| G0/G1新实验产物 | 2 GiB | 未在服务器产生 | 含runs/logs/data/audit及失败/临时空间和预留；服务器quota未核验 |
| 单run非模型产物 / 单日志 | 128 MiB / 10 MiB | 无服务器run | 达限停止新增或结束该作业，不循环产生日志 |
| 新合成数据 | 128 MiB | 三机制audit尚未生成 | 已有toy数据保留、复用 |
| 新增模型缓存 | 24 GiB、1个底模revision | 本会话下载0 | 服务器共享缓存未查询 |
| 后续final adapter | 每run≤512 MiB，正式矩阵≤12个 | 0 | 尚不在本轮范围 |

### 阻塞与决策

| 项目 | 当前结论 | 解决条件 |
|---|---|---|
| 服务器访问 | 未建立 | 在用户实际服务器执行G0-ENV；不要自行寻找凭据 |
| 三机制audit入口 | 尚未实现 | 本轮允许新增audit_cases.py和audit_teacher.py两文件 |
| 模型revision | 尚未在服务器固定 | 解析唯一缓存commit并写manifest/run记录 |
| 保存策略GPU验证 | 仅静态/CPU检查完成 | 先验证无保存profile；正式adapter回读在后续首次保存时验证 |
| 初始研究gate | NOT_EVALUATED | 完成固定audit与对照，不以toy或20step profile替代 |
| 磁盘强制quota | 未核验 | 如只有目录检查，明确标应用层限制，不称OS硬限额 |
| 续训能力 | resume_supported=false | 本方案不保存完整checkpoint；失败只能受控新attempt |
| 下一阶段 | 不启动 | G1交付证据、残留缺口和实测资源账后另立执行任务 |

### 最近维护

- 2026-10-28：补齐执行协议、status初始表与文件/计算预算；训练入口默认0模型保存。
- 既有2026-10-27 CPU结果保留；本次没有新增服务器提交、GPU训练或模型推理。

### 协调者更新规则

提交前写唯一attempt、实际job-name/comment标签、完整命令、依赖、代码/数据/model版本、计算及磁盘预留、输出路径和SUBMITTING；真实回执后再查队列。Slurm查当前squeue与结束后的sacct；无调度器验证主机/PID/启动时间。提交结果不确定时先核实，不能重交。
作业结束先VERIFYING，校验退出码、预期步数、结果和文件白名单后SUCCEEDED；失败保留记录。断线不刷新last_verified_at，另记当前观察错误。
一个协调者串行修改本文件；用同目录.status.md.tmp写完后原子替换。任务不直接写status；不创建状态副本或后台daemon。

