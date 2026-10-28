# 实验状态

最后维护：2026-10-28 19:41:57 +08:00（Asia/Shanghai）。
当前阶段：G0/G1已完成并最终核验；预注册gate结论为“机制证据不足/no-go”。G2、PRM、在线RL、第二模型和付费API均未启动。
本项目在本会话中已登记的直接运行attempt数：4；全部SUCCEEDED，无FAILED/CANCELED、无运行中或排队任务。当前主机未安装Slurm命令，因此无Slurm队列可查。

执行边界：默认仅G0/G1。G2及后续为计划，未启动。协调者是status.md唯一writer；脚本和子agent不直接修改此表。详细规则见README第12节和AGENT_PROMPT.md。

## A. 已完成实验与工程核验

| RUN_ID | 状态 | 完成位置/日期 | 工作与结果 | 证据 | 范围限制 |
|---|---|---|---|---|---|
| CPU-01 | SUCCEEDED | 本地CPU / 2026-10-27 | 机制反例与2,772配置枚举；严格偏好反转436/2772；终端全可见控制价值差0 | results/probe_summary.json；results/grid.csv | 人工有限状态实验，不是LLM失败率 |
| CPU-02 | SUCCEEDED | 本地CPU / 2026-10-27 | 800配置；两组各4,768匹配pair，train1830/validation330/test2608 | data/manifest.json；results/dataset_summary.json | 同一数学模型；尚非三个机制audit数据 |
| CPU-03 | SUCCEEDED | 本地CPU / 2026-10-27 | 11项单元验证通过；动作解析和scenario读取检查通过 | results/verification.json | 不含GPU训练/模型推理 |
| ENG-01 | SUCCEEDED | 本地静态/CPU / 2026-10-28 | train默认不保存模型；显式final adapter；512MiB上限；内存语法、CLI与默认参数检查通过 | train_dpo.py；results/verification.json | 真实GPU训练、safetensors导出和adapter回读尚未验证 |
| G0-ENV-CHECK | SUCCEEDED | klclab-X660-G45 / 2026-10-28 | 无Slurm；Python 3.12.13；torch 2.12.0+cu130可见8张A800 80GB；目标Qwen缓存未找到；/data4可用约13.75GB | 现场命令输出已由协调者核验；status.md | nvidia-smi因driver/library mismatch失败；8张可见不等于分配给本项目；未运行GPU smoke |
| G0-SMOKE | SUCCEEDED | klclab-X660-G45 / 2026-10-28 | 物理GPU 0和6均完成128x128 BF16矩阵验证；每进程仅可见1张A800 80GB；结果有限且正确 | logs/G0-SMOKE0-a01.log；logs/G0-SMOKE6-a01.log | 峰值allocated 8.27MiB/reserved 22MiB；NVML警告仍存在；这不是模型加载或训练 |
| G0-MODEL | SUCCEEDED | klclab-X660-G45 / 2026-10-28 | ModelScope唯一snapshot；commit `2de2439ea21be1dc5cb21f22f88af07e43393cbb`；3个权重分片和索引SHA-256/尺寸均匹配官方元数据 | /data4/sjj/.cache/modelscope/.../snapshots/2de2439...；logs/G0-MODEL-a01.log | 缓存8,060,942,868 bytes；Qwen3ForCausalLM/BF16；纯本地tokenizer可加载；未进行模型推理 |
| G1-DATA | SUCCEEDED | CPU / 2026-10-28 | 360根case；三机制各120；每层40；72dev/288audit；72-case full子集；真实posterior精确Qπ和同π CPU-MC | data/audit/cases.jsonl；data/audit/manifest.json | 根case是独立n；276,480个MC rollout为嵌套对照，不增n |
| G1-DEV-FREEZE | SUCCEEDED | 物理GPU 0 / 2026-10-28 | 648唯一dev调用；格式648/648=100%；无提示修订；prompt/data/model/metric/gate已冻结 | runs/G1-TEACHER-a01/run.json；predictions.jsonl；data/audit/manifest.json | dev不是audit效果；不触发研究gate；0模型保存 |
| G1-AUDIT | SUCCEEDED | 物理GPU 0 / 2026-10-28 | audit 2,976次；H=288、P/PR/PR-IC各768、PR-full/PR-full-IC各192；格式2,976/2,976=100% | runs/G1-TEACHER-a01/run.json；runs/G1-TEACHER-a01/predictions.jsonl | 全attempt累计3,624次调用；无无效/未知调用；0模型保存 |
| G1-ANALYZE | SUCCEEDED / NO-GO | CPU / 2026-10-28 | Δ=mean(r_PR-r_PR-IC)=-0.001341，95% CI [-0.002671, 0.000041]；未达+0.01且不支持正向；完备性后残余Δ=-0.000331 | runs/G1-TEACHER-a01/run.json | 2,000次按机制分层、根case bootstrap，seed23；结论是证据不足，不自动进入G2 |

## B. 服务器排队 运行及待核验作业

本节只放实际发起提交或已有运行证据的attempt；计划放C节。
最近一次当前主机成功核验：2026-10-28 19:41:57 +08:00。
当前观察来源：klclab-X660-G45交互会话。当前查询限制：无squeue/sbatch/sacct；nvidia-smi报NVML driver/library version mismatch。

| RUN_ID / ATTEMPT_ID | 状态 / 原始调度状态 | Job ID及cluster或主机/PID/启动时间 | GPU | 提交/开始/结束时间 | 最后核验时间 | 进度/排队原因 | 预算/证据/下一步 |
|---|---|---|---:|---|---|---|---|
| G0-SMOKE0 / G0-SMOKE0-a01 | SUCCEEDED / 无调度器 | klclab-X660-G45；PID 3586916；启动2026-10-28 14:48:37 +08:00 | 1（物理0→逻辑0） | 登记14:47:37；开始14:48:37；结束14:48:41 +08:00 | 2026-10-28 14:48:41 +08:00 | rc=0；BF16 finite=true、expected_result=true；0模型 | 实用约0.0011 GPU-hour；`PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 python gpu_smoke.py --device 0`；logs/G0-SMOKE0-a01.log；无下一步 |
| G0-SMOKE6 / G0-SMOKE6-a01 | SUCCEEDED / 无调度器 | klclab-X660-G45；PID 3586918；启动2026-10-28 14:48:37 +08:00 | 1（物理6→逻辑0） | 登记14:47:37；开始14:48:37；结束14:48:41 +08:00 | 2026-10-28 14:48:41 +08:00 | rc=0；BF16 finite=true、expected_result=true；0模型 | 实用约0.0011 GPU-hour；`PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=6 python gpu_smoke.py --device 0`；logs/G0-SMOKE6-a01.log；无下一步 |
| G0-MODEL / G0-MODEL-a01 | SUCCEEDED / 无调度器 | klclab-X660-G45；PID 3598217；启动2026-10-28 14:57:33 +08:00 | 0 | 登记14:56:25；开始14:57:33；结束15:26:35 +08:00 | 2026-10-28 15:28:13 +08:00 | rc=0；无incomplete；仅1个snapshot；官方SHA-256/尺寸全部匹配；config/tokenizer纯本地加载成功 | 实用8,060,942,868 bytes/24GiB；log 1,921,980 bytes/10MiB；commit `2de2439ea21be1dc5cb21f22f88af07e43393cbb`；下一步G1-DEV24 |
| G1-AUDIT / G1-TEACHER-a01 | SUCCEEDED / 无调度器 | klclab-X660-G45；dev PIDs 3637642/3641582、audit PID 3646248均已结束 | 1（物理0→逻辑cuda:0） | dev24 15:30:09–15:30:47；dev72 15:33:42–15:34:53；audit 15:36:48–15:44:14 +08:00 | 2026-10-28 19:41:57 +08:00 | dev24/dev72/audit288全部rc=0；3,624条唯一且完整；格式100%；0 unknown/inflight/error；文件白名单通过；GPU 0/6空闲 | 实际三阶段GPU walltime约0.1542小时；峰值allocated/reserved 8.30/9.13GiB；run+log约2.29MB；0模型；证据runs/G1-TEACHER-a01、logs/G1-TEACHER-a01.log |

## C. 待执行计划

表中PLANNED和BLOCKED均不表示已经排队；不预创建输出目录。
每次实际执行使用固定逻辑RUN_ID加a01；修复重试最多a02，原attempt记录保留。

| RUN_ID | 阶段 / 目的 | 当前状态 | 依赖 | 资源与规模上限 | 计划保存 | 下一步 |
|---|---|---|---|---|---|---|
| G0-ENV | 服务器、权限、驱动、缓存、磁盘及队列核验 | SUCCEEDED | GPU 0/6 smoke成功；ModelScope唯一commit官方hash校验成功 | CPU检查+两个极小smoke | status与G0单一logs | 不重复建环境/缓存；NVML仅作为已知监控限制 |
| G0-SMOKE | 已分配GPU的BF16检查 | SUCCEEDED | 用户授权物理GPU 0和6 | 2个并行短作业，每卡1个 | 每attempt单一log；0模型 | 不重复smoke；模型revision固定后才可考虑唯一20-step profile |
| G0-PROFILE | 独立训练显存/吞吐profile | NOT_RUN（本轮非必要） | G0-SMOKE、固定模型commit | 协议允许至多1 GPU-hour、20steps、256train/64val | 若运行才允许run.json + metrics.jsonl + log；0模型 | G1已在预算内完成，未用profile冒充训练或效果证据 |
| G1-DATA | 三机制case、posterior与固定π gold | SUCCEEDED | CPU环境；不依赖GPU | 360根case；72dev/288audit；同π CPU-MC | data/audit/cases.jsonl与manifest.json | 已验证三机制各120、每层40、dev/audit每单元8/32、full子集72；完成且不重跑 |
| G1-DEV | 解析/格式/吞吐、最多一次提示修订 | SUCCEEDED | 648/648调用、100%格式 | 实用dev24+dev72两个受控阶段 | 同一run记录+predictions+log | 无提示修订；不重复dev |
| G1-FREEZE | 冻结数据、提示、指标及72-case对照子集 | SUCCEEDED | G1-DEV 100%格式 | CPU | 更新同一manifest/status | `audit_frozen=true`、`audit_started=true`；audit完成后仍未修改科学字段 |
| G1-H | 历史教师audit | SUCCEEDED | G1-FREEZE | 288次短调用；物理GPU 0 | 单一run记录+predictions+log | 未提供隐藏信息；288/288有效 |
| G1-P | 隐状态教师audit | SUCCEEDED | G1-FREEZE | 768次短调用；物理GPU 0 | 同上 | 各世界评分后按真实posterior聚合；768/768有效 |
| G1-PR | 隐状态+参考动作audit | SUCCEEDED | G1-FREEZE | 768次短调用；物理GPU 0 | 同上 | 保留参考评分语义；768/768有效 |
| G1-PRIC | 增加信息可实施约束audit | SUCCEEDED | G1-FREEZE | 768次短调用；物理GPU 0 | 同上 | 主对比PR→PR-IC；768/768有效 |
| G1-FULLREF | 参考完备性对照 | SUCCEEDED | G1-FREEZE | 72case，共384次短调用 | 同上 | 未借用history gold；两条件各192/192有效 |
| G1-MC | 相同history-only π的MC | SUCCEEDED | G1-DATA | CPU 276,480嵌套rollout | data/audit/cases.jsonl与manifest.json | 每case/动作256次；不增加独立n；未使用可选LLM-MC |
| G1-ANALYZE | 后验聚合、case bootstrap与gate | SUCCEEDED / NO-GO | G1各固定条件完整 | 2000次bootstrap，seed23 | 单一汇总写run.json；status写结论 | PR→PR-IC无正向证据；本轮结束 |
| G2-MATRIX | 4标签×3seed偏好学习 | NOT_STARTED（NO-GO；后续范围） | 需明确新决策、预算和训练任务 | 最多12正式训练run；预算未开放 | 每run至多1个final adapter | 不自动启动 |
| G3-G4 | PRM/在线RL/独立工具环境 | PLANNED（后续范围） | G2及明确新预算 | 未开放 | 未开放 | 本次不提交 |

## D. 预算 阻塞与阶段决策

### 当前账目

| 项目 | 协议上限 | 当前已登记使用/预留 | 执行与核验说明 |
|---|---:|---:|---|
| G0/G1 GPU时间 | 36 GPU-hours | 已使用约0.1564；运行剩余0；排队预留0 | 两smoke约0.0022+教师三阶段约0.1542；无训练profile |
| 并发GPU任务 | 2 | 当前0；已完成smoke 2个和教师attempt 1个 | 物理0和6当前均无本项目进程；教师仅用物理0，不跨卡加载 |
| 真正QUEUED任务 | 4 | 当前主机本项目0 | 当前主机无Slurm命令且无本项目运行进程；不声称其他服务器全局队列为0 |
| 教师调用 | 4272 | 已执行3,624；预留0；剩余648 | 全attempt按固定调用键计数；3,624/3,624有效，无失败补采样 |
| 可选LLM-MC生成 | 2592 | 0 | 同π CPU-MC另记，不占GPU |
| G0/G1模型保存 | 0 | 0 | profile不得传save-final-adapter |
| G0/G1新实验产物 | 2 GiB | 5,370,928 bytes | data/audit 1,156,378 + runs 2,284,470 + logs 1,930,080；核验时/data4可用5,611,134,976 bytes；模型缓存另计 |
| 单run非模型产物 / 单日志 | 128 MiB / 10 MiB | 最大run 2,284,470 bytes；最大log 1,921,980 bytes | 均在限制内；未产生逐样本文件或镜像汇总 |
| 新合成数据 | 128 MiB | 1,156,378 bytes | 仅cases.jsonl与manifest.json；不按模型/种子复制 |
| 新增模型缓存 | 24 GiB、1个底模revision | 已使用8,060,942,868 bytes；预留0 | ModelScope `Qwen/Qwen3-4B-Instruct-2507` commit `2de2439...3cbb`；唯一snapshot；官方hash匹配 |
| 后续final adapter | 每run≤512 MiB，正式矩阵≤12个 | 0 | 尚不在本轮范围 |

### 阻塞与决策

| 项目 | 当前结论 | 解决条件 |
|---|---|---|
| GPU资源 | READY_WITH_LIMITATION | 用户已授权物理GPU 0和6，两卡BF16 smoke成功；NVML driver/library mismatch仍使用量查询不可用，后续作业仅依赖PyTorch自身记录 |
| 三机制audit入口 | COMPLETE | audit_cases.py和audit_teacher.py已实现；dev216/648与audit2976调用全部完成，格式100% |
| 模型revision | READY | ModelScope唯一snapshot `2de2439ea21be1dc5cb21f22f88af07e43393cbb`；官方SHA-256/尺寸匹配；后续全部local-files-only |
| 保存策略GPU验证 | G0/G1零模型保存已核验 | profile未运行；正式adapter回读超出本轮范围，后续若明确启动再验证 |
| 初始研究gate | NO-GO / 机制证据不足 | PR→PR-IC标准化regret差=-0.001341，95% CI [-0.002671, 0.000041]；full-reference残余=-0.000331，未支持正向效应 |
| 磁盘强制quota | 未核验 | 如只有目录检查，明确标应用层限制，不称OS硬限额 |
| 续训能力 | resume_supported=false | 本方案不保存完整checkpoint；失败只能受控新attempt |
| 下一阶段 | NOT_STARTED | G1已交付且gate为no-go；若继续须显式新决策与新预算，不自动扩展 |

### 最近维护

- 2026-10-28：补齐执行协议、status初始表与文件/计算预算；训练入口默认0模型保存。
- 2026-10-28 14:46 +08:00：在klclab-X660-G45完成G0只读环境核验；无Slurm，NVML失败，GPU归属与目标模型缓存未解决；未启动GPU作业。
- 2026-10-28 14:48 +08:00：用户授权物理GPU 0和6；两个BF16 smoke均rc=0，未加载或保存模型。
- 2026-10-28 14:55 +08:00：生成并验证360个三机制根case、真实posterior精确Qπ与同π CPU-MC；未执行教师调用。
- 2026-10-28 15:28 +08:00：ModelScope唯一Qwen3-4B-Instruct-2507 snapshot下载与官方hash校验完成；未执行教师调用或训练。
- 2026-10-28 15:34 +08:00：dev24后补足72 dev；累计648/648格式有效，无提示修订；数据、提示、指标、gate和模型版本冻结。
- 2026-10-28 15:44 +08:00：固定audit完成；2,976/2,976格式有效，attempt累计3,624次；0模型保存。
- 2026-10-28 19:41 +08:00：完成2000次根case分层bootstrap、预算/白名单/进程复核；gate为机制证据不足/no-go，无运行或排队任务。
- 既有2026-10-27 CPU结果保留；本次新增本地教师推理，但没有GPU训练、训练profile、模型保存或G2提交。

### 协调者更新规则

提交前写唯一attempt、实际job-name/comment标签、完整命令、依赖、代码/数据/model版本、计算及磁盘预留、输出路径和SUBMITTING；真实回执后再查队列。Slurm查当前squeue与结束后的sacct；无调度器验证主机/PID/启动时间。提交结果不确定时先核实，不能重交。
作业结束先VERIFYING，校验退出码、预期步数、结果和文件白名单后SUCCEEDED；失败保留记录。断线不刷新last_verified_at，另记当前观察错误。
一个协调者串行修改本文件；用同目录.status.md.tmp写完后原子替换。任务不直接写status；不创建状态副本或后台daemon。

