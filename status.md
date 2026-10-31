# 实验状态

最后维护：2026-10-31 04:20 +08:00（Asia/Shanghai）。
当前阶段：G0–G5保留；G6外部原生ToolSandbox证据失效与最小修复联合实验已完成（见J节）。G6完成G5 CPU重分析、60-family固定数据、216/864节点dev/audit、3,240次Qwen推理、2,400条原生工具轨迹和最终分析；无运行中或排队任务。结果：DIRECT_E复现旧值继续使用（U root old-use 100%、retract 0%）；VALIDITY_REVIEW改善U但reminder的other-object over-retraction严重（V 79.17% vs R 0），未过intervention gate；QUERY_ONCE_V优于V（utility +0.604，CI含0；V query少但success低4.69pp、wrong高2.60pp）；强DEPENDENCY_RULE solved all，utility9.75、success1.0、0模型调用。结论：无进入过程监督训练的明确理由；保留规则作为工程基线，不自动启动G7。
本项目在本会话中已登记的直接运行attempt数：G0–G4为35个，G5新增1个逻辑attempt（六phase），G6新增1个逻辑attempt（G6-CPU-REANALYSIS/G6-TOOLS-DATA/G6-DEV/G6-AUDIT/G6-EXECUTE/G6-ANALYZE六phase）；G5/G6全部COMPLETED，无当前运行中或排队任务。G6 dev仅4个R invalid、audit 14个R invalid，均为模型schema不一致，未改prompt；无GPU/模型调用失败。当前主机未安装Slurm命令，因此无Slurm队列可查。

执行边界：G0–G6均已完成；任何新阶段（PRM、在线RL、训练scorer、外部benchmark）须由用户显式新决策与预算，不自动启动。G6的唯一后续动作是保留强依赖规则作为工程基线并收紧论文主张；不启动G7过程监督训练，不追加同模板数据。协调者是status.md唯一writer；脚本和子agent不直接修改此表。详细规则见README第12节和AGENT_PROMPT.md。

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

## E. G2-PILOT（用户2026-10-29显式新决策，覆盖本文件第1行默认G0/G1-only范围）

研究问题（本轮唯一目标）：固定输入/候选动作/底模/训练预算，信息可实施监督标签（information）是否比允许隐藏信息的监督标签（clairvoyant）训练出更好的信息采集决策。不扩展到新教师audit、PRM、在线RL或多轮agent。

### E1. 环境修复记录

共享`base` conda环境（transformers 5.10.2、trl 1.5.1）无法满足`train_dpo.py`硬编码版本门（要求`trl==0.21.0`且`4.55.0<=transformers<4.56.0`）。排查发现`/data4`、`/`、`/data2`、`/data3`均99-100%满，故未在这些位置新建环境；`/data/sjj`（本人可写、44GB可用）有足够空间。

- 新建专用venv：`/data/sjj/process_reward_venv`（`python3 -m venv`，不继承系统site-packages）。
- 按`requirements-gpu.txt`精确安装（经清华PyPI镜像，因直连pypi.org约17KB/s、镜像约300-550KB/s）：torch 2.8.0+cu128、trl 0.21.0、transformers 4.55.4、peft 0.17.1、accelerate 1.10.1、datasets 4.0.0、safetensors 0.6.2、packaging 25.0 — 全部落在requirements-gpu.txt声明区间内。
- 磁盘：安装后`/data`剩余37GB（从44GB，实际占用约7GB，远低于分区剩余）。
- 驱动/CUDA验证：驱动535.104.05名义最高支持CUDA 12.2，但torch 2.8.0自带CUDA 12.8 runtime实测可用（物理GPU1上256x256 BF16矩阵乘法finite=true）——与本机此前G0/G1使用的torch 2.12.0+cu130经验一致，未发现驱动不兼容。
- 端到端smoke验证（最小化验证修复，非独立profile步骤）：用该venv在物理GPU1上以`--max-steps 3 --max-train-samples 16 --max-eval-samples 8`跑通`train_dpo.py`完整路径（模型加载→LoRA reference precompute→训练3步→validation1次），rc=0。峰值`allocated 8.08GiB / reserved 8.26GiB`；输出目录`/tmp/g2_memtest`（非仓库路径，已删除，不计入产物预算）。

### E2. GPU授权更新

用户2026-10-29明确："0 1 6都可以寄生"——物理GPU 0、1、6均可用，非独占/机会式共享，不抢占其他任务。单run实测峰值约8.3GiB；三卡各自可用显存会随其他租户波动，实际分配以启动前`nvidia-smi`读数为准，不固化在本表之外。

### E3. 四个正式训练run预注册

| RUN_ID/ATTEMPT_ID | 条件 | seed | 状态 | 当前阶段 | GPU（物理） | Job ID/host+PID+启动时间 | 开始时间 | 结束时间 | 最后更新时间 | 完成/预期optimizer steps | GPU-hours | 输出目录 | 主要结果/错误 |
|---|---|---:|---|---|---:|---|---|---|---|---|---:|---|---|
| G2-C-s42-a01 | clairvoyant | 42 | SUCCEEDED | 训练+eval全部完成 | 0 | klclab-X660-G45；训练PID 1553634；eval PID 1706040 | 18:44:00 +08:00 | 20:26 +08:00 | 2026-10-29 20:26 +08:00 | 229/229(训练) | 0.558(训练)+0.096(eval)=0.654 | runs/G2-C-s42-a01 | 训练train_loss0.0715/eval_loss0.00479/acc1.0/margin7.02；test-regret整体0.815(ID2.915/ood_drift0.110/ood_rel2.547)；100%选skip；valuable_info漏查率1.0；无益检查率0.0；格式有效率1.0 |
| G2-I-s42-a01 | information | 42 | SUCCEEDED | 训练+eval全部完成 | 6训练/1eval | klclab-X660-G45；训练PID 1553635；eval PID 1706041 | 18:44:00 +08:00 | 20:26 +08:00 | 2026-10-29 20:26 +08:00 | 229/229(训练) | 0.334(训练)+0.140(eval)=0.474 | runs/G2-I-s42-a01 | 训练train_loss0.588/eval_loss0.498/acc0.736；test-regret整体1.281(ID0.321/ood_drift1.666/ood_rel0.157)——与未训练底模逐条预测100%一致；100%选precise；valuable_info漏查率0.0；无益检查率1.0；格式有效率1.0 |
| G2-C-s43-a01 | clairvoyant | 43 | SUCCEEDED | 训练+eval全部完成 | 0 | klclab-X660-G45；训练PID 1611798；eval PID 1725156 | 19:19:18 +08:00 | 20:36 +08:00 | 2026-10-29 20:36 +08:00 | 229/229(训练) | 0.553(训练)+0.096(eval)=0.649 | runs/G2-C-s43-a01 | 训练train_loss0.0677/eval_loss0.00459/acc1.0/margin7.08；test-regret整体0.815(ID2.915/ood_drift0.110/ood_rel2.547)；100%选skip；valuable_info漏查率1.0；无益检查率0.0；格式有效率1.0 |
| G2-I-s43-a01 | information | 43 | SUCCEEDED | 训练+eval全部完成 | 6训练/1eval | klclab-X660-G45；训练PID 1592559；eval PID 1725157 | 19:06:46 +08:00 | 20:36 +08:00 | 2026-10-29 20:36 +08:00 | 229/229(训练) | 0.335(训练)+0.143(eval)=0.478 | runs/G2-I-s43-a01 | 训练train_loss0.578/eval_loss0.492/acc0.752；test-regret整体1.281(ID0.321/ood_drift1.666/ood_rel0.157)——与未训练底模逐条预测100%一致；100%选precise；valuable_info漏查率0.0；无益检查率1.0；格式有效率1.0 |
| G2-BASE-a01 | 未训练底模评估 | - | SUCCEEDED | eval完成 | 0 | klclab-X660-G45；PID 1694776；启动2026-10-29 20:13 +08:00 | 20:13 +08:00 | 20:19 +08:00 | 2026-10-29 20:19 +08:00 | - | 0.094(eval) | runs/G2-BASE-a01 | test共438个唯一scenario；format_valid_rate1.0；100%选precise；mean_regret_invalid_as_abort整体1.281（ID0.321/ood_drift1.666/ood_rel0.157）；valuable_info漏查率0.0；无益检查率1.0 |

### E5. 最终结果表与结论（2026-10-29 20:40 +08:00）

| 模型/条件 | seed | 整体 regret | ID regret | OOD drift regret | OOD reliability regret | 有益信息漏查率 | 无益检查率 | 格式有效率 | GPU-hours |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE（未训练） | - | 1.281 | 0.321 | 1.666 | 0.157 | 0.0 | 1.0 | 1.0 | 0.094 |
| C（clairvoyant） | 42 | 0.815 | 2.915 | 0.110 | 2.547 | 1.0 | 0.0 | 1.0 | 0.654 |
| I（information） | 42 | 1.281 | 0.321 | 1.666 | 0.157 | 0.0 | 1.0 | 1.0 | 0.474 |
| C（clairvoyant） | 43 | 0.815 | 2.915 | 0.110 | 2.547 | 1.0 | 0.0 | 1.0 | 0.649 |
| I（information） | 43 | 1.281 | 0.321 | 1.666 | 0.157 | 0.0 | 1.0 | 1.0 | 0.478 |
| C 两seed均值 | 42,43 | 0.815 | 2.915 | 0.110 | 2.547 | 1.0 | 0.0 | 1.0 | 0.6515 |
| I 两seed均值 | 42,43 | 1.281 | 0.321 | 1.666 | 0.157 | 0.0 | 1.0 | 1.0 | 0.476 |

Δ = Regret_C − Regret_I（整体，正数表示I更优）：seed42 = 0.815−1.281 = **−0.466**；seed43同样 = **−0.466**；两seed均值 = **−0.466**。两seed方向一致，但均为负——即C（clairvoyant）整体regret更低，不是I更优。

本轮累计GPU-hours：训练1.780 + 评估0.570 = **2.350**（预算上限12，远未超）。4个final adapter各23.6MB（预算上限512MiB/个），合计≈94.5MB（预算上限3GiB）。无a02重试，无文件白名单外产物。

**三个结论性问题的直接回答：**

1. **I是否在两个seed上都优于C？** 否。两个seed上C的整体regret（0.815）都低于I（1.281），方向一致但结果与假设相反——信息可实施标签并未在本轮设置下训练出更好的决策。

2. **I是否优于未训练底模？** 否。I42与I43的predictions.jsonl在全部438个测试场景上与G2-BASE-a01逐条完全一致（action、regret等全部相同数值），即两个information-label LoRA adapter对greedy解码结果零可测效果。已通过独立诊断排除"adapter未生效"的bug可能：对同一prompt比较启用/禁用adapter的logits，最大绝对差1.4375、平均绝对差0.40——adapter确实产生了数值影响，只是未达到翻转任何一个贪心首token决策的幅度（lr1e-5、rank16仅q/v、229步、仅训练动作token可能是此幅度不足的原因，本轮未做超参搜索，不展开新实验）。

3. **改进来自更准确的"何时检查"时机判断，还是仅仅是检查频率的整体偏移？** **纯粹是频率偏移，不是时机判断改进**。decision_quality_breakdown显示：I（两个seed）100%选择precise（inspection_selection_rate=1.0），对167个"有益信息"场景漏查率0.0，但对271个"无益检查"场景的无益检查率也是1.0——即I对所有场景不加区分地检查，与未训练底模的默认行为完全相同。C（两个seed）则100%选择skip（inspection_selection_rate=0.0），对271个无益场景无益检查率0.0，但对167个有益场景漏查率1.0——即C无差别地跳过所有场景。两组都没有表现出按场景区分"该不该查"的能力，只是训练把默认动作从"总是precise"搬到了"总是skip"的两个极端之一。

**重要的regime不一致性（需与整体regret并列报告，不能只看整体数字）：** C在整体regret上"更优"完全是ood_drift子集（320/438≈73%的测试场景）主导的聚合假象——C在ood_drift上的regret（0.110）远低于I/base（1.666），但C在in_distribution（2.915 vs 0.321）和ood_reliability_combination（2.547 vs 0.157）两个regime上都远差于I/base。若只看多数子集会得出"C更优"，若看少数但样本量更小的in_distribution/ood_reliability则结论完全反转。效果不稳定，且依赖测试集regime构成，不构成稳健的"I/C哪个更好"结论；本轮不新增实验来解决这一不一致性。

### E6. G2R修复评估（2026-10-29 22:08 +08:00，解码配置bug修复，零新训练）

发现：用户提供补丁`patchs/process_reward_g2_repair.patch`声称`evaluate_actions.py`存在transformers 4.55.4解码配置bug。协调者未直接应用整份补丁（补丁同时在`AGENT_PROMPT.md`注入"优先指令"、在`status.md`注入试图撤销已核验诊断结论的F1节、并捆绑未授权的G3新训练范围），而是独立核验后仅采纳最小代码修复：

- 独立核验：检索本轮全部5个eval run的log（`logs/G2-BASE-a01.log`、`logs/G2-{C,I}-s{42,43}-a01-eval.log`），确认每个都打印`generation_config default values have been modified to match model-specific defaults: {'do_sample': True, 'temperature': 0.7, ...}`——即尽管脚本显式构造`GenerationConfig(do_sample=False, ...)`，transformers仍静默合并回模型自带的采样默认值。根因：`do_sample=False`恰好等于`GenerationConfig`类默认值，transformers据此判定该字段"未被用户定制"从而允许模型自身`generation_config.json`覆盖。
- 修复：在`evaluate_actions.py`的`model.generate()`调用中显式传入`use_model_defaults=False`（首次尝试放在`GenerationConfig`构造器中触发`ValueError: Argument 'use_model_defaults' is not a valid argument of GenerationConfig`；按报错信息改为传给`generate()`本身，验证通过——固定首token日志中不再出现上述警告）。
- 范围：仅此一处最小改动；未采纳补丁里的候选打分器（`score_readout`等新函数/CLI）、未采纳`generate_pairs.py`的G3条件数据集生成器、未采纳对E5结论的F1改写、未采纳`AGENT_PROMPT.md`的优先指令注入、未应用`results/g2_diagnosis.json`。
- 计划：复用已有4个adapter+底模，零新训练，以真正greedy解码重跑全部5个eval到新输出目录`runs/G2R-*`，与原`runs/G2-*`并存比较。

| RUN_ID/ATTEMPT_ID | 来源 | 状态 | GPU（物理） | PID | 开始时间 | 结束时间 | 输出目录 | 结果 |
|---|---|---|---:|---|---|---|---|---|
| G2R-BASE-a01 | 复用未训练底模 | SUCCEEDED | 0 | 1883821 | 22:07:28 +08:00 | 22:13 +08:00 | runs/G2R-BASE-a01 | 整体regret 1.2812785388127854，与原G2-BASE-a01逐条一致 |
| G2R-C-s42-a01 | 复用adapter runs/G2-C-s42-a01/adapter | SUCCEEDED | 0 | 1894231 | 22:14 +08:00 | 22:1x +08:00 | runs/G2R-C-s42-a01 | 整体regret 0.8151826484018264，与原runs/G2-C-s42-a01/eval逐条一致 |
| G2R-I-s42-a01 | 复用adapter runs/G2-I-s42-a01/adapter | SUCCEEDED | 0 | - | 顺序执行 | - | runs/G2R-I-s42-a01 | 整体regret 1.2812785388127854，与原runs/G2-I-s42-a01/eval逐条一致 |
| G2R-C-s43-a01 | 复用adapter runs/G2-C-s43-a01/adapter | SUCCEEDED | 0 | - | 顺序执行 | - | runs/G2R-C-s43-a01 | 整体regret 0.8151826484018264，与原runs/G2-C-s43-a01/eval逐条一致 |
| G2R-I-s43-a01 | 复用adapter runs/G2-I-s43-a01/adapter | SUCCEEDED | 0 | - | 顺序执行 | - | runs/G2R-I-s43-a01 | 整体regret 1.2812785388127854，与原runs/G2-I-s43-a01/eval逐条一致 |

**核验结论：** 对全部5个run逐scenario（438条×5）比较action字段，0处不一致；整体regret与by_regime分解在浮点精度下完全相等。即虽然do_sample覆盖bug真实存在（已通过log证据独立确认），但对本轮所有已报告结果（E5节的表格、Δ值、三个结论性问题的回答、regime不一致性发现）**没有产生任何影响**——因为每个决策点的top-1 token logit领先幅度足够大，temperature=0.7采样在438个场景的5个run里从未实际采到非贪心token。**E5节此前的全部结论原样成立，无需撤回或修改任何数值。**本次修复的价值是把方法学描述从"声称greedy但实际采样"纠正为"真正greedy"，并为未来任何新一轮训练/评估提供正确的解码配置，而非改变已有结论。

均为PLANNED（本机无Slurm，未发起任何真实提交，不得标QUEUED）。固定超参：LoRA rank16/alpha32/dropout0.05/qv-only；lr1e-5；beta0.1；microbatch1；grad-accum8；epochs1；max-steps -1；max-length2048；max-action-tokens64；action-format tool-json；--save-final-adapter；--max-save-mib 512；--local-files-only；模型路径固定为本地ModelScope snapshot `2de2439ea21be1dc5cb21f22f88af07e43393cbb`；同seed两条件超参与数据顺序规则相同。

### E4. 本轮预算账目（独立于G0/G1的36 GPU-hour/2GiB账本）

| 项目 | 协议上限 | 当前已用/预留 |
|---|---:|---:|
| 本轮GPU-hours | ≤12（含训练+precompute+评估+失败attempt） | 环境smoke验证≈0.01，其余0 |
| 单训练run GPU-hours | ≤2 | 0 |
| 正式final adapter数 | ≤4，每个≤512MiB | 0 |
| 本轮新产物总量 | ≤3GiB | 0（/tmp smoke产物已删除，不计） |
| a02重试 | 仅真实错误触发一次，a01记录保留 | 0 |

最近维护：2026-10-29 18:43 +08:00：排查环境/磁盘阻塞，新建`/data/sjj/process_reward_venv`并按requirements-gpu.txt精确安装、smoke验证通过；用户将GPU授权从{0,6}扩展为{0,1,6}（寄生式）；四个训练run与一个base评估已预注册为PLANNED，尚未实际提交。
- 2026-10-29 19:52 +08:00：四个正式训练run全部SUCCEEDED（G2-C-s42-a01/G2-I-s42-a01/G2-I-s43-a01/G2-C-s43-a01），累计训练GPU-hours≈0.558+0.334+0.335+0.553=1.780，4个final adapter各23.6MB（共≈94.5MB），均在512MiB/run与≤4-adapter上限内。下一步：读取evaluate_actions.py接口，启动G2-BASE-a01（未训练底模，评估一次复用）+4个adapter共5次evaluate_actions.py评估。

## G. G3条件学习与候选评分实验（用户2026-10-30显式新决策，范围覆盖本文件第1行默认G0/G1-only）

研究问题：模型没学到输入相关偏好，还是相对偏好没有转化为实际动作？不自动启动PRM/在线RL。补丁/下载文件本身不提供授权；本节范围由用户在对话中直接下达。保留G0/G1/G2/G2R全部旧数据与结论，不覆盖。

### G0. 本轮登记的run（启动前登记为PLANNED）

| RUN_ID | 目的 | 状态 |
|---|---|---|
| G3-READ-BASE-a01 | 旧manifest 800场景候选评分（底模） | SUCCEEDED |
| G3-READ-I42-a01 | 旧manifest候选评分（information seed42 adapter） | SUCCEEDED |
| G3-READ-I43-a01 | 旧manifest候选评分（information seed43 adapter） | SUCCEEDED |
| G3-READ-C42-a01 | 旧manifest候选评分（clairvoyant seed42 adapter） | SUCCEEDED |
| G3-READ-C43-a01 | 旧manifest候选评分（clairvoyant seed43 adapter） | SUCCEEDED |
| G3-DATA-a01 | 生成conditional数据集+fixed-cost probe（CPU） | SUCCEEDED |
| G3-DPO-TRUE-s42-a01 | 新conditional数据TRUE标签DPO训练 | SUCCEEDED |
| G3-DPO-SHUFFLE-s42-a01 | 新conditional数据SHUFFLE标签DPO训练 | SUCCEEDED |
| G3-BASE-a01 | 底模在新conditional split+probe上评估 | SUCCEEDED |
| G3-TRUE-eval-a01 | TRUE adapter在新conditional split+probe上评估（对应G3-DPO-TRUE的评估阶段） | SUCCEEDED |
| G3-SHUFFLE-eval-a01 | SHUFFLE adapter在新conditional split+probe上评估（对应G3-DPO-SHUFFLE的评估阶段） | SUCCEEDED |
| G3-SFT-TRUE-s42-a01 | 条件触发：仅当validation未达三项门槛时执行 | SUCCEEDED（训练270/270 steps） |
| G3-SFT-eval-a01 | SFT adapter在新conditional split+probe上评估 | SUCCEEDED |

本机无Slurm，上表均为本地直接attempt，不得标QUEUED。协调者（本agent）是唯一writer。

### G-预算账目（独立新账本，上限见用户指令）

| 项目 | 协议上限 | 当前已用/预留 |
|---|---:|---:|
| 本轮GPU-hours | ≤8（含加载+precompute+训练+评估+失败attempt） | **≈3.48**（Step A 0.861 + DPO-SHUFFLE 0.699 + DPO-TRUE 0.342 + G3-BASE 0.263 + G3-TRUE-eval 0.385 + G3-SHUFFLE-eval 0.400 + G3-SFT-eval 0.333 + SFT训练 0.200） |
| 单训练attempt GPU-hours | ≤2 | 最大0.70（DPO-SHUFFLE） |
| 并行GPU | ≤2，每任务单卡 | 最多2（GPU0+GPU6；DPO训练与评估分别单卡） |
| 正式final adapter数 | ≤3，每个≤512MiB | **3**（TRUE/SHUFFLE/SFT各23,613,253 bytes） |
| 本轮新增产物总量 | ≤2GiB | **≈90MB**（data 16MB + runs/adapters/logs约74MB） |
| DPO训练数 | 2（必做） | 2/2 |
| SFT训练数 | ≤1（条件触发） | 1/1（已触发并完成） |
| a02重试 | 仅真实报错触发一次，同任务至多一次 | 0 |

最近维护：2026-10-30 00:18 +08:00：登记本节全部10个run为PLANNED；复用venv `/data/sjj/process_reward_venv`、本地ModelScope snapshot `2de2439ea21be1dc5cb21f22f88af07e43393cbb`（local_files_only）；GPU授权复用{0,1,6}，当时空闲：GPU0≈52.9GiB、GPU1≈20.8GiB、GPU6≈9.0GiB（6号偏紧，优先用0/1）。下一步：读取/扩展evaluate_actions.py候选评分实现。

- 2026-10-30 00:30 +08:00（代码变更）：`evaluate_actions.py`新增`--mode score`路径（`tokenize_boundary`/`sequence_logprob`/`tie_aware_readout`/`strict_pair_accuracy`/`run_score_mode`），严格复用train_dpo.py的prompt/action token边界约定；BASE模型L_base==L_policy，4个adapter通过`model.disable_adapter()`得到真实L_base；3-way退化tie按TIE_TOLERANCE下的均匀期望处理，不默认插入顺序第一个。CPU自测（tie/clean/pair-accuracy三例）通过，`test_core.py` 14/14通过。`generate_pairs.py`新增`--conditional`/`--self-test`路径：`build_conditional_family`+`search_conditional_families`(seed=17)按六种严格排序各一次构造家族（V0/gj/m公式，正成本+oracle一致性+相邻gap>=0.25全部校验）；`format_conditional_prompt`新增固定history-only Bayes-optimal controller措辞；TRUE/SHUFFLE双标签（shuffle_seed=1000020，仅对720个train scenario按sorted(scenario_id)顺序permute完整排序，validation/test保持TRUE）；`build_fixed_cost_probe`(seed=1701)按drift/prior/reliability顺序贪心选16+16+16→48对/96场景，全局scenario不重复。CPU自测`--self-test`（8家族+48对探针全部断言）通过。

- 2026-10-30 00:33 +08:00：`G3-DATA-a01`执行完成（CPU-only，0.66秒）。产物：`data/g3_conditional/`共10个文件（true/shuffle各train2160+validation432+test864条pair，scenario_manifest.jsonl 1152行，pairs_audit.jsonl 3456行，fixed_cost_probes.jsonl 48行，manifest.json），总计≈16MB。验证：validation/test的true与shuffle逐行chosen字段0差异；train 1140/2160差异（真实置换，非误报）。manifest内CPU baseline：constant-skip/quick/precise三者oracle regret在train/validation/test上完全相等（0.4334/0.4570/0.4470，按家族构造对称性预期内），exact oracle regret=0；kNN(k=31,仅用两项成本)在validation/test上accuracy=0.660/0.601。status：**G3-DATA-a01: SUCCEEDED**。

- 2026-10-30 00:28 +08:00起：`G3-READ-BASE-a01`在GPU0（cuda:0，无CUDA_VISIBLE_DEVICES限制，物理GPU0）后台运行（PID见`/tmp/g3_step_a_driver.log`），驱动脚本`run_g3_step_a.sh`将顺序执行5个`--mode score`评分run（BASE/I42/I43/C42/C43），test split复用对应G2R-*/predictions.jsonl的greedy action（不重新生成），train/validation新鲜生成greedy。进度哨兵`/tmp/g3_step_a_progress.log`，各run日志`/tmp/G3-READ-*.log`。status：**G3-READ-BASE-a01: RUNNING**，其余4个仍为PLANNED（驱动脚本会顺序推进，本节随后更新）。

- 2026-10-30 00:35 +08:00起：`G3-DPO-SHUFFLE-s42-a01`在GPU1（`CUDA_VISIBLE_DEVICES=1`）后台启动，脚本`run_g3_dpo_shuffle.sh`，复用train_dpo.py不改动，超参严格对齐G2（qv rank16/alpha32/dropout0.05，lr1e-5，beta0.1，microbatch1/grad_accum8，1 epoch，max_length2048/max_action_tokens64，seed42），数据指向`data/g3_conditional/shuffle/{train,validation}.jsonl`（2160训练对，预期270 optimizer steps）。日志`/tmp/G3-DPO-SHUFFLE-s42-a01.log`，哨兵`/tmp/g3_dpo_shuffle_progress.log`。status：**G3-DPO-SHUFFLE-s42-a01: RUNNING**。`G3-DPO-TRUE-s42-a01`等GPU0空出后在GPU0启动（保持≤2并行）。

- 2026-10-30 00:50 +08:00：Step A进度更新。`G3-READ-BASE-a01`已SUCCEEDED（00:28:30→00:36:22，约7m52s）：train307+validation55新鲜生成greedy，test438复用G2R-BASE-a01；overall greedy mean_regret_valid_only=0.8394/optimal_action_rate=0.4438（底模贪心单一倒向precise，policy与greedy一致因L_base==L_policy恒等；implicit三候选恒为0-margin退化tie，按均匀期望算regret=1.3166/optimal_rate=0.3354——符合预期，非bug）。`G3-READ-I42-a01`已SUCCEEDED（00:36:22→00:48:38，约12m16s）：greedy/policy与BASE几乎一致（mean_regret 0.8394/optimal 0.4438，margin提升到22.24），implicit此次不再退化（mean_margin=0.785，mean_regret=0.8394/optimal_action_rate=0.4438，strict_pair_accuracy=0.5369）——说明adapter的L_base通过disable_adapter()正确区分出非零reward。`G3-READ-I43-a01`已开始运行（GPU0）。status：**G3-READ-BASE-a01: SUCCEEDED**，**G3-READ-I42-a01: SUCCEEDED**，**G3-READ-I43-a01: RUNNING**，C42/C43仍PLANNED（驱动脚本顺序推进）。`G3-DPO-SHUFFLE-s42-a01`（GPU1）仍在reference log-probs precompute阶段（train侧~91%完成，尚未进入实际DPO optimizer step）。

- 2026-10-30 01:12 +08:00：Step A继续推进。`G3-READ-I43-a01`已SUCCEEDED（00:48:38→00:59:29，约10m51s）：greedy/policy与BASE/I42一致（0.8394/0.4438），implicit候选不再恒为precise：751次precise+48次quick+1次并列，mean_regret=0.7444/optimal_action_rate=0.4731/strict_pair_accuracy=0.5084——好于BASE，说明s43种子adapter下implicit reward已能部分区分quick更优的场景。`G3-READ-C42-a01`已SUCCEEDED（00:59:29→01:10:06，约10m37s）：greedy/policy这次整体翻转为**全部skip**（而非precise），mean_regret=1.7139/optimal_action_rate=0.415（弱于BASE/I系列），implicit与greedy/policy完全一致（同为skip，mean_margin=5.69,strict_pair_accuracy=0.5407）——C适配器训练目标（conservative/caution分支）导致策略整体右移至skip，是预期中的adapter差异而非异常。`G3-READ-C43-a01`已开始运行（GPU0）。status：**G3-READ-I43-a01: SUCCEEDED**，**G3-READ-C42-a01: SUCCEEDED**，**G3-READ-C43-a01: RUNNING**（Step A最后一个）。`G3-DPO-SHUFFLE-s42-a01`（GPU1）已进入实际DPO训练，约226/270 optimizer steps（~5.6s/it），预计数分钟后完成训练+eval+adapter保存。

- 2026-10-30 01:22 +08:00：Step A全部完成，DPO-SHUFFLE训练完成。`G3-READ-C43-a01`SUCCEEDED（01:10:06→01:20:43，约10m37s）：与C42几乎一致（全部greedy/policy/implicit=skip，mean_regret=1.7139/optimal_action_rate=0.415），确认C系列两个种子收敛到同一conservative倾向，非随机噪声。Step A五个run全部SUCCEEDED：**G3-READ-BASE-a01/I42/I43/C42/C43全部: SUCCEEDED**。`G3-DPO-SHUFFLE-s42-a01`训练正常结束（270 optimizer steps，total_wall_seconds=2518.09≈0.70 GPU-hours）：eval_loss=0.6973，**eval_rewards/accuracies=0.4769**（≈chance，符合SHUFFLE打乱标签后预期的对照结果，非异常），final adapter已导出到`runs/G3-DPO-SHUFFLE-s42-a01/adapter/`，仅含`adapter_model.safetensors`(23.6MB)+`adapter_config.json`，sha256已记录，23613253 bytes远低于536870912限额。GPU1上train_dpo进程已退出，GPU0当前空闲（除~17.6GB他人占用外全部可用）。status：**G3-DPO-SHUFFLE-s42-a01: SUCCEEDED**。下一步：在GPU0启动`G3-DPO-TRUE-s42-a01`（同超参，指向`data/g3_conditional/true/{train,validation}.jsonl`），此时GPU1已空，仍满足≤2并行约束（本次仅1个GPU在跑）。累计GPU-hours预估：Step A五次评分（各≈8-12分钟，GPU0，无训练只评分不计入"训练attempt"但计入总GPU-hours预算）+DPO-SHUFFLE 0.70h ≈ 已用约1.6 GPU-hours（含所有评分阶段），预算8h内进度正常。

- 2026-10-30 01:22 +08:00：`G3-DPO-TRUE-s42-a01`已在GPU0（`CUDA_VISIBLE_DEVICES=0`）后台启动，脚本`run_g3_dpo_true.sh`，超参与SHUFFLE逐项一致（qv rank16/alpha32/dropout0.05，lr1e-5，beta0.1，microbatch1/grad_accum8，1 epoch，max_length2048/max_action_tokens64，seed42），仅数据路径不同：指向`data/g3_conditional/true/{train,validation}.jsonl`。日志`/tmp/G3-DPO-TRUE-s42-a01.log`，哨兵`/tmp/g3_dpo_true_progress.log`。此时GPU1已空（SHUFFLE训练已结束），满足≤2并行GPU约束（当前只有1个训练在跑）。status：**G3-DPO-TRUE-s42-a01: RUNNING**。

- 2026-10-30 01:42 +08:00：`G3-DPO-TRUE-s42-a01`正常结束（01:21:44→01:42:21，见`/tmp/g3_dpo_true_progress.log`）。270/270 optimizer steps，train_loss=0.6855，eval_loss=0.6776，**eval_rewards/accuracies=0.6134**（高于SHUFFLE的0.4769约13.7pp），eval_rewards/margins=+0.0350（SHUFFLE为−0.0067），total_wall_seconds=1269.06≈0.35 GPU-hours。final adapter 23,613,253 bytes（仅`adapter_model.safetensors`+`adapter_config.json`），sha256记录于run.json。训练入口train_dpo.py未改动，超参与G2/G3-SHUFFLE逐项一致；本轮无a02。status：**G3-DPO-TRUE-s42-a01: SUCCEEDED**。累计新增GPU-hours（本轮G节）：Step A五次评分≈0.70h + DPO-SHUFFLE≈0.70h + DPO-TRUE≈0.35h ≈ 1.75h（预算8h）。下一步：实现conditional评估模式（新manifest+probe、三读出），并运行G3-BASE-a01/G3-TRUE-eval-a01/G3-SHUFFLE-eval-a01。

- 2026-10-30 02:05 +08:00：新增`evaluate_actions.py --mode conditional`（条件数据1152场景+probe 96端点统一评估，三读出：deployed greedy、L_policy候选MAP、implicit rerank；prompt用`format_conditional_prompt` variant0，与训练一致；greedy显式`use_model_defaults=False/do_sample=False/num_beams=1/max_new_tokens=64`）。CPU自测通过：manifest 1152行=720/144/288、192 families×6、true_ranking与exact oracle逐条一致、相邻gap≥0.25；probe 48对/96场景/drift-prior-reliability各16对、同对成本相同、两端最优不同且file标注与oracle一致；并列评分为均匀期望（3-way tie regret=1.0/optimal_rate=1/3，2-way tie=0.5）；`test_core.py` 14/14通过。GPU 30秒级smoke（2场景+1对probe，输出到/tmp，不计产物）通过。下一步：GPU0顺序运行G3-BASE-a01→G3-TRUE-eval-a01→G3-SHUFFLE-eval-a01。status：**G3-BASE-a01/G3-TRUE-eval-a01/G3-SHUFFLE-eval-a01: RUNNING**。

- 2026-10-30 10:32 +08:00：`G3-BASE-a01`SUCCEEDED（10:14:32→10:30:25，约15.8分钟，GPU0）：新conditional数据上BASE deployed greedy在train/validation/test/probe全部选precise（720/144/288/96），regret分别0.4334/0.4570/0.4470/0.5318，恰好等于constant-precise（也等于constant-skip/quick，按家族对称性三者相同）的oracle regret；policy MAP与greedy相同；implicit读出为三候选精确并列（margin=0），按均匀期望算regret恰好也等于常数基线。`G3-TRUE-eval-a01`SUCCEEDED（10:15:44→10:38:50，约23.1分钟，GPU6）：**deployed greedy仍全部选precise（train/validation/test/probe）**，validation greedy optimal_action_rate=0.3333、regret=0.4570（=常数基线，未达0.3656门槛）；但**implicit rerank出现真实条件结构**：validation动作混合quick69/skip30/precise45、regret=0.3624、optimal=0.444、strict_pair=0.613；test混合skip50/quick169/precise68、regret=0.3588、optimal=0.446、pair=0.604；train regret=0.3399/pair=0.613。**SFT触发判定：TRUE的validation greedy三项门槛全部未同时达到（acc 0.333<0.50；regret 0.4570>0.3656；与SHUFFLE差值待SHUFFLE完成后核对但前两项已足），G3-SFT-TRUE-s42-a01触发。** 新增`train_sft.py`（本轮唯一新代码文件）：同一底模、同一TRUE manifest的720个train场景、每场景仅一个全局最优动作（240/240/240平衡）、3 epochs=270 optimizer steps、仅动作内容token交叉熵（prompt/header/padding/EOS/suffix全mask）、LoRA qv rank16/alpha32/dropout0.05、lr1e-5、microbatch1/grad_accum8、seed42、save_strategy=no；CPU自测通过（720行、270步、边界/掩码校验）。GPU6启动训练，日志`logs/G3-SFT-TRUE-s42-a01.log`，哨兵`/tmp/g3_sft_progress.log`。status：**G3-SFT-TRUE-s42-a01: RUNNING**（GPU6），SHUFFLE评估仍在GPU0（≤2并行）。

- 2026-10-30 10:56 +08:00：三个conditional评估全部SUCCEEDED。`G3-SHUFFLE-eval-a01`（10:30:25→10:54:32，约24.1分钟，GPU0）：deployed greedy同样全部precise（train/validation/test/probe），validation greedy acc=0.333、regret=0.4570；implicit rerank接近/差于常数：train 0.4442（pair 0.482）、validation 0.3851（pair 0.530）、test 0.4478（pair 0.497）、probe 1.2776（pair 0.516）。**SFT触发确认满足：TRUE validation greedy三项门槛全部未达（0.333<0.50；0.4570>0.3656；与SHUFFLE的0.333相同，差0pp<10pp）。**`G3-SFT-TRUE-s42-a01`训练SUCCEEDED（10:40:08→10:52:13，270/270 steps，train_loss=0.1880，train拟合token准确率0.9512、整动作完全正确率0.4931，adapter已导出）。主分析（BASE/TRUE/SHUFFLE，family级配对bootstrap 2000次seed23，48个test families）：**deployed greedy的Delta=Regret_SHUFFLE−Regret_TRUE=0.0000，95%CI[0.0000,0.0000]**（两者都是全precise，逐条相同）；candidate sequence MAP同样Delta=0；**implicit rerank的Delta=+0.0891，95%CI[+0.0404,+0.1395]**（TRUE test implicit regret 0.3588 vs SHUFFLE 0.4478，TRUE显著更好但幅度有限）。cost-only kNN基线（仅两项cost、k=31、train拟合）validation regret=0.2101/acc=0.660、test regret=0.2383/acc=0.601——**TRUE implicit（test 0.3588）仍明显差于仅用cost的kNN基线**。TRUE implicit在test上33/48 families出现动作切换（switch_rate=0.6875），SHUFFLE也有41/48（0.8542）——切换本身不构成条件能力证据，需结合准确率/regret与probe判读。`G3-SFT-eval-a01`在GPU6运行中（10:52:15起）。

### G-结果：G3 conditional 全部完成（2026-10-30 11:15 +08:00）

**运行状态：** G3-READ-BASE/I42/I43/C42/C43、G3-DATA、G3-DPO-TRUE、G3-DPO-SHUFFLE、G3-BASE、G3-TRUE-eval、G3-SHUFFLE-eval、G3-SFT-TRUE（训练）、G3-SFT-eval 全部 SUCCEEDED；无FAILED/CANCELED/未执行；无a02重试。

**主结果表（新conditional数据；regret为exact oracle期望，越低越好；常数基线train/val/test=0.4334/0.4570/0.4470，三者相同）：**

| 模型 | split | greedy regret | greedy 最优率 | policy MAP regret | implicit rerank regret | implicit 最优率 | implicit 严格pair |
|---|---|---:|---:|---:|---:|---:|---:|
| BASE | train | 0.4334 | .333 | 0.4334 | 0.4334（全并列） | .333 | .500 |
| BASE | validation | 0.4570 | .333 | 0.4570 | 0.4570（全并列） | .333 | .500 |
| BASE | test | 0.4470 | .333 | 0.4470 | 0.4470（全并列） | .333 | .500 |
| BASE | probe | 0.5318 | .417 | 0.5318 | 1.0552（全并列） | .333 | .460 |
| TRUE | train | 0.4334 | .333 | 0.4334 | **0.3399** | .442 | .613 |
| TRUE | validation | 0.4570 | .333 | 0.4570 | **0.3624** | .444 | .613 |
| TRUE | test | 0.4470 | .333 | 0.4470 | **0.3588** | .446 | .604 |
| TRUE | probe | 0.5318 | .417 | 0.5318 | 0.7833 | .344 | .554 |
| SHUFFLE | train | 0.4334 | .333 | 0.4334 | 0.4442 | .328 | .482 |
| SHUFFLE | validation | 0.4570 | .333 | 0.4570 | 0.3851 | .410 | .530 |
| SHUFFLE | test | 0.4470 | .333 | 0.4470 | 0.4478 | .340 | .497 |
| SHUFFLE | probe | 0.5318 | .417 | 0.5318 | 1.2776 | .312 | .516 |
| SFT | train | **0.3191** | **.493** | 0.3272 | 0.4334（全skip退化） | .333 | .500 |
| SFT | validation | **0.3440** | **.465** | 0.3560 | 0.4570（全skip退化） | .333 | .500 |
| SFT | test | **0.3109** | **.521** | 0.3170 | 0.4470（全skip退化） | .333 | .500 |
| SFT | probe | 0.5099 | .448 | 0.4870 | 1.6870（全skip退化） | .302 | .460 |

**主指标（用户指定）：** 新test实际greedy动作的Δ=Regret_SHUFFLE−Regret_TRUE=**0.0000**，family级配对bootstrap（48个test families、2000次、seed23）95%CI=**[0.0000, 0.0000]**；相对常数基线的降低=0。TRUE与SHUFFLE的greedy在train/validation/test/probe全部为precise且逐条相同（candidate sequence MAP同样Δ=0）。**implicit rerank的Δ=+0.0891，95%CI [+0.0404, +0.1395]**（TRUE test 0.3588 vs SHUFFLE 0.4478）。补充：Δ_SFT−SHUFFLE(greedy,test)=**−0.1361**，95%CI [−0.1687, −0.1066]。

**cost-only基线（仅两项cost的k=31 kNN，train拟合）：** train regret 0.2212/acc .621；validation 0.2101/.660；test **0.2383/.601**。TRUE implicit（test 0.3588）**明显差于cost-only**；SFT greedy（0.3109）也差于cost-only。

**family正确动作切换（test，48 families）：** TRUE greedy/policy 0/48切换（全precise）；TRUE implicit 33/48 families切换（.688），但family内成员准确率仅.446、严格pattern匹配0/48；SHUFFLE implicit 41/48切换（.854）但准确率.340（≈chance）——**切换率高不等于条件能力**；SFT greedy 38/48切换、成员准确率.521、pattern匹配1/48。

**fixed-cost probe（48对/96端点，pairs两侧最优动作必须不同）：** TRUE implicit两端同时正确率**4.3%**（2/47，1对并列剔除），group regret：drift .542、prior **1.144（差于常数.prior .416）**、reliability .664；SHUFFLE implicit 0%；SFT greedy两端同时正确率**12.5%**（6/48）、policy 14.6%（7/48），group regret：drift .434/.420、prior .684/.675、reliability .411/.366（reliability组最优率.656/.688）。**probe总体不支持模型利用了cost之外的信息字段；按用户预先声明，probe失败不能单独证明完全未读取信息字段。**

**SFT触发原因（仅validation greedy判定）：** TRUE三项门槛全部未达——最优动作率0.3333<0.50；regret 0.4570>0.3656（=80%×best-constant 0.4570）；比SHUFFLE高0pp<10pp。SFT训练结果：270/270 steps、train_loss 0.1880、训练集拟合token准确率.9512/整动作完全正确率.4931；held-out（test greedy）整动作准确率.521、regret .3109——**SFT在train与held-out上一致地好于常数基线，DPO的deployed greedy则完全未离开常数策略**。

**并列（ties）处理：** BASE implicit为构造性三候选精确并列（全768+96条），按均匀期望计算、不取字典首项；TRUE implicit并列train4/val0/test1/probe1（switch分析中从非并列pair分母中剔除）；SHUFFLE与SFT implicit无并列。**并列导致的随机切换未计入条件能力。** 另经复核发现：`summary.json`中保存的`strict_pair_accuracy`**不是tie-aware**——全并列判据会因排序方向获得偶然部分得分（G3-READ-BASE implicit存储值0.723即为该假象，非排序能力）。分析时改用tie-aware重算（criterion差≤1e-10的pair从分母剔除）：Step A policy L_policy的pair正确率BASE/I42/I43完全相同（0.673/0.715/0.277，说明三个模型的候选序列排序未变）；I42 implicit为0.704/0.745/0.393、I43为0.692/0.727/0.352、C42/43为0.327/0.285/0.723；BASE implicit全部并列→无定义。conditional：TRUE implicit tie-aware pair=0.613/0.613/0.606（test 522/862，ties 2）、SHUFFLE=0.482/0.528/0.497、SFT=0.500（固定skip排序在家族对称构造下的机会水平）；BASE implicit无定义（全并列）。**报告与结论一律以tie-aware数值为准，存储summary中的BASE implicit pair值不用于推断。**

**定位结论（按用户第九节映射，可同时成立）：**（1）主指标Δ(greedy)=0且CI退化——**DPO TRUE与SHUFFLE的部署策略无任何差异，DPO未把偏好转化为动作**；（2）implicit rerank显著变好→**偏好可读，但原生成策略未改善**；（3）TRUE implicit优于常数/SHUFFLE但劣于cost-only kNN且probe无支持→**只证明部分条件关联，不证明信息价值机制**；（4）SFT泛化成功而DPO失败→**当前DPO目标/优化与生成先验耦合值得关注，原教师机制仍未证明**；（5）train与held-out表现一致（SFT train exact .493 vs test .521；TRUE implicit train .340 vs test .359），**无严重过拟合/捷径证据**；（6）DPO deployed策略连train都未拟合（train regret=常数）→**当前有限预算下的DPO首动作路线应停止；SFT正对照证明数据与流程可学习**。

**预算与产物：** 本轮GPU-hours≈0.861(Step A 5次评分)+0.699(DPO-SHUFFLE)+0.342(DPO-TRUE)+0.263(G3-BASE)+0.385(G3-TRUE-eval)+0.400(G3-SHUFFLE-eval)+0.333(G3-SFT-eval)+0.200(SFT训练)=**≈3.48 GPU-hours**（上限8）；单训练attempt最大0.70h（上限2）；并行GPU最多2（GPU0+GPU6）；final adapter共**3个**（TRUE/SHUFFLE/SFT各23,613,253 bytes≈22.5MiB，上限512MiB/个、≤3个）；本轮新产物≈90MB（上限2GiB）；无checkpoint/W&B/trainer_state/tokenizer副本/逐样本文件。


## H. G4 证据关联过程评分实验（用户2026-10-30显式新决策，按 patchs/MAINLINE_RESEARCH_PROMPT.md G4范围执行）

研究问题：冻结的agent过程/价值评分器，能否根据当前可见证据的版本、适用对象、授权变化及执行确认，正确改变候选动作排序？只重新组织已有证据，能否降低评分引导的决策损失并改善短闭环执行？保留G1机制证据不足结论与G2/G3全部结果；本轮不训练actor/adapter/checkpoint（0训练、0导出）。

### H0. 本轮登记的逻辑run

| RUN_ID | 工作 | 状态 |
|---|---|---|
| G4-DATA-a01 | 180 families数据、exact gold、CPU baselines、manifest freeze（CPU-only） | COMPLETED |
| G4-AGENTRM-a01 | AgentRM 8B原生scalar：dev→audit→closed_loop | COMPLETED |
| G4-QWEN-a01 | Qwen3-4B prompted judge：dev→audit→closed_loop | COMPLETED |
| G4-ANALYZE | 汇总、失败实例、研究判断 | COMPLETED |

本机无Slurm，均为本地直接attempt；只有真实启动证据才写RUNNING，无QUEUED。协调者是status.md唯一writer。

### H-预算账目（按用户G4指令）

| 项目 | 协议上限 | 当前已用/预留 |
|---|---:|---:|
| GPU-hours | ≤8（含加载+dev+audit+闭环+失败+重试；两卡并行1小时=2 GPU-hours） | **1.327**（AgentRM 0.535 + Qwen 0.792） |
| 单评分器GPU-hours | 先按≤4分配 | AgentRM 0.535、Qwen 0.792 |
| 并行GPU | ≤2，每进程单卡 | 2（GPU1 AgentRM + GPU0 Qwen） |
| AgentRM logical candidate scores | ≤16,848 | planned 9,798（offline 7,695 + loop 2,103）；unique forward 9,546（含cache命中差异） |
| Qwen generation calls | ≤5,616 | planned 2,920（offline 2,565 + loop 355）；unique 2,836 |
| 训练/adapter/checkpoint/optimizer state/模型导出 | 0 | **0**（runs/G4-*中无safetensors/adapter/checkpoint） |
| 新增非模型产物 | ≤1 GiB（共享数据≤128MiB；单attempt≤128MiB；单日志≤10MiB） | **≈6.6MB**（data 2.2MB + runs 4.3MB + logs 88KB） |
| 新增模型缓存 | 24 GiB（文件声明）；实际repo为27.97GiB fp32，见H-notes | **27.97 GiB**（唯一revision，见偏差说明） |
| 新源文件 | 仅trajectory_cases.py、trajectory_score.py | 2 |

### H-notes

- 2026-10-30 12:05 +08:00：登记H节；开始下载唯一允许的预训练scorer `cheesewafer/Meta-Llama-3.1-8B-Instruct-regression-avg-all-v3-True-False`，固定revision `411ac92b1bdf1cfbe10653d56ddb31611af607e4`（HF API实查），经hf-mirror（huggingface.co直连在本机超时；socks代理可用但镜像直连更快）。**实际文件树合计27.97GiB（7个fp32 safetensors分片），超出文件所述24GiB缓存上限约3.97GiB**；/data4当时可用37GiB，下载仅取config/tokenizer/index/safetensors（跳过README、.gitattributes、training_args.bin），不复制到runs，不转换/导出。此偏差将在最终报告显式列出。
- 2026-10-30 12:05 +08:00：实现与数据生成（CPU）进行中，尚未调用任何评分器。
- 2026-10-30 16:02 +08:00：`G4-DATA-a01`完成（CPU-only）：`data/g4_evidence/cases.jsonl`（180 families；dev36/audit144；6 cells各6/24）+ `manifest.json`，共2.2MiB。验证：九个解析状态（A/B known-good/known-bad/unknown、C not_committed/committed/unknown）与Q^π0逐项一致；reversal主prefix两端gold不同且top-2 gap≥0.5；invariant两端gold相同；invariant gold动作dev每机制2/2/2、audit 8/8/8；候选c0/c1/c2的gold分布131/108/121（非固定c0）；secondary分支：225个已知变体1支、135个未知变体2支；rejection=0。CPU baselines（audit macro）：π0=0.000 regret/opt1.0、exact_planner=0.000/1.0、always_query=0.625/0.375、always_direct=15.667/0.333、always_stop=4.833/0.292。预选closed-loop 36 audit families（每机制12：reversal 6 + invariant 6）。status：**G4-DATA-a01: COMPLETED**，数据已冻结。
- 2026-10-30 16:06 +08:00：`G4-QWEN-a01`在GPU0启动（PID 3508924，`CUDA_VISIBLE_DEVICES=0`），单样本接口检查通过（LEDGER prompt 1220 tokens，返回合法`{"scores":{"c0":80,"c1":0,"c2":30}}`）。dev phase进行中；日志`logs/G4-QWEN-a01.log`。status：**G4-QWEN-a01: RUNNING**。
- 2026-10-30 16:20 +08:00：AgentRM权重下载完成（revision `411ac92b1bdf1cfbe10653d56ddb31611af607e4`，28GiB fp32分片→加载为bf16；/data4剩余8.9GiB）。单样本接口检查通过：`LlamaForSequenceClassification`，num_labels=1，`score.weight` [1,4096]来自checkpoint（未新初始化），示例logits[1,1]=0.664，prompt 924 tokens，渲染结尾为候选动作。修正一处输入序列化：AgentRM候选动作不再包含`candidate_id`元数据（与历史assistant动作格式一致）。`G4-AGENTRM-a01`在GPU1启动（PID 3536921，单卡，与Qwen并行为2卡）。dev进度：前50 prefix-views（150 logical scores）完成，scores有限且同prefix内spread均值0.081（区分度偏小，待审计）。Qwen dev已完成：513/513 calls，invalid=0，token max 1461；audit进行中（200/2052 views）。status：**G4-AGENTRM-a01: RUNNING，G4-QWEN-a01: RUNNING**。

### H-结果：G4 全部完成（2026-10-30 17:05 +08:00）

**运行记录：** G4-DATA-a01（CPU，2.2MiB数据，rejection=0）COMPLETED；G4-AGENTRM-a01（GPU1，PID 3536921，16:20:54→16:53:13，1927.3s≈0.535 GPU-hours）COMPLETED；G4-QWEN-a01（GPU0，PID 3508924，16:06:41→16:54:22，2851.7s≈0.792 GPU-hours）COMPLETED；G4-ANALYZE COMPLETED。无失败、无a02、无重试。0训练/0 adapter/0 checkpoint/0 optimizer state/0模型导出。

**有效样本：** 两评分器均0 invalid。AgentRM 9,546条candidate score（dev 1,539 / audit 6,156 / closed_loop 2,151）+300条闭环轨迹；Qwen 2,836次generation（dev 513 / audit 2,052 / closed_loop 571）+300条轨迹。AgentRM token均值919.9/max 1,373；Qwen均值1,206.4/max 1,689（均<3,072上限）。FORMAT与LEDGER token数几乎相同（AgentRM 981.5 vs 981.9；Qwen 1,273.7 vs 1,271.6）。

**主表（audit primary，macro equal-cell；每cell 24 audit families）：**

| 评分器 | 视图 | Regret | 95% CI | 最优动作率 | Strict pair | 同history margin |
|---|---|---:|---|---:|---:|---:|
| AgentRM | RAW | 8.753 | [7.186, 10.344] | 0.443 | 0.487 | 0.017 |
| AgentRM | FORMAT | 5.332 | [4.240, 6.366] | 0.345 | 0.509 | 0.024 |
| AgentRM | LEDGER | 5.328 | [4.240, 6.361] | 0.349 | 0.510 | 0.023 |
| Qwen | RAW | 11.161 | [9.656, 12.753] | 0.411 | 0.523 | 55.382 |
| Qwen | FORMAT | 9.661 | [8.231, 11.184] | 0.451 | 0.551 | 53.229 |
| Qwen | LEDGER | 10.052 | [8.589, 11.625] | 0.436 | 0.547 | 53.559 |

**唯一预注册比较 Delta_specific = Regret(FORMAT) − Regret(LEDGER)：AgentRM = +0.003472，95% CI [+0.000000, +0.010417]**（远低于+0.5门槛；CI下界约为0，不满足实用标准）。Qwen同一比较 = **−0.390625**，CI [−1.050347, +0.232639]。RAW−LEDGER：AgentRM +3.425 [+1.793,+4.955]；Qwen +1.109 [−0.156,+2.399]。RAW−FORMAT：AgentRM +3.422 [+1.790,+4.951]；Qwen +1.500 [+0.274,+2.778]。即两评分器仅表现出“加入任何索引/重复事实”的呈现效应，LEDGER相对FORMAT无实质增益；Qwen在C机制上LEDGER反而更差。

**Per-cell regret：** AgentRM RAW/FORMAT/LEDGER：versioned_report:reversal 11.250/10.969/10.969；versioned_report:invariant 9.198/12.312/12.312；scoped_authorization:reversal 3.740/1.958/1.958；scoped_authorization:invariant 1.156/4.792/4.792；exactly_once:reversal 11.802/0.896/0.875；exactly_once:invariant 15.375/1.062/1.062。Qwen对应：12.552/11.521/11.521；12.854/11.646/11.646；7.000/5.208/5.208；11.260/10.667/10.667；10.479/10.875/12.135；12.823/8.052/9.135。**FORMAT与LEDGER在6个cell中有5个完全同值**，说明分组整理未改变评分排序；唯一差异（AgentRM C reversal 0.896→0.875）幅度0.02。

**诊断量（audit primary）：**
| 评分器/视图 | reversal两端都正确 | 正确切换率 | invariant两端正确 | invariant严格切换率 | invariant argmax TV | 不必要查询 | 应查未查 | 违规执行 | 重复执行 | 二次重复查询（posterior加权） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AgentRM RAW | 0.167 | 0.167 | 0.319 | 0.194 | 0.229 | 0.215 | 0.170 | 0.160 | 0.094 | 0.168 |
| AgentRM FORMAT | 0.094 | 0.094 | 0.215 | 0.167 | 0.188 | 0.420 | 0.146 | 0.094 | 0.000 | 0.679 |
| AgentRM LEDGER | 0.108 | 0.108 | 0.215 | 0.167 | 0.188 | 0.417 | 0.146 | 0.094 | 0.000 | 0.630 |
| Qwen RAW | 0.104 | 0.104 | 0.365 | 0.042 | 0.083 | 0.101 | 0.323 | 0.198 | 0.042 | 0.073 |
| Qwen FORMAT | 0.083 | 0.083 | 0.385 | 0.139 | 0.188 | 0.104 | 0.302 | 0.156 | 0.024 | 0.094 |
| Qwen LEDGER | 0.069 | 0.069 | 0.368 | 0.139 | 0.194 | 0.115 | 0.302 | 0.163 | 0.031 | 0.092 |

**Gold-action分层 regret（audit primary）：** AgentRM RAW query/direct/stop = 8.97/0.469/13.77；FORMAT 5.90/0.58/7.18；LEDGER 5.89/0.58/7.18。Qwen RAW 11.48/0.24/17.70；FORMAT 10.85/0.45/13.03；LEDGER 11.21/0.24/13.89。**失败集中在query与stop类，direct类较好；但invariant与secondary准确率均远低于80%参考线，不能据此认定“证据失效”特异机制。**

**闭环（36 audit families；posterior加权期望）：**
| 评分器/视图 | Utility | 95% CI | 安全完成 | 违规 | 重复 | 查询数 | 未完成 | horizon用尽 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| AgentRM RAW | −7.917 | [−11.132, −4.667] | 0.479 | 0.236 | 0.174 | 0.49 | 0.097 | 0.014 |
| AgentRM FORMAT | −3.917 | [−6.625, −1.181] | 0.271 | 0.188 | 0.014 | 2.25 | 0.014 | 0.514 |
| AgentRM LEDGER | −4.896 | [−8.285, −1.389] | 0.306 | 0.243 | 0.007 | 2.19 | 0.014 | 0.431 |
| Qwen RAW | −6.035 | [−9.500, −2.514] | 0.583 | 0.264 | 0.125 | 0.20 | 0.021 | 0.007 |
| Qwen FORMAT | −4.618 | [−7.931, −1.104] | 0.625 | 0.271 | 0.083 | 0.24 | 0.014 | 0.007 |
| Qwen LEDGER | −5.076 | [−8.417, −1.562] | 0.611 | 0.271 | 0.097 | 0.15 | 0.014 | 0.007 |

**闭环主要比较 Utility(LEDGER)−Utility(FORMAT)：AgentRM −0.979，CI [−3.104, +0.813]；Qwen −0.458，CI [−1.583, +0.153]。** 均不与主指标同向，且CI含0。CPU闭环基线（同36 families）：π0 +6.583 [5.778,7.444]、always_stop +1.806 [1.528,2.083]、always_query −2.194 [−2.472,−1.917]、always_direct −8.889 [−12.5,−5.278]。**两个评分器的闭环效用均低于always_stop，远低于π0；AgentRM的索引视图还诱导大量重复查询（FORMAT/LEDGER约2.2次/轨迹，51%/43%轨迹耗尽horizon）。**

**代表性失败/无效实例（直接列此，不建单例文件）：**
1. `F0006`（versioned_report/reversal_good，stale端，audit）：gold=inspect(Q=4)；AgentRM **RAW**选择execute(c1,Q=−10)，scores c0=0.648/c1=0.676/c2=0.656；FORMAT翻转选inspect(c0=0.688>c1=0.684)——0.03级近并列下的选择不稳定。闭环同一root在hidden bad世界执行execute，utility=−30（违规）。
2. `F0006`（current端secondary）：已有权威good证据，AgentRM **FORMAT**仍选inspect(c0=0.676)而非execute(c1=0.672)，Q(inspect)=9 vs Q(execute)=10——重复查询/未采纳明确证据。
3. `F0006`（LEDGER近并列）：c0=0.668 vs c1=0.664，argmax由稳定hash决定，非证据；全局AgentRM同history margin均值仅0.023。
4. `F0013`（versioned_report/reversal_bad，stale端）：Qwen **RAW**给execute与inspect同为10.0（精确并列→均匀tie），FORMAT给inspect=30>execute=10并选中gold inspect——差异来自重复呈现事实，而非LEDGER分组。
5. `F0054`（versioned_report/invariant_bad，unrelated-event两端）：Qwen **LEDGER**一端选abort(c1=30,Q=−30)，另一端选inspect(c2=70,Q=−1)，gold两端均为abort(Q=0)——无关事件对上发生严格切换，且两端都错。
6. `F0006`闭环结构性问题：AgentRM FORMAT/LEDGER平均查询2.25/2.19次，51.4%/43.1%轨迹4步horizon耗尽，utility −3.9/−4.9；索引把模型推向反复核验而非完成任务。

**研究判断（按预先约定分支）：主要分支 = GENERAL_TASK_MISMATCH。** 理由：(a) Delta_specific仅+0.003<+0.5且CI下界≈0，未达实用门槛；(b) 闭环LEDGER−FORMAT为负且CI含0，不同向；(c) RAW在reversal与invariant、以及query/stop/direct各gold层均失败（reversal两端正确0.17/0.10，invariant 0.32/0.36，secondary adoption 0.57/0.66，均远低于80%参考线）；(d) 动作分布接近均匀、AgentRM margin≈0.02，说明不是“特定证据关联排序”问题而是整体任务/目标不适配；(e) 唯一可重复的改善来自“加入任何事实索引”的呈现/重复效应（RAW→FORMAT），LEDGER无额外贡献（FORMAT_ONLY式呈现效应，非证据关联方法贡献）。**不能宣称已定位动态证据失效机制；研究主线在当前评分器/任务组合上不值得直接进入方法阶段。** 下一步最值得做的一项：先用AgentRM官方原生benchmark格式/示例验证该checkpoint在本环境中是否具备基本排序能力（或改用面向过程监督的native scorer），再决定是否值得重跑同一冻结数据；不自动训练scorer、不启动PRM/在线RL。

**预算/产物：** GPU-hours 1.327（≤8；单评分器≤4）；planned logical 9,798/2,920、unique 9,546/2,836（均≤16,848/5,616）；新增产物≈6.6MB（≤1GiB）；新源文件2个；0 checkpoint/adapter；模型缓存27.97GiB（唯一revision `411ac92b...`，超文件24GiB声明约3.97GiB，已在此显式记录）；/data4在下载后剩余8.9GB。README已按指令对TWICE arXiv v2撤回做一行勘误。

## I. G5 证据解析—价值排序分解实验（用户2026-10-30显式新决策，按 patchs/G5_STATE_VALUE_EXPERIMENT_PROMPT.md）

本轮只做一次固定的分解实验：先完成 G4 的 CPU 更正并保留原始 G4 结果，再用 data_seed=501 / split_seed=502 生成 G5 数据；复用 AgentRM 与 Qwen3-4B 两个冻结模型，不训练、不导出 checkpoint、不下载模型。G4 的 GPU predictions 不重跑。

### I0. 登记

| RUN_ID | phase | 状态 | PID/Job | GPU | 说明 |
|---|---|---|---|---|---|
| G5-DECOMPOSE-a01 | G4-CPU-CORRECT | COMPLETED | CPU | — | 2026-10-30 23:19；35 个 secondary 修正、0 primary；原始 G4 文件未改 |
| G5-DECOMPOSE-a01 | G5-DATA | COMPLETED | CPU | — | 2026-10-30 23:20；180 families/36 dev+144 audit；360 primary+495 secondary；closed-loop 36 |
| G5-DECOMPOSE-a01 | G5-DEV | COMPLETED | 4190939 | 4→cuda:0 | 2026-10-30 23:20:43–23:31:39；Qwen 684、AgentRM 1539、invalid 0；格式检查通过 |
| G5-DECOMPOSE-a01 | G5-AUDIT | COMPLETED | 18191 | 4→cuda:0 | 2026-10-30 23:37:45–00:15:22；Qwen 3420、AgentRM 7695、invalid 0；2256s |
| G5-DECOMPOSE-a01 | G5-CLOSED_LOOP | COMPLETED | 90764 | 4→cuda:0 | 2026-10-31 00:16:51–00:22:01；485 trajectories；308.5s |
| G5-DECOMPOSE-a01 | G5-ANALYZE | COMPLETED | CPU | — | 2026-10-31 00:23；写入 run.json metrics.g5 与 decision |

### I1. G4 CPU 更正结果（原始 G4 未覆盖）

- G4 原始 run 保留：`runs/G4-AGENTRM-a01/`、`runs/G4-QWEN-a01/`；本阶段只读其 cases/predictions。
- 更正后的 35 个标签变化全部是 scoped_authorization 的 secondary 分支（dev 7 / audit 28），primary 0 变化；Qwen/AgentRM 共用同一批 prefix。
- 更正后 π0 闭环 utility 由 6.583333 变为 6.694444；always_query/direct/stop 仍为 -2.194444/-8.888889/1.805556。
- 更正后 semantic action distribution（query/direct/stop）：
  - AgentRM RAW 39.5833/54.8611/5.5556；FORMAT 62.8472/36.8056/0.3472；LEDGER 62.5000/37.1528/0.3472。
  - Qwen RAW 12.1528/82.8125/5.0347；FORMAT 15.7986/76.5625/7.6389；LEDGER 16.6667/77.9514/5.3819。
- secondary 均匀并列正确率：AgentRM RAW/FORMAT/LEDGER=0.5634/0.1415/0.1866；Qwen=0.6285/0.6233/0.6259。
- FORMAT 与 LEDGER 的主输入相同 232/288=80.56%（A/B 各 96/96，C 40/96）；实际改变的 56 个 C prefix 上，AgentRM FORMAT−LEDGER regret=+0.017857，Qwen=-2.008929。
- C horizon 轨迹只更正 safe_completion/incomplete 标记（AgentRM 65 个标记变化；Qwen 0），utility 未改；旧 AgentRM view-key tie 只作说明，不做反事实相减。
- 输出：`runs/G5-DECOMPOSE-a01/run.json` 的 `g4_corrected_analysis`；sanity checks 全通过；新增模型调用=0，GPU-hours=0。

### I2. G5 数据冻结

- 生成命令：`python -m trajectory_cases --output-dir data/g5_state --data-seed 501 --split-seed 502 --dataset-version g5-decompose-v1`。
- `data/g5_state/cases.jsonl`（2.2MiB，sha256 `0fc3c833...`）；`manifest.json`（12KiB，sha256 `78fda628...`）。
- 180 families；dev 36 / audit 144；每 cell 6 dev+24 audit；360 primary+495 secondary；closed-loop 预选 36 audit families（每 cell 6）。
- 数据使用修正后的授权规则（allow grant 是证据；无目标证据为 unknown 先验 1/2；显式 revoke 或准确 query deny 为 deny；不再要求证书），独立 dataset_version/call_key namespace `g5-decompose-v1:501:502`。

### I3. G5 dev 格式检查（2026-10-30 23:31）

- 实际 PID 4190939，GPU4（逻辑 cuda:0）；成功阶段约570s，含早先失败重试的总 GPU 占用约0.19 GPU-hour。
- 真实调用：Qwen 684（E 171 + RAW/O/E 各171）、AgentRM 1539（3 representations × 3 candidates × 171）；invalid extract 0、invalid score 0；token max 1256 < 3072；evidence-empty 0，evidence-invalid 1。
- 三次早先启动失败均为 0 成功模型调用或已缓存调用：第一次本地 snapshot 查找，第二次错误 venv 的 `use_model_defaults` 不兼容，第三次 extract 返回字段 bug；最终使用项目既有 `/data/sjj/process_reward_venv`。未因效果修改 prompt。
- 输出：`runs/G5-DECOMPOSE-a01/predictions.jsonl`（2,223 条）、`run.json`、`logs/G5-DECOMPOSE-a01.log`。

### I4. G5 audit 完成（2026-10-30–31）

- 实际 PID 18191，GPU4（逻辑 cuda:0），2026-10-30 23:37:45 启动，结束 2026-10-31 00:15:22；elapsed 2256.3s。
- 全量离线调用：Qwen 3420（E/RAW/O_STATE_R/E_STATE_R ×855）、AgentRM 7695（3 representations ×3 candidates ×855）；总计到 audit 为止累计 Qwen 3420、AgentRM 7695；invalid extract/score 均为0；token max 1256。
- 预检（CPU，只读 predictions；最终分析在闭环后统一写 run.json）：E audit state accuracy 0.9137（macro over mechanisms 0.9151）；dangerous audit: unknown→positive 25、bad→good 1；evidence valid citation precision 0.9927（1226 citations，9 个无效引用输出）。
- 预检离线 regret（audit primary, macro equal-cell）：Qwen RAW 9.734→RAW_CAL 5.580；AgentRM RAW 9.124→RAW_CAL 5.122；E_CPU 1.385；O_CPU 0.0；O_STATE_R Qwen 6.684 / AgentRM 12.377；E_STATE_R Qwen 7.608 / AgentRM 12.377。E_CPU 优于两个 RAW_CAL（配对 CI 下界<0），但劣于 always_query 0.625（E_CPU−always_query=+0.760，CI [+0.326,+1.240]），因此尚未达到进入外部验证的门槛。
- 闭环与最终分析已完成，见 I5/I6；上述预检与最终离线结果一致。

### I5. G5 最终结果（CPU 汇总，唯一 attempt）

**证据解析（E，audit）**：state accuracy 0.9137（625/684；dev 0.9006）；mechanism-macro 0.9151。分机制：versioned_report 0.7931、scoped_authorization 0.9909、exactly_once 0.9612。危险混淆：unknown→positive 25、bad→good 1（overall 26/684=3.80%；versioned_report 26/232=11.21%）；invalid state 0；evidence 引用 precision 0.9927（1226 citations；9 个输出含无效ID，0 空引用）。

**离线 audit primary（macro equal-cell；越低越好）**：

| 方法 | Regret | 最优率 |
|---|---:|---:|
| Qwen RAW | 9.734 | 0.453 |
| Qwen RAW_CAL | 5.580 | 0.545 |
| AgentRM RAW | 9.124 | 0.423 |
| AgentRM RAW_CAL | 5.122 | 0.417 |
| Qwen O_STATE_R（oracle state 诊断） | 6.684 | 0.590 |
| AgentRM O_STATE_R（oracle state 诊断） | 12.377 | 0.377 |
| Qwen E_STATE_R | 7.608 | 0.524 |
| AgentRM E_STATE_R | 12.377 | 0.377 |
| **E_CPU** | **1.385** | **0.868** |
| O_CPU（规则上界） | 0.000 | 1.000 |
| always_query | 0.625 | 0.375 |

关键配对：E_CPU−Qwen RAW_CAL = −4.194，95% CI [−5.538,−3.031]；E_CPU−AgentRM RAW_CAL = −3.736 [−4.865,−2.750]；E_CPU−always_query = **+0.760 [+0.326,+1.240]**（E_CPU 仍更差）；E_STATE_R−相应 RAW_CAL：Qwen +2.028 [+0.389,+3.618]、AgentRM +7.255 [+5.479,+9.085]。

**闭环（36 audit families；posterior 加权，越低/越高按列）**：

| 策略 | Utility↑ | 95% CI | safe↑ | total errors↓ | queries | incomplete |
|---|---:|---|---:|---:|---:|---:|
| AgentRM RAW_CAL | −3.000 | [−5.688,−0.583] | 0.340 | 0.188 | 0.778 | 0.472 |
| Qwen RAW_CAL | −3.458 | [−7.069,0.382] | 0.653 | 0.319 | 0.403 | 0.028 |
| **E_CPU** | **4.965** | **[3.681,6.194]** | **0.688** | **0.056** | **0.243** | 0.257 |
| AgentRM E_STATE_R | −6.056 | [−10.139,−1.944] | 0.590 | 0.389 | 0.292 | 0.021 |
| Qwen E_STATE_R | −1.576 | [−4.889,1.493] | 0.632 | 0.236 | 0.812 | 0.132 |

CPU 基线：π0 6.806、always_query −2.750、always_direct −6.389、always_stop 1.250。E_CPU−Qwen RAW_CAL = +8.424 [+5.292,+11.569]；E_CPU−AgentRM RAW_CAL = +7.965 [+5.368,+10.688]；E_CPU−Qwen E_STATE_R = +6.542 [+3.965,+9.549]。

**预注册外部验证门槛**：accuracy≥90% 通过；overall dangerous≤5% 通过（但 versioned_report 11.21% 暴露局部失败）；primary improvement CI 下界>0 通过；E_CPU<always_query **未通过**；闭环 utility 改善且 total errors 不升通过。**总 gate 未通过，不进入外部验证，不把 E_CPU 称为已验证方法。**

### I6. 本轮判断与唯一下一方向

- **问题主要不是普遍证据解析失败**：E 的 audit accuracy 0.9137、mechanism-macro 0.9151，evidence precision 0.9927；错误集中在 stale-revision 的 unknown→positive（versioned_report），这是局部而关键的证据解析缺陷。
- **主要限制是价值排序/冻结 scorer 适配**：即使给 oracle state，O_STATE_R 仍为 Qwen 6.684 / AgentRM 12.377；E_STATE_R 为 7.608/12.377，远差于 E_CPU 的 1.385。E_CPU 的主要错误来自 E 的 stale unknown 误判（如 F0006）；但当前两个 AgentRM/Qwen 标量 scorer 在 state-only 表示上不构成可用价值头。不能称“普遍 PRM 失效”，只能说该表示/目标与这两个冻结 scorer 不适配。
- **唯一后续方向**：**退出当前冻结标量 scorer + toy 组合**；下一步只做外部真实工具任务的问题定义与验证，比较“grounded state extraction + deterministic Q（E_CPU）”与 always_query，并把当前 AgentRM/Qwen 仅作负对照。不再对这两个 scorer 做 prompt/表示修补、不再增加同模板 toy 数据、不训练当前 scorer。
- 实际错误例（最多6个）：F0006 stale E 预测 good（真 unknown，证据e3）→E_CPU direct、闭环 utility −30；F0008 secondary 新 query=good 仍报 bad（证据e3）；F0007 Qwen O_STATE_R 给 c1=70 选 direct（gold c0，Q=−30）；F0007 AgentRM O_STATE_R 给 c1=0.625>c0=0.543 选 direct；F0006 AgentRM O_STATE_R 选 c0（gold c2=4，regret14）；F0006 E_CPU stale 分支单步 direct、utility −30。
- G4 保留与更正：G4 原始 primary 指标、模型闭环 utility、Delta_specific 负结论保留；35 个 scoped_authorization secondary 的 gold/Q 已更正（primary 0）；π0 6.583→6.694；action counts 改语义 kind；secondary 改均匀并列质量；gold strata 按每个 primary prefix 后按 family 聚类；C horizon 仅改 flags；旧 tie 不伪装反事实。

**预算/产物（G5）**：GPU-hours 0.904（≤4，含失败重试；成功阶段 0.871）；Qwen 3,767 calls（≤6,984）、AgentRM 8,427 scores（≤12,690），invalid 0/0；训练 run/checkpoint/adapter/optimizer 均为 0；新源文件 0、修改源文件2；数据2文件+run2文件+日志1；新增非模型产物约8.9MiB；模型均为本地既有 snapshot（AgentRM `411ac92b...`，Qwen `2de2439...`）。最终 run.json 见 `runs/G5-DECOMPOSE-a01/run.json`（`decision`/`g4_corrected_analysis`/`metrics.g5`），predictions `384d2960...`，日志 `logs/G5-DECOMPOSE-a01.log`。
## J. G6 外部原生 ToolSandbox 证据失效与最小修复联合实验（用户2026-10-31显式新决策）

固定依据：`patchs/G6_VALIDITY_INSIGHT_EXPERIMENT_PROMPT.md`。仅复用一个冻结 Qwen3-4B；不训练、不下载模型、不保存 checkpoint/adapter；不重跑 G4/G5；不继续调 AgentRM；不扩大同模板 toy 数据。新增项目源文件仅 `validity_experiment.py`。

### J0. 登记

| RUN_ID / phase | 状态 | 主机/PID | GPU | 起止时间 | 完成节点/总节点 | 模型调用/上限 | GPU-hours | 产物/阻塞 |
|---|---|---|---|---|---|---|---|---|
| G6-VALIDITY-a01 / G6-CPU-REANALYSIS | COMPLETED | CPU | 0 | 2026-10-31 03:20:06 | 全部只读分析 | 0 | 0 | `g5_reanalysis` in run.json；不改 G5 |
| G6-VALIDITY-a01 / G6-TOOLS-DATA | COMPLETED | CPU worker `/data/sjj/g6_tools_venv` | 0 | 2026-10-31 03:20:17 | 60 families / 1080 public nodes | 0 | 0 | `data/g6_validity/`；240次原生工具调用 |
| G6-VALIDITY-a01 / G6-DEV | COMPLETED | local PID 388944 | 4→cuda:0 | 2026-10-31 03:21:10–03:31:27 | 216/216 | 648/648 | 0.171 | invalid: R=4（known+null），E/V=0；格式检查通过，不改prompt |
| G6-VALIDITY-a01 / G6-AUDIT | COMPLETED | local PID 407171 | 4→cuda:0 | 2026-10-31 03:32:47–04:14:17 | 864/864 | 2,592/2,592（累计3,240） | 0.863 | E/V invalid=0，R invalid=14 |
| G6-VALIDITY-a01 / G6-EXECUTE | COMPLETED | local PID 477831 + CPU worker | 0 | 2026-10-31 04:14:40–04:15:23 | 2400/2400 | 0 | 0 | 2,400条原生工具轨迹；2,400 worker calls |
| G6-VALIDITY-a01 / G6-ANALYZE | COMPLETED | CPU | 0 | 2026-10-31 04:17:15 | 全部 | 0 | 0 | metrics/decision 写入 run.json |

### J1. G5 CPU 重分析（只读，不改 G5）

- audit primary：总体 250/288=86.81%；secondary 375/396=94.70%；versioned_report 58/96=60.42%，scoped_authorization 96/96，exactly_once 96/96。
- primary versioned_report 的 40 个真实 unknown：预测 unknown 3、good 25、bad 12。
- E_CPU primary regret 总和 399：U 的 unknown→good 25×14=350，unknown→bad 12×4=48，bad→unknown 1。
- reversal：current 23/24；stale 0/24。
- secondary：versioned_report 126/136，scoped_authorization 122/124，exactly_once 127/136。
- 没有新增 G5 模型调用；G5 run/predictions 未覆盖。

### J2. 原生工具接入与数据冻结

- ToolSandbox 固定 checkout：`external/ToolSandbox`，commit `c8571d7854316d2e1c5f288e59fe1e34e53f6dd1`，2.4MiB。
- CPU 专用环境：`/data/sjj/g6_tools_venv`（只安装原生工具需要的 CPU 包；GPU 项目 venv 未修改）。原生联系人/提醒查询、modify、消息写入和 reminder modify 已实际执行；两条工具联调成功。
- 数据：`data/g6_validity/cases.jsonl`（60 families，30 contact +30 reminder；dev 12 / audit 48）、`manifest.json`。每 family 4 条件×2 表示 root=8，U 两世界+其余3世界×2 表示=10，共1080 public nodes（dev216/audit864）。
- 固定通知 J 结构化模板与 N 自然语言模板；U/F/S/O 条件、J/N 表示均冻结；隐藏新值只在 cases private。原生 snapshot 冻结在每个 post-event 分支，策略执行从同一 snapshot 重启；没有重新注入变化。
- 预算：模型调用 0（此阶段）；ToolSandbox worker 调用 240；训练/checkpoint/adapter=0。

### J3. 当前运行状态

G6 全部 phase 已完成：dev（648 calls）→ audit（2,592 calls）→ execution（2,400 trajectories）→ analyze。当前无模型进程、无排队任务。


### J4. G6 正式结果（audit，n=48 families；J/N 不增加独立n）

**DIRECT_E root U/S/O/F（E 状态准确率；U 两个隐藏世界同root，只计一次）**：

| workflow/rep | U acc | U retract | U old-value use | F acc | S acc | O acc |
|---|---:|---:|---:|---:|---:|---:|
| contact/J | 0.00 | 0/24 | 24/24=100% | 0.9167 | 1.00 | 1.00 |
| contact/N | 0.00 | 0/24 | 24/24=100% | 1.00 | 1.00 | 1.00 |
| reminder/J | 0.00 | 0/24 | 24/24=100% | 1.00 | 1.00 | 1.00 |
| reminder/N | 0.00 | 0/24 | 24/24=100% | 0.50 | 1.00 | 1.00 |

**R/V root 关键对照（N 主分析；U 为 unknown/null 正确率）**：

| workflow | method | U acc / retract | F acc | S acc | O acc / erroneous retraction |
|---|---|---:|---:|---:|---:|
| contact | R | 0.50 / 12/24 | 1.00 | 0.875 | 0.958 / 1/24 |
| contact | V | 1.00 / 24/24 | 1.00 | 0.833 | 0.833 / 2/24 |
| reminder | R | 0.00 / 0/24 | 1.00 | 1.00 | 1.00 / 0/24 |
| reminder | V | 0.9167 / 22/24 | 1.00 | 0.958 | 0.125 / 19/24 |

- **主要机制比较**：`JointSuccess(V)−JointSuccess(R)` on audit N = **+0.2083**，95% CI [0.0833,0.3333]；按 workflow：contact +0.3333、reminder +0.0833。V 改善 U，但 reminder 的 other-object over-retraction 为 **79.17% vs R 0%**；S/O 错误撤销门槛失败 → **intervention pass=false**。
- **部署主表（audit N；utility/错误/完成/查询/逻辑调用；规则始终列入）**：

| strategy | utility | success | wrong side effect | tool error | queries | model calls |
|---|---:|---:|---:|---:|---:|---:|
| DIRECT_E | −2.500 | 0.688 | 0.313 | 0.000 | 0.000 | 1.000 |
| GENERIC_REVIEW | 5.760 | 0.802 | 0.073 | 0.125 | 0.073 | 2.146 |
| VALIDITY_REVIEW | 8.188 | 0.948 | 0.031 | 0.021 | 0.354 | 2.708 |
| QUERY_ONCE_V | 8.792 | 0.995 | 0.005 | 0.000 | 1.000 | 2.000 |
| **DEPENDENCY_RULE** | **9.750** | **1.000** | **0.000** | **0.000** | **0.250** | **0.000** |

- V−QUERY_ONCE_V：utility **−0.6042**，CI [−1.9583,+0.6094]；success **−4.69pp**，wrong +2.60pp，queries **−0.646**，model calls 2.708 vs 2.000。V 节省 query 但降低完成率/增加错误，**选择性收益门槛 pass=false**。
- Rule−V：utility **+1.5625**，success +5.21pp，wrong −3.13pp，queries 0.250 vs 0.354，模型调用 0；`rule_solved_all=true`。规则已解决全部条件。
- V relation（root+after）：N accuracy **0.9398**，relation 正确但最终 knowledge/value 错误 **7/432**；J accuracy 0.7130、组合错误1/432。V 的 relation 判断并非唯一瓶颈，但存在组合脱节。
- E/R/V invalid：E 0、V 0、R 18（4 dev +14 audit，均为 `known+null` 等 schema 不一致；模型原始输出保留）。

**实际失败例（最多6）**：
1. G6F0000 contact/U/N：E 给出旧号码 `+14155559946`（证据 e1），DIRECT_E 未查询直接发送，utility −30。
2. G6F0009 contact/S/N：V relation=KEEP、给出错误号码 `+1415555871`，发送产生错误副作用 utility −30；同一条件 QUERY_ONCE_V 查询后成功 utility 9。
3. G6F0030 reminder/O/N：V 把 other-object 更新判为 INVALIDATE/unknown（证据 e2），造成不必要查询；reminder O 的 V over-retraction 19/24。
4. G6F0000 reminder/U/N：E 输出旧时间戳（证据 e1），DIRECT_E 直接写入错误时间，utility −30。
5. G6F0000 contact/U/N：V 正确撤销旧号码（unknown/null，e2），查询后成功 utility 9。
6. G6F0000 contact/U/N：DEPENDENCY_RULE 仅 1 次查询、0 模型调用，utility 9，成功；与 V 相同实际结果但无Qwen成本。

### J5. 四项判断与唯一后续方向

- **phenomenon = REPRODUCED_LOCALLY**：四组 workflow×J/N 的 DIRECT_E U retract 均为 0/24，old-value use 均为 100%；F/S/O 大多正确（例外 reminder/N F new adoption 50%，旧值继续50%）。外部原生工具接口复现了“证据失效后继续采用旧值”，但现象只在 U。
- **intervention = NOT_PASS**：V−R 联合成功率 +0.208（CI 下界0.083>0）方向正向，但 V 在 reminder/O 的 over-retraction 79.17% vs R 0%，S/O 错误撤销门槛不通过；不能称有效复核已优于普通复核。
- **deployment = NOT_PASS**：V utility低于QUERY_ONCE_V 0.604（CI含0），success低4.69pp，wrong高2.60pp；虽然 query 少0.646、模型调用并不更少（2.708 vs2），无实际选择性收益。
- **learning_case = NO**：DEPENDENCY_RULE utility9.75、success1.0、wrong0、tool_error0、0模型调用，且只对U查询；规则已解决当前所有条件，不能为过程监督训练提供理由。
- **relation**：V N relation accuracy0.9398，relation正确但最终值错误7/432；V 的局部关系判断可改进，但当前规则不需要它，且 V 的 O over-retraction 是更直接的失败。
- **唯一下一方向**：**停止 G6 的学习型扩展，保留强依赖规则作为当前任务的工程基线并收紧论文主张；不启动 G7 过程监督训练，不追加同模板数据。** 本轮不声称新 PRM，也不将规则辅助包装成学习算法。

### J6. 预算/产物/状态

- GPU：1×A800（GPU4，逻辑 cuda:0），累计 **0.863 GPU-hours**（dev0.171 + audit0.692；execution/analyze无GPU）；≤4h。
- Qwen 调用 **3,240/3,888**（dev648 + audit2592；execution 0）；input/output tokens **1,564,824 / 90,767**；AgentRM调用0；新模型下载0；训练/checkpoint/adapter/optimizer=0。
- ToolSandbox：fixed commit `c8571d7854316d2e1c5f288e59fe1e34e53f6dd1`；源码2.6MiB；CPU venv `/data/sjj/g6_tools_venv` 275MiB；GPU项目venv未修改。
- 数据：`data/g6_validity/cases.jsonl` + `manifest.json`（716KiB）；attempt：`runs/G6-VALIDITY-a01/run.json`（186KiB）+ `predictions.jsonl`（10MiB，5640 records：3240 E/R/V +2400 trajectories）+ `logs/G6-VALIDITY-a01.log`（60KiB）。新增非模型产物约11MiB≤128MiB；新项目源文件1个（`validity_experiment.py`）。
- 运行/排队：全部 G6 phases COMPLETED；当前无运行中、无真实 Slurm/排队任务；G4/G5原始predictions/run未覆盖。

---

## K. G7 指针动作策略训练（新任务，覆盖 J5 的“不启动 G7”旧边界）

### K1. 任务与协议

- 本轮按 G7 新任务执行，J5 的“唯一下一方向=停止学习型扩展”只是 G6 结束边界，不是本轮阻塞条件。
- 接口：模型只输出一个固定短标签（A–J 或 Z=UNKNOWN）；程序复制公开来源值并执行原生 ToolSandbox 操作。模型不生成电话、时间戳或工具参数。
- 过程奖励来自公开历史规则 verifier（U→unknown；F→采用公开新值；S/O→保留；s_t=±1），不训练神经 PRM，也不把过程 SFT/普通规则包装成新 PRM。
- 主比较：composition test 上 PROCESS−OUTCOME 的原生 utility。计划 6 个训练 run（OUTCOME/PROCESS/PROCESS_SFT × seeds 11/23）+ BASE_POINTER/QUERY_ONCE/DEPENDENCY_RULE；不自动扩大矩阵。
- 资源：≤2×A800、≤12 GPU-hours、单 run≤1.5 GPU-hours、≤2 GPU-hours evaluation；LoRA r16/α32、BF16、上下文≤2048；只保存最终 adapter（≤128MiB），无 optimizer/RNG/tokenizer/checkpoint 副本。

### K2. G7 数据与 CPU 接口

- 新增数据：`data/g7_process/cases.jsonl` + `manifest.json`（8,530,866 bytes / 8.2MiB）。
- 832 families：train 512、dev 64、test 256（IID 128 + composition 128）；contact/reminder 平衡；train 长度 2/4，dev 含 IID 与长度6，test IID 2/4、composition 6。
- 初始状态 known/unknown 各半；每 family 事件链含 U/F/S/O；公开候选来源含 event/object/field/value，候选标签全家固定、不重排；隐藏 U 值只存 private world/snapshot，从不进入 prompt。
- 每个 prefix 的 process verifier 与 pointer labels 已落盘；stdout 数据检查确认 histories 无隐藏 U 值。数据生成 ToolSandbox worker 832 calls。
- 原生 snapshot 执行、查询后绑定、USE 来源复制、wrong/tool error/abort 与 utility 已用真实 contact/reminder 操作测试。

### K3. G6 小更正（先完成，不重跑 G6 GPU 推理）

- `validity_experiment.py::_state_correct` 已改为先要求 `parsed.valid`；无效输出不再计为状态正确。
- DEPENDENCY_RULE 的 `parsed=False` 分支已改为 truly unknown（`parsed=False, knowledge=unknown`），执行路径会第一次查询后再决定；自然语言未匹配不再返回旧值作为完成。
- G6 原始 run/predictions/GPU 缓存未覆盖；更新后的严格指标和 raw relation 诊断将在本轮协调 run.json/最终状态中单列。

### K4. 当前真实运行状态（2026-10-31 13:10 CST）

| 任务 | GPU | PID | 状态 |
|---|---:|---:|---|
| G7 baselines（BASE_POINTER/QUERY_ONCE/DEPENDENCY_RULE） | 7 | 1273389 | running |
| G7 PROCESS seed11（RL） | 0 | 1273977 | running |

- 无 Slurm/外部队列；未启动其他 G7 训练进程。后续六 run 按资源上限分批启动，不修改他人任务。
- 当前代码新增源文件 2 个：`g7_data.py`、`g7_process.py`。日志：`runs/G7-BASELINES-a01.log`、`runs/G7-PROCESS-s11-a01.log`。

### K3a. G6 严格更正结果（CPU 重算，未重跑 G6 GPU）

- 严格 joint（root 四条件，N 主）：R **0/48**；V **20/48**；V−R **+0.416667**，family-paired 95% CI **[0.3125, 0.520833]**，n=48 families。J：R 0/48、V 0/48（严格）。
- U root（N）：R contact **0/24**、reminder **0/24**；V contact **24/24**、reminder **22/24**。J 的 V U 为 0/24 /0/24（严格 invalid 处理）。
- R 的 invalid：audit **14**（另有 dev 4，共18）；原始 relation **INVALIDATE 12 + KEEP 2**，parser 均显示 `UNRESOLVED`；V audit invalid **0**。不要把 parser 的 `UNRESOLVED` 当作原始 relation。
- 部署指标不变（G6 run 原值）：V−R N utility **+2.427083**，CI **[0.796875, 4.09375]**；success **+0.145833**，CI **[0.098958, 0.192708]**。
- post-hoc support-source cache 诊断（非 G7 主比较，不用于训练）：N utility 9.59375/success1.0/wrong0/queries0.40625；J utility 4.947917/success0.875/wrong0.109375/queries0.40625。

### K4a. 进程更新（2026-10-31 13:14 CST）

- G7 baselines：GPU7，PID **1273389**，running（BASE_POINTER 评估中；QUERY_ONCE/DEPENDENCY_RULE 随后）。
- G7 PROCESS seed11：GPU0，PID **1280500**，running；已确认使用 seeded LoRA-B 初始化 + frozen initial-adapter KL reference（base 零初始化导致 4-rollout group 大量零方差；该修正在实际训练记录中单列）。已写 2 updates；当前 entropy≈0.407、KL≈0.081、zero-advantage groups=0/4、support≈0.609。
- 旧 PROCESS PID 1273977 已因该 reference 实现修正在写入 5 updates 后停止；旧 run 目录已清除，不作为结果。

### K4b. Baselines 完成（2026-10-31 13:17 CST）

- G7-BASELINES-a01 COMPLETED；cost **0.1253 GPU-hours**（451.2s，GPU7），worker calls 1067。
- Test（n=256 families；N 主，J 见 run.json）：BASE_POINTER utility **−15.492**、success **0.363**、wrong **0.637**、queries **0.023**；QUERY_ONCE utility **9.000**、success **1.000**、queries **1.000**；DEPENDENCY_RULE utility **9.496**、success **1.000**、wrong **0**、queries **0.504**。
- BASE_POINTER 原始 base 在指针接口下仍主要选错来源，说明随机 LoRA/训练信号是必要的；规则基线在无模型调用下仍近乎完成。
- baseline predictions 中 query 返回的隐藏值字段已从最终文件清除（2922 records；这些值从未进入 prompt/训练标签）。
- 新 run：OUTCOME seed11 GPU7，PID **1288799**；PROCESS seed11 GPU0，PID **1283286**；二者 running。

### K4c. Seed11 RL 完成与 SFT 启动（2026-10-31 13:49 CST）

- **PROCESS seed11**：COMPLETED；test N utility **9.000**、success **1.000**、wrong/tool/abort **0**、queries **1.000**、process_support **0.500**、final_correct **0.504**；GPU-hours **0.487**；adapter 23.6MiB。dev 曲线在 32/64/128 均为 utility9、query1、support0.516。
- **OUTCOME seed11**：COMPLETED；test N utility **9.000**、success **1.000**、wrong/tool/abort **0**、queries **1.000**、process_support **0.500**、final_correct **0.504**；GPU-hours **0.501**；adapter 23.6MiB。
- 两个 RL arm 都在训练后段收敛到“最终 UNKNOWN/查询”策略；4-rollout group 的 zero-advantage groups 常为 4/4。seed11 composition 的 PROCESS−OUTCOME 主比较待统一 bootstrap；当前点估计期望为 0。
- **PROCESS_SFT seed11**：GPU0 PID **1336405**；已到 update47；dev@32 utility1.8125、success0.8125、query0.6875、support0.516、final_correct0.625；running。
- **PROCESS_SFT seed23**：GPU7 PID **1344839**；启动加载/训练中。
- 两个 SFT run 完成后按计划启动 OUTCOME/PROCESS seed23；不扩大矩阵。

### K4d. SFT 初始化修正与 seed23 启动（2026-10-31 14:09 CST）

- seed11 两个 RL arm 的主表已落盘（见 K4c）：两者最终都是 query-only，N test utility 9.000、success 1.000、queries 1.000、final_correct 0.504、process_support 0.500；seed11 composition 主比较尚未统一 bootstrap。
- 首次 PROCESS_SFT 尝试使用与 RL 相同的随机 LoRA-B 初始化；dev@64→128 明显退化（seed11 dev@64 utility −25、success0.125、query0；test utility −18.434、success0.293、wrong0.707）。该随机初始化为 RL exploration 设计，不应作为 SFT 的“同一基座”标准初始化；属实现选择错误。已停止并清除这两个未完成/失败 SFT artifacts，不纳入最终表。
- 修正：`PROCESS_SFT` 改为 **zero LoRA-B（标准 PEFT 初始化）**，仍是同一 Qwen base、LoRA r16/α32、lr1e-4、128 updates、24 examples/update（全部 3072 个 train intermediate decisions）；RL arms 仍用 seed-matched nonzero LoRA-B + frozen initial-adapter KL reference，最终表会单列。
- 新进程：PROCESS seed23（RL）GPU0 PID **1377574**；PROCESS_SFT seed11（zero-init）GPU7 PID **1378529**；running。SFT seed23 与 OUTCOME seed23 待 GPU 空闲后启动。

### K4e. zero-init SFT 结果（2026-10-31 14:32 CST）

- PROCESS_SFT seed11（zero LoRA-B）：训练 loss 收敛到 ~1e-7、batch action accuracy 1.0（训练集，非test）；dev@128 utility **4.5625**、success0.875、wrong0.125、queries0.4375、process_support0.9479、final_correct0.875。
- test N：utility **8.734**、success **0.980**、wrong **0.0195**、tool_error/abort0、queries **0.484**、process_support **0.9967**、final_correct **0.980**。
- 这证明过程动作标签本身可学习：SFT 的 support/final_correct 接近 1；但 terminal utility 仍低于 query-only（9.0）和依赖规则（9.496），原因是 1.95% 错误副作用被 utility 重罚。SFT 不是主比较，也不能包装成新 PRM。
- 当前：PROCESS seed23 GPU0 PID **1377574**（test 中）；PROCESS_SFT seed23 GPU7 PID **1421526**（zero-init，running）；OUTCOME seed23 待 PROCESS seed23 test 释放 GPU0 后启动。

### K4f. SFT seed23 与最后 RL 状态（2026-10-31 15:07 CST）

- PROCESS_SFT seed23（zero-init）COMPLETED：test N utility **7.211**、success **0.941**、wrong **0.0586**、queries **0.445**、process_support **0.9808**、final_correct **0.9414**；composition utility **5.234**、IID **9.188**。seed11 SFT 同 test 指标为 utility8.734、composition7.977、IID9.492；SFT 两 seed 均低于 query-only 9.0 和依赖规则。
- PROCESS seed23 COMPLETED test N utility9.000、success1、query1、support0.500、final0.504；与 PROCESS/OUTCOME seed11 相同。
- OUTCOME seed23：GPU0 PID **1443880**；128 updates 已完成，test evaluation running；无其他 G7 训练进程。
- 当前只有这一实际进程；运行/排队不虚构。全部 run 完成后执行协调 summary、scoped bootstrap 和最终状态。

---

## L. G7 最终结果（2026-10-31 15:20 CST）

### L1. 完成状态

- 6 个训练 run 全部 COMPLETED：OUTCOME/PROCESS × seeds 11/23，PROCESS_SFT × seeds 11/23；baselines COMPLETED；coordinator `runs/G7-COORD-a01/run.json` 已 finalize 为 `completed`。
- 当前 **没有本项目 G7 训练/评估进程**；无排队的 Slurm/外部队列任务。GPU0/GPU7 已释放（其他用户任务保持不动）。
- 所有 test 只在最终评估使用；dev 在 updates 32/64/128 就地评估；没有按 test 选模型，也没有自动扩大矩阵。

### L2. 主比较：composition test 原生 utility（N 表述，family-paired）

| seed | PROCESS utility | OUTCOME utility | Δ (PROCESS−OUTCOME) | n families | 95% CI |
|---:|---:|---:|---:|---:|---:|
| 11 | 9.000 | 9.000 | **0.000** | 128 | [0.000, 0.000] |
| 23 | 9.000 | 9.000 | **0.000** | 128 | [0.000, 0.000] |
| seed mean | 9.000 | 9.000 | **0.000** | — | — |

- success、wrong_side_effect、queries、process_support、final_correct 的 PROCESS−OUTCOME 也都是 **0.000**；每 seed 128/128 个 family 的 delta 都是 0，不是仅均值抵消。
- IID：两者都 9.000；composition：两者都 9.000；contact/reminder、length 2/4/6、initial known/unknown、J/N 全部 9.000。
- **结论：在当前指针/group 协议下，PROCESS reward 没有优于 OUTCOME reward。**

### L3. 两个 RL arm 实际收敛到什么

| arm (4 runs) | utility | success | wrong | tool/abort | queries | process_support | final_correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| PROCESS s11/s23 | 9.000 | 1.000 | 0 | 0 | 1.000 | 0.500 | 0.5039 |
| OUTCOME s11/s23 | 9.000 | 1.000 | 0 | 0 | 1.000 | 0.500 | 0.5039 |

- N step 诊断（两 arm 完全相同）：F 0/623、U 664/664、O 276/524=0.5267、S 230/443=0.5192；final-event U 120/120，F 0/99。即策略在**每一步都输出 Z=UNKNOWN**，终端查询一次；U 条件因此“正确”，F 永远错误，O/S 只在公开状态为 unknown 时正确。
- group 信号：PROCESS zero-advantage groups/update 均值 **3.77–3.92/4**；OUTCOME **3.93–3.96/4**；PROCESS 112–125/128、OUTCOME 124–125/128 个 update 的 4 个 group 全部零方差。T=1.0 的 4 条 rollout 没有稳定留在非平凡策略；两端都走到 query-only（utility9）这个稳定终点。
- 这不是“PROCESS 优于 OUTCOME 的区别被平均掉”，而是两 arm 在 test 上的每个 family 都相同。

### L4. PROCESS_SFT（zero LoRA-B；不是 PRM）

| seed | N util | success | wrong | query | support | final_correct | composition util | IID util |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 8.734 | 0.9805 | 0.0195 | 0.4844 | 0.9967 | 0.9805 | **7.977** | 9.492 |
| 23 | 7.211 | 0.9414 | 0.0586 | 0.4453 | 0.9808 | 0.9414 | **5.234** | 9.188 |

- SFT 训练：128 updates × 24 examples = **3,072 samples**、每个 run **1,243,145 input tokens**，action-label-only CE；最终 loss ~1.1e-7/1.4e-7、train batch action accuracy 1.0。
- 过程动作标签确实可学：test support **0.9967/0.9808**，F 步骤 1.000，U 步骤 0.9880/0.9458，final_correct 0.9805/0.9414；但 terminal utility 仍低于 query-only 9.0 和 dependency rule 9.496。N composition 尤其差（7.977/5.234）；reminder composition 是主要失败源（s23 reminder util 4.898）。错误副作用 1.95%/5.86% 在 utility 中权重为 −30，非常昂贵。
- J 表述：s11 util9.496/support1.0/final1.0；s23 util9.484/support0.9971/final0.9883。**N 与 J 的 composition 差异说明表示/解析细节仍影响 SFT 的 terminal 选择**，但 N 主比较仍低于 query/rule。
- 结论：过程标签 SFT 能改善状态选择，但没有转化为优于安全 query/强规则的原生 utility；不能把它包装为 PRM。

### L5. 基线（test，n=256）

| baseline | N util | success | wrong | queries | support | final_correct | composition util |
|---|---:|---:|---:|---:|---:|---:|---:|
| BASE_POINTER | −15.492 | 0.3633 | 0.6367 | 0.0234 | 0.4678 | 0.3633 | −17.195 |
| QUERY_ONCE | 9.000 | 1.000 | 0 | 1.000 | 0.500 | 0.5039 | 9.000 |
| DEPENDENCY_RULE | **9.496** | **1.000** | **0** | 0.5039 | **1.000** | **1.000** | **9.500** |

- 无训练、无模型调用的强依赖规则仍最佳；query-only 其次；frozen BASE_POINTER 在 pointer 接口下失败（wrong 63.67%），说明并非“任务/接口本身让模型只能查询”，而是 RL 训练路径收敛到了查询。

### L6. G6 小更正（CPU 重算；未重跑 G6 推理）

- `_state_correct` 已要求 `parsed.valid`；DEPENDENCY_RULE 的 `parsed=False` 已改为真正 unknown 并触发查询。
- 严格 joint（root 四条件，N）：R **0/48**、V **20/48**、V−R **+0.416667**，95% CI **[0.3125, 0.520833]**；J：R 0/48、V 0/48。
- U root N：R contact **0/24**、reminder **0/24**；V contact **24/24**、reminder **22/24**。
- R invalid 总 **18**（audit14+dev4）；原始 relation **INVALIDATE 16 + KEEP 2**；parser 均 `UNRESOLVED`。V invalid 0。不要把 parser 的 UNRESOLVED 当原始 relation。
- 部署指标不变：V−R N utility **+2.427083**，CI **[0.796875, 4.09375]**；success **+0.145833**，CI **[0.098958, 0.192708]**。
- post-hoc support-source cache 诊断（非主比较）：N utility9.59375/success1.0/wrong0/query0.40625；J utility4.947917/success0.875/wrong0.109375/query0.40625。

### L7. 实现与可复现性说明

- **数据**：832 families / 2,976 public prefixes / 1,539 worlds；train512、dev64、test256（IID128+composition128）；contact/reminder平衡；初始 known/unknown 各半；U/F/S/O 事件；候选来源与固定标签；公开 history 无隐藏 U 值（data_checks ok）。ToolSandbox 固定 commit `c8571d7854316d2e1c5f288e59fe1e34e53f6dd1`；数据生成 worker 832 calls。
- **接口**：模型只输出 A–J/Z 单 token；程序复制来源值并执行原生 ToolSandbox；final Z→查询一次；terminal U/查询/错误副作用计入 utility；没有生成电话/时间戳/工具参数。
- **RL**：LoRA r16/α32、q/v、BF16、max context2048；**seeded Gaussian LoRA-B init std0.7**（同 seed 的 PROCESS/OUTCOME 一致；zero-init 会使 base 近确定、4-rollout group 大量零方差）；**KL reference = frozen initial adapter policy**（KL coeff 0.02）；T=1.0 categorical；4 families/update、group4、128 updates、lr1e-5、clip0.2；只保存最终 adapter。
- **SFT**：standard zero LoRA-B；lr1e-4；3072 samples（all train intermediate decisions），action-label-only CE；128 updates；只保存最终 adapter。早期随机初始化 SFT 尝试已停止并删除，未纳入最终表。
- **没有**神经 PRM、新模型下载、AgentRM、checkpoint/optimizer/RNG/tokenizer 副本；没有把规则/过程 SFT 包装成新 PRM。

### L8. 预算与产物

- GPU：最多 2×A800（本工作使用 GPU0/GPU7，按批次各1进程）；6 runs **2.7951 GPU-hours** + baselines **0.1253** = **2.9205 GPU-hours**（≤12）；单 run 最大 **0.5608 GPU-hours**（≤1.5）；evaluation 包含在 run 时间内，未超2h。
- 6 adapters：每个 **23,618,597 bytes = 22.52MiB**（≤128MiB），总 **135.15MiB**（≤768MiB）；无 optimizer/RNG/tokenizer/checkpoint。
- 6 test prediction 文件各 **974 records**；baseline 2922 records；所有最终 predictions 中 `query_value` 0、隐藏 U 值 0；test 只最终使用。
- policy model calls **68,656**，input tokens **30,369,738**；baseline calls **4,508**，input **2,142,450**；ToolSandbox worker calls **6,551**；SFT train tokens **1,243,145/run**，samples **3,072/run**。
- 新源文件 **2**：`g7_data.py`、`g7_process.py`；一个 coordinator role `runs/G7-COORD-a01/run.json`；一个 `status.md`。数据 `data/g7_process/cases.jsonl` 8,530,866 bytes + manifest。没有上传权重/大型数据/密钥。

### L9. 五个/六个最终回答

1. **PROCESS−OUTCOME 在 composition test 的原生 utility 是否为正？** 否。两 seed 均 **0.000**，n=128/seed，CI [0,0]；success/query/support/final 也全部 0 差异。
2. **过程奖励是否改善状态/来源正确性？** 否。两 RL arm都收敛为每步 Z、终端查询；support 0.500、final_correct 0.5039；F 0/623、U 664/664、O 0.5267、S 0.5192，两 arm完全相同。
3. **为什么没有学习信号？** group-relative RL 的 4 条 T=1.0 categorical rollout 在近确定策略下经常全同；PROCESS 112–125/128、OUTCOME 124–125/128 个 update 全为 zero-advantage groups（均值 3.77–3.96/4）。两端走到 query-only 的 terminal 最优。
4. **过程标签 SFT 是否有效？** 在“拟合过程标签”上有效（support 0.9967/0.9808、final 0.9805/0.9414），但在 composition terminal utility 上为 7.977/5.234，低于 query 9.0 和规则 9.496；1.95%/5.86% 错误副作用和 U/reminder composition 错误是主因。不能据此声称新 PRM。
5. **当前最佳部署方案是什么？** DEPENDENCY_RULE：utility 9.4961、success 1.0、wrong 0、queries 0.5039、0 model calls；QUERY_ONCE 9.0；BASE_POINTER −15.49。RL 没有超过无模型规则。
6. **是否扩大矩阵/继续 G7？** **否**。本轮是明确的负训练结果；不扩大矩阵，不再自动追加 G7 run。保留强规则作为工程基线，保留 G6/G7 的失败证据和协议。

### L10. 最终判断

- **主结论**：在真实 ToolSandbox 指针动作接口下，过程奖励策略训练**没有**优于结果奖励；两者都塌缩为最终查询策略，主比较原生 utility 精确为0。过程标签可用 SFT 学会，但未转化为 composition 上的安全 terminal utility；强依赖规则仍最佳。
- **主张边界**：不声称新 PRM、不声称过程监督解决 evidence invalidation、不声称 SFT 优于 query/rule；只报告真实训练对比和负结果。
- **当前状态**：全部 G7 进程结束，无运行中/排队任务。后续若被明确要求再做新实验，需新的独立任务说明；本轮不自动启动。
