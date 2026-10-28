# 服务器实验 Agent 执行 Prompt

你负责执行“信息条件一致过程监督”的第一轮实验。工作目录是本文件所在的 information_feasible_prm 项目。先阅读 README.md 第12节与 status.md，核实已有代码和结果后继续。当前已完成的CPU结果可以复用；不得把尚未发生的GPU训练、服务器排队或模型效果写成完成。

## 任务范围和执行终点

本次范围仅 G0/G1：环境检查、有限状态gold与数据冻结、教师审计、参考完备性/同π MC对照、统计分析；最多一次20-step训练profile，且不保存模型。满足依赖和预算的常规步骤自行完成，不为每一步反复确认。

G2正式DPO、学习型PRM、在线RL、第二个模型、大规模外部benchmark、额外付费API均不属于本次执行范围。完成G1或达到预算后给出结果，不自动扩大任务。结果可以是no-go；不能为得到正面结果不断换提示、样本、种子和指标。

## 第一件事

1. 定位项目根目录，只查看本项目及必要的服务器资源信息；不搜索其他用户目录或凭据。
2. 读取status.md。核对已登记Job ID/PID及输出，消除上次提交是否成功的不确定性，再决定是否提交新任务。
3. 识别是否有Slurm或已有队列。检查本项目可用的GPU、驱动、Python环境、共享模型缓存、磁盘与quota。记录真实结果；缺少访问写BLOCKED，不编造服务器资源。
4. 找到Qwen/Qwen3-4B-Instruct-2507的唯一共享缓存，解析并冻结实际commit。优先复用已有环境；确需环境变更，在status登记一次，不为每个实验建立环境。不重复下载模型。不得重设HOME、CODEX_HOME。
5. 启用PYTHONDONTWRITEBYTECODE=1。已有CPU产物hash一致时不反复生成副本。只对代码变更带来的具体风险做必要验证。

## 第一轮研究协议

- 360个独立根case，三机制各120：静态诊断、动态证据失效、执行进度不确定。
- 每机制“信息有明确价值/无益/边界”各40；每层8 dev+32 audit，即72 dev、288锁定audit，split seed=17。
- 每根case只取一个prefix，固定三个候选动作。前两机制2个隐藏状态、第三机制4个；按正确posterior完整枚举。动态噪声按冻结转移模型处理，不能默认各世界等权。
- H仅合法历史；P额外给当前隐藏状态；PR再给参考动作；PR-IC再加信息可实施性约束。
- 预先选72个audit case（每机制24）做PR-full与PR-full-IC。full只补全同一参考规则下等价/合格动作，不从history oracle偷偷添加正确答案。
- PR沿用目标流程的动作匹配/评分语义。更换原论文模型或提示时标为“VPR-style适配”，不得宣称已忠实复现原论文。
- 主机制比较固定同一个history-only continuation π；gold是精确Q^π。普通MC也rollout同一个π，有解析controller就用CPU。冻结LLM continuation另列Q^π_LLM，不能把其执行能力差距当作监督语义收益。
- 先按真实posterior聚合候选分数/实际训练目标的偏好概率，再计算其诱导的history-only选择；禁止平均各隐藏世界的特权argmax。最高分并列预先约定均匀选择。
- 主指标是case级标准化regret，归一化跨度事先冻结；唯一主对比为PR相对PR-IC的配对regret降低。根case分层bootstrap 2,000次，seed=23。隐藏世界、候选动作、改写、rollout不增加独立n。
- dev最多两版提示，格式有效率目标>=98%；audit开始后冻结提示、阈值、主要指标、样本与候选顺序。不因为无效输出反复采样至成功；无效率单列，按预注册fallback纳入主分析并给敏感性结果。
- 详细工程gate在README第12.4节：15%的必要信息case存在r>=0.05误排且跨至少两机制；平均配对改善>=0.01且95%区间支持正向；参考完备性不能解释全部效应；继续新方法前需指出简单修复后仍存的具体缺口。这些是项目门槛，不是统计学定理。

严禁以下推断：隐藏世界评分不同就证明偏差；把真实伤害标签抹掉；把Q*与弱策略Qπ的差距归为信息约束；把三个换皮模板称机制泛化；把toy、DPO首动作或20-step profile称完整PRM/Agent安全结果。

## 文件允许列表

本次只新增两种源文件：audit_cases.py、audit_teacher.py。复用并按需修改已有core/generation/evaluation模块；不要生成train_v2.py、final_final.py或每个实验一份脚本。提示模板放在audit_teacher.py中。若确需超出这两个新源文件，先在status写清必要性和替代方案，完成不受影响的工作；不要自行扩张工程。

持久输出仅允许：
- status.md：唯一实验台账，由协调者更新。
- data/audit/cases.jsonl与data/audit/manifest.json：一份共享冻结数据，不按模型/种子复制。
- runs/<ATTEMPT_ID>/run.json：集中参数、版本、代码/数据hash、起止时间、指标、错误、资源、文件清单。
- 同目录metrics.jsonl（训练，每10 optimizer steps及阶段事件）或predictions.jsonl（推理/审计，每次调用一行）。
- 沿用evaluate_actions.py时允许其summary.json作为唯一汇总，不再复制为run.json。
- logs/<ATTEMPT_ID>.log：stdout/stderr合并，一个attempt一个日志，放在训练output-dir外。
- 仅以后明确进入正式训练并指定--save-final-adapter时，允许一个adapter目录，只有adapter_model.safetensors和adapter_config.json两个文件。

可为原子写入使用一个临时.tmp文件，完成后替换正式文件，不保留版本副本。不得按样本/step生成文件，不创建重复csv/json镜像、每run Markdown报告、时间戳目录副本、自动W&B/TensorBoard目录。不要先创建run目录并写metadata或日志，再传给拒绝非空目录的训练入口；提交metadata先写status，日志放logs，运行产物由脚本自己创建。

项目已有data/results文件是历史产物，保留并复用；不因新白名单而删除。不要删除或压缩用户既有文件、其他项目缓存或其他人的任务。已失败attempt的记录也必须保留。

## checkpoint规则

本次G0/G1模型保存数必须为0。20-step profile不传--save-final-adapter。
使用现有train_dpo.py：save_strategy=no；禁止Trainer自动save_model/save_state/checkpoint；禁止额外save_pretrained调用。
禁止checkpoint-*、全量基座权重、optimizer、scheduler、RNG、trainer_state、tokenizer副本，以及best/last多份保存。
以后正式run每次最多一个最终LoRA adapter，两个文件合计<=512MiB，使用--max-save-mib 512；正式矩阵最多12个adapter。
当前训练resume_supported=false。最终adapter不能冒充完整断点续训状态。训练失败重试用新attempt从头跑，不覆盖旧目录，不从最终adapter继续并伪称同次resume。审计推理则以case/condition/world/prompt/model/seed构成固定调用键，核对旧attempt中已持久化的调用记录；a02只执行尚未完成的固定调用，引用旧记录而不复制或重采样。这是推理记录去重，不是训练checkpoint恢复。

## 资源与尝试预算

- 硬件上限2×A800 80GB；最多2个单GPU运行任务，每张已分配卡最多1个；真实QUEUED任务最多4个。
- 初始G0/G1总计<=36 GPU-hours，包含profile、失败和重试。计入已用时间+运行剩余额度+已排队预留，不能超售预算。
- profile最多20steps、256train、64validation、1 GPU-hour；其质量不参与科学gate。
- 单审计/MC作业默认walltime<=4小时；先用dev吞吐估计。max_steps不限制加载/预计算，必须限制整个作业时间。
- 教师调用<=4,272、每次<=128新tokens；可选LLM-MC<=2,592次生成、每次<=64新tokens。总数跨所有attempt累计，包含无效输出、试跑和失败前已执行调用；重试不增加预算。执行后尚未落盘的未知调用按保守上界计入，不把它当免费重试。普通同π的CPU-MC不占GPU预算。
- 同一逻辑run最多a01和一次有明确修复的a02。确定性错误不原样重试；第二次失败BLOCKED。格式无效不是无限重试理由。
- G0/G1新增产物<=2GiB，覆盖runs、logs、data/audit中的本轮内容、失败attempt及临时空间；单独计账的共享模型缓存除外。单run非模型文件<=128MiB，包含对应logs文件；单日志<=10MiB，新数据<=128MiB。提交前预留运行/排队作业输出额度，不能仅统计已写文件。以后G2项目产物总上限10GiB。
- 共享模型缓存只允许一个底模revision，新增预算24GiB；超出时停止新增任务并说明，不自动删用户缓存。应用层检查不等于OS硬quota，status里要写真实执行方式。
- 达到任一预算就结束当前范围并报告覆盖率与剩余工作，不自动增加机器/模型/样本。

Slurm作业内保留调度器设置的CUDA_VISIBLE_DEVICES，不能改指向未分配物理卡。普通服务器只有确认两张卡属于本项目后才显式绑定0或1。不要取消、抢占或清理不属于本项目的进程。

## status.md强制维护

status.md固定四部分：
A. 已完成实验；B. 服务器排队/运行/待核验作业；C. 待执行计划；D. 预算、阻塞与阶段决策。

每条已提交attempt记录：RUN_ID、ATTEMPT_ID、阶段、依赖、状态、原始调度状态、Job ID/cluster或主机+PID+启动时间、GPU数、提交/开始/结束时间、last_verified_at、当前进度、命令/版本摘要、预算、证据路径、下一步。统一使用Asia/Shanghai并标时区；精确时间来自实际观察，不猜测。

状态定义：
- PLANNED：仅计划，未提交。
- READY：输入/资源/预算已核查，可提交但未提交。
- SUBMITTING：已登记提交意图，正在提交或结果不确定。
- QUEUED：有真实任务ID且调度器当前明确PENDING。
- RUNNING：调度器/本次进程有实际运行证据；记录loading/training等stage。
- VERIFYING：作业结束，正在核对退出码、指标和产物。
- SUCCEEDED：成功退出且预期步数/记录/保存政策均核验通过。
- FAILED/CANCELED/BLOCKED：必须写明证据或明确原因。

只允许一个协调者写status。子agent/训练进程不直接编辑；协调者串行在同目录用.status.md.tmp完整写入后原子替换。不能创建status_backup、status_v2或每run的status.md/status.json；允许且要求的run.json是该attempt唯一集中运行记录。

每次提交前先登记唯一attempt、完整命令、版本、预算、输出路径和SUBMITTING；持久写入成功后再提交。Slurm提交必须把唯一project_tag/ATTEMPT_ID写入job-name或comment，并预先把该标签登记到status。使用sbatch --parsable并为当前不可resume的任务指定--no-requeue，记录可能的jobid;cluster，再查询实际状态；如集群强制重排，应先登记并核对同一job的重启；非预期重启不提供额外尝试或预算，不直接继续执行或提交副本。提交断线时按attempt/job-name/comment和时间窗口查询squeue/sacct；未排除已成功提交的可能性，禁止重复提交。squeue查不到不能直接判成功，用sacct与产物核对。

无调度器时等待启动的计划不叫QUEUED；nohup/tmux任务须验证主机、PID和进程启动时间。日志存在或GPU利用率本身不是本次实验运行证据。

必须在提交前、拿到回执后、开始运行后、失败/完成后、恢复连接时、结束会话前更新status。长任务在会话活跃期间每5–10分钟核验一次；查询失败不刷新last_verified_at，保留最后确认状态并另写错误。不要声称离线时会自动更新，不创建常驻daemon、cron或无限轮询。

当前没有服务器连接证据时应写“本项目已登记提交数0；服务器队列未查询”，不能写“服务器排队0个”。不得将计划表伪装成真实队列。

## 执行顺序

1. 完成G0环境检查、旧结果核对和必要CPU验证，更新status。
2. 实现受控audit数据和调用入口；先用24个dev case做解析/格式/成本试跑，再完成72 dev；同一提示版本下复用已完成的24个case，不能重复调用增加预算。最多一次提示修订。
3. 在manifest冻结数据/提示/模型版本、primary指标/gate、参考完备性子集与文件/调用预算。
4. 在资源和预算允许时提交固定audit条件，保持配对。不提前提交依赖尚未满足的后续阶段。
5. 完成固定对照、根case统计与负结果分析。若预算不足，保存已完成记录并报告缺失，不能只保留好看的case。
6. 如执行独立20-step profile，只用于工程评估，0模型保存。
7. 复核全部本项目job，核算实际/预留GPU时间与文件大小，更新status最后核验时间。
8. 最终报告：已完成/失败/正在排队或运行的ID；真实证据；核心效应及不确定性；预算使用；“继续/简单修复足够/证据不足”判断；下一阶段的明确问题。此处结束本次G0/G1任务。

