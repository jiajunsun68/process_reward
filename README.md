# 信息条件一致的过程监督：研究计划与最小实验包

更新：2026-10-28（文献调研截至2026-10-27）。研究对象是动态、部分可观测环境中的语言 Agent。建议以 2 × A800 80GB 为资源上限，先完成可证伪的小规模实验，再决定是否扩大为顶会投稿。

执行入口：[AGENT_PROMPT.md](AGENT_PROMPT.md)；唯一台账：[status.md](status.md)。详细执行计划、文件白名单和预算在第12节。本轮默认执行范围仅G0/G1；训练脚本已改为默认不保存模型。

## 1. 这个包完成了什么

- 已实际运行：有限状态 CPU 机制反例、2,772 个参数配置的期望枚举、11 项单元验证、两组严格匹配的合成偏好数据。
- 已编写但未运行 GPU：4B 模型的 BF16 + LoRA DPO 入口、显卡自检、首动作生成评估。GPU 脚本只完成语法检查，当前环境没有 PyTorch/CUDA；未访问用户的 A800 服务器。
- 尚未完成：现有 LLM 教师/PRM 的偏差审计、真实模型效果、学习型 PRM、在线 RL、完整语言 Agent 闭环、外部环境迁移。包中的 toy oracle 不代表对任何既有论文的复现。

**当前证据只证明人为构造的信息错配机制可以发生，不证明现有过程奖励普遍存在问题，也不证明这个选题已经达到主会录用标准。**

## 2. 主推研究问题

工作标题：**Information-Feasible Process Supervision for Partially Observable Agents**。

核心问题：给过程标签的教师可以看到模拟器隐状态、完整任务描述或未来轨迹，而学生只能看到当时的交互历史。教师在评估某一步之后的收益时，是否无意中让后续策略使用学生得不到的信息，从而低估查询、检查、复核的价值？能否在保持相同任务奖励和真实伤害成本的条件下，修复这种监督？

具体机制不是“所有隐藏信息都有害”。问题在于：老师推荐的后续决策是否能由学生当时的观察实施。同样的观察历史必须对应同样的动作分布，直到实际的新观察允许分支。这是运筹学的 nonanticipativity（非前视/信息适应）约束，属于已有理论，不能宣称发明了它。

可研究的贡献应是：真实过程监督流程的诊断；信息条件可审计的标注协议；在相同预算下有效、必要的修复方法；学习后安全、完成率和核验成本的改善。

## 3. 跨领域来源

| 来源 | 可借鉴的概念 | 对本项目的作用 |
|---|---|---|
| Pauker & Kassirer, NEJM 1980 [1] | 根据概率、检测成本/可靠度与后果决定不处理、先检测或直接处理 | 奖励应学到检查边界；检查越多并不越好 |
| Baron & Hershey, JPSP 1988 [2] | Outcome bias：已知结果会影响对同一决策质量的评价 | 用事后轨迹做步骤判断，需要区分当时可知与后来才知 |
| Brown, Smith & Sun, Operations Research 2010 [3] | 随机动态规划中的信息松弛与非前视约束 | 全知后续计划通常只是乐观界，不能直接当可实施动作价值 |

这些都是启发和数学基础。医学不是本项目的首个部署对象；首轮选择可精确控制的设备检修任务，避免把简化实验当成临床结论。

## 4. 最小机制例子与实际 CPU 结果

设备有 A/B 两种故障，先验各 0.5。第一步可跳过检查或花费 1 做完美检查。第二步只能根据实际获得的报告和历史选择修 A、修 B 或暂停；正确修理 +10，错误修理 -30，暂停 0。

| 第一动作 | 全知后续策略预测 | 可实施后续策略的期望收益 |
|---|---:|---:|
| 跳过检查 | 10 | 0 |
| 检查 | 9 | 9 |

不检查时，两种盲修的期望收益均为 -10，最优动作是暂停，收益 0；检查后可选正确维修，净收益 9。全知老师却假设即使没检查，第二步也能知道真实故障，所以错误偏好跳过。

这对应经典的“先对隐藏世界分别最大化再求期望”与“先按可见信息合并再决策”的区别。CPU 包通过枚举计算期望，没有训练语言模型。

已运行的参数网格共有 2,772 个配置，扫描先验、检测成本、可靠度、检查后故障翻转概率。严格偏好反转为 436/2,772（15.7287%）；包含约定平局规则的首动作差异为 565/2,772（20.3824%）。这些比例只描述人为选择的格点，不能解释成真实 LLM 的错误率。终端决策前直接揭示真实终端状态的控制组，2,772 个配置的价值差和部署 regret 都为 0。

有意加入能推翻“总是检查”的条件：先验很确定时直接处理可能更好；检测成本过高时暂停更好；可靠度 0.5 的检测没有信息收益；观察后状态发生大幅变化时，原检查可能失效。特别是当前两阶段 toy 中高翻转率可能消除检查价值，因此不预设动态性越强方法收益越大。

## 5. 截至本次检索的相近研究与创新边界

| 研究 | 已确认状态 | 与本项目的关系 |
|---|---|---|
| LEAP: Better than Your Teacher [4] | ICLR 2025 | 已讨论 privileged teacher 与学生 realizability，必须比较其思想对应的强提示/教师基线 |
| To Distill or Decide? [5] | NeurIPS 2025 主会 | 已说明最优 latent teacher 不一定是最适合蒸馏的 teacher，不能把这一点作为新发现 |
| A Theoretical Justification for Asymmetric Actor-Critic Algorithms [6] | ICML 2025 | 说明特权 critic 有合理用途，不能将所有特权信息判为错误 |
| VPR [7] | 本次确认至 arXiv v3，2026-10-12 | 已有 oracle 过程奖励和 posterior oracle；其工具任务附录提供了可具体审计的 privileged expert 流程 |
| AMV-Guard [8] | 2026-10-22 预印本 | 已有按动作后果选择复核，也有信息受限 exact planner；不能把运行时 VOI gate 当新贡献 |
| TWICE [9] | 2026-10-22 预印本 | 已研究共享下游噪声的反事实信用分配；控制运气与限制可用信息是不同问题 |

VPR v3 附录 D 明确：教师输入包括政策与交互历史，以及学生看不到的 task identifier、user scenario、evaluation criteria 和 reference resolution actions。论文按参考动作匹配生成反馈。这为第一轮审计提供了真实研究流程，但不构成“VPR 已被证明有错”的证据。应忠实复现其评分目的与输入，再分别消融隐藏任务信息和参考解决动作，并检测它是否压低实际有用的核验行为；仅发现评分随参考动作变化并不足以判错。

AMV-Guard 的信息受限 planner 是评价参照，不直接监督主方法。我们的候选空间在监督如何影响主动信息获取的学习，而非重新提出检查/执行/弃权的决策框架。

## 6. 拟研究的方法（尚未在包内完整实现）

1. 明确历史 h 中哪些是学生可见的；隐藏状态和未来反馈只用于环境演化、真实任务/伤害计分，以及离线评价。
2. 对一个前缀构造多个与历史一致的隐藏世界。先验和转移/观察模型必须来自训练环境或训练数据；不能任意均匀扰动后宣称校准。
3. 对候选动作执行后续模拟。同一可见历史下，后续 actor 不得读取隐藏世界编号或隐状态。只有工具真实返回不同观察后才允许策略分支。
4. 得到 Q^pi(h,a) 或规划近似 Q*(h,a) 后，形成动作偏好或相对优势。比较方法时必须明确、固定 continuation 的定义，不能让新方法用最优规划器而基线用弱 actor，再把差距归因于信息约束。
5. 从小型 oracle 偏好验证开始，再训练仅看历史与候选动作的 PRM。进一步研究共享可见历史的场景树与按不确定度分配模拟预算，检验其是否在相同标签误差下比普通历史 MC 更省成本。
6. 在在线训练中保留相同的终局任务奖励和实际伤害成本。过程价值只是信用分配/辅助监督，不因 agent 不知情而抹掉真实损害。

**普通历史条件 MC rollout 已经满足信息边界，是不可省略的强基线。** 若它用同等预算、同等后续策略解决了问题，就必须承认方法收益有限。信息受限优化和方差降低不是同一个贡献。

## 7. 第一周：先审计教师，不急着训练

建议 200–500 个前缀，包含至少三个机制不同的任务族：隐藏故障诊断、会失效的证据、需要联合满足的授权/安全条件。现阶段包里的 pump/cooling-unit/instrument 只是同一数学模型的改写，不能算三个领域泛化。

教师条件保持候选动作与任务目标相同：

- 历史信息教师：仅可见交互历史。
- 特权教师：沿用目标论文的输入与提示，额外可见其规定的隐信息。
- 特权教师 + realizability 提示：要求按学生可见信息评估，允许核验与暂停。
- 历史条件 MC：所有后续动作使用同一个合法历史 actor。
- 小型 exact information planner：仅作为可解任务的评价参照，拥有额外模型能力的组要单列。

至少记录必要核验被压低的比例、候选排序与可实施价值的偏差、检查成本敏感性，以及未来/隐状态消融后的变化。按完整 scenario 聚类统计，不能把同任务的多次采样当作独立任务。

进入下一阶段的条件：问题在正常教师流程中出现，跨至少两个不同机制，并且不是一句明确的信息边界提示就全部解决。若只在故意设错的 clairvoyant oracle 中发生，应停止以此主打新方法论文；CPU 反例作为教学与测试材料保留即可。

## 8. 学习实验与主会证据要求

阶段 A（当前代码支持）：同一 4B 模型、相同数据前缀与候选对，分别用 information / clairvoyant 标签做 DPO。它只检验第一步决策是否能从偏好中学到正确边界，不证明训练好了 PRM，也不证明在线 RL 稳定。

阶段 B：补充历史教师、realizability 教师和普通 MC 标签；控制样本数、token、rollout、训练步数及后续策略，至少三个随机种子。加入 LLM 解析 belief + 普通 planner，以及不训练/终局任务奖励基线，避免只击败不合理教师。

阶段 C：学习 PRM，再进行小规模在线优化；完整 actor 执行所有步骤，不能继续把终端推理交给 Bayes oracle。若希望声称 step-level credit，应把优势分配到对应 assistant 动作 token，屏蔽 user/tool token。把所有过程分数求和后交给只接收标量的 GRPO，不足以证明实现了逐步信用分配。

阶段 D：迁移到一个独立环境。首选 ToolSandbox 的本地可复现子集 [10]，或在 AgentDojo [11] 上显式定义核验工具/信息变化扩展。新增工具与扰动属于自定义协议，应和原 benchmark 结果分开报告。ToolSandbox 是 Findings NAACL 2025，AgentDojo 是 NeurIPS 2024 Datasets and Benchmarks track，不能都写成主会方法论文。

最终指标应同时包含：任务完成率、安全完成率、实际违规与损失、有效与无效检查成本、暂停/拒绝比例、格式错误、可实施 regret、训练及标注预算。不以低违规率单独取胜，因为全部暂停可以轻易降低违规。阈值在 validation 上选；按任务结构/规则组合划分测试集，保留未见信息失效机制，而不仅是换措辞。

主会潜力来自可复现、普遍的监督失效与有必要的修复，以及独立环境中的学习收益。经典 QMDP 反例、oracle 蒸馏、换名的 VOI、只赢一个故意全知的基线，都不够。若方法贡献弱但评测揭示系统问题，可考虑以严谨的评测/数据贡献定位，仍需跨模型与真实轨迹证据。

## 9. 双卡 A800 80GB 配置

默认模型为 Qwen/Qwen3-4B-Instruct-2507 [12]。4B 参数的 BF16 权重约 8 GB（十进制），这不是总训练显存。当前提供的实现使用 LoRA r16、q/v projection、dropout 0.05、microbatch 1、gradient accumulation 8、gradient checkpointing，最长序列 4,096 token。偏好样本包含两个 completion，不能只按单序列估算。

GPU 0 单卡训练；GPU 1 做独立推理/评测或另一随机种子。两卡不需要拼接成一块 160GB 内存。训练脚本要求每个进程只看一张卡，避免隐式 DataParallel。预计算冻结底模 reference log probability，不常驻第二个完整 reference 模型。首次运行只做 20 个 optimizer steps，记录峰值显存、token、耗时，再确定总预算。未实测前不承诺训练小时数。

建议顺序：20 步 profile → 两组一轮 DPO → 3 个 seed → 若审计和效用证据均成立，再投在线 PRM/RL。可以将第一轮研究分成约一周诊断、二至三周学习验证、后续独立环境和稳健性实验；这是工作安排，非实测耗时或投稿保证。

## 10. 文件与运行

| 文件 | 用途 |
|---|---|
| core.py | 两阶段环境与解析期望 oracle |
| run_probe.py | 机制反例、参数扫描、全可见控制 |
| generate_pairs.py | 严格匹配的两套偏好数据与 scenario 分割 |
| test_core.py | 11 项核心数值/数据边界测试 |
| train_dpo.py | 单卡 4B + LoRA 的第一动作 DPO |
| gpu_smoke.py | 本机 CUDA/BF16 小规模自检，无模型下载 |
| evaluate_actions.py | 模型首动作 + 相同 Bayes 终端控制器的期望收益评测 |
| data/ | 已生成的 synthetic JSONL；全部 metadata 与模型输入分离 |
| results/ | 已实际运行的 CPU 结果，非模型效果 |

### CPU 复现（仅标准库）

```bash
cd information_feasible_prm
python3 -m unittest -v test_core.py
python3 run_probe.py
python3 generate_pairs.py
```

两种标签各 4,768 个匹配 pair：train 1,830、validation 330、test 2,608。共 800 个完整配置，每配置两个文字变体；任一 teacher 为平局时，两组同时删除该 pair。test 较大是因为特意保留高 drift 与一个可靠度组合。所有同配置变体不跨 split，但当前只隔离数值范围/组合，没有实现跨任务机制测试。

### GPU 环境与 profile

在服务器的专用 Python 环境中，先安装与驱动匹配的 CUDA PyTorch，再安装 requirements-gpu.txt；参考官方安装页面 [13]。依赖面向 TRL 0.21.0 与 Transformers 4.55.x 的发布接口 [14]，不是最新版承诺，实际安装版本集中记录到 run.json，不额外生成 lock 文件。

```bash
python -m pip install -r requirements-gpu.txt
CUDA_VISIBLE_DEVICES=0 python gpu_smoke.py
CUDA_VISIBLE_DEVICES=1 python gpu_smoke.py
CUDA_VISIBLE_DEVICES=0 python train_dpo.py \
  --train-file data/information/train.jsonl \
  --eval-file data/information/validation.jsonl \
  --output-dir runs/information-profile \
  --max-steps 20 \
  --max-train-samples 256 \
  --max-eval-samples 64
```

检查 runs/information-profile/run.json 的 metrics 字段，包括状态、setup/reference 和 training peak allocated/reserved 显存。模型首次使用需要下载，离线服务器可先准备本地模型路径，再用 --model /absolute/model/path --local-files-only。正式实验应用 --revision 固定模型 commit；版本、数据hash、参数、计时和指标集中保存于 run.json。profile默认不保存模型。

### 两组各一轮 DPO

```bash
CUDA_VISIBLE_DEVICES=0 python train_dpo.py \
  --train-file data/information/train.jsonl \
  --eval-file data/information/validation.jsonl \
  --output-dir runs/information-s42 --seed 42 --epochs 1 \
  --save-final-adapter --max-save-mib 512
CUDA_VISIBLE_DEVICES=0 python train_dpo.py \
  --train-file data/clairvoyant/train.jsonl \
  --eval-file data/clairvoyant/validation.jsonl \
  --output-dir runs/clairvoyant-s42 --seed 42 --epochs 1 \
  --save-final-adapter --max-save-mib 512
```

脚本仅用 prompt/chosen/rejected，metadata 不送给模型；超过 token 限制会报错，不静默截断历史。偏好损失只覆盖 JSON 动作 token，历史与模板后缀不计。输出目录非空会停止，避免混淆运行。默认仅生成run.json和metrics.jsonl；正式运行显式开启后才保存一个最终adapter的两个文件，无周期checkpoint、tokenizer副本或完整续训状态。以上正式训练命令属于后续阶段，不在本轮G0/G1范围内。

### 首动作评估

先在 validation 上检查格式/参数；冻结设置后再运行完整 test。

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate_actions.py \
  --split validation --max-scenarios 24 \
  --output-dir results/base-profile
CUDA_VISIBLE_DEVICES=1 python evaluate_actions.py \
  --adapter runs/information-s42/adapter \
  --split test --output-dir results/information-s42-eval
CUDA_VISIBLE_DEVICES=1 python evaluate_actions.py \
  --adapter runs/clairvoyant-s42/adapter \
  --split test --output-dir results/clairvoyant-s42-eval
CUDA_VISIBLE_DEVICES=1 python evaluate_actions.py \
  --split test --output-dir results/base-eval
```

每个完整 scenario 默认只评一个措辞变体，greedy 最多 64 新 token。输出 predictions.jsonl 与 summary.json，分 regime 汇总。无效 JSON 单独计数，另外明确报告 invalid-as-safe-abort 聚合，不能只挑有效输出报告。

**这个评估中 LLM 只决定第一步，终端动作由同一信息受限 Bayes controller 选择；它不能证明 LLM 已学会完整闭环安全决策。**

## 11. 原始来源

[1] Pauker & Kassirer. The threshold approach to clinical decision making. NEJM, 1980. https://pubmed.ncbi.nlm.nih.gov/7366635/

[2] Baron & Hershey. Outcome bias in decision evaluation. JPSP, 1988. https://pubmed.ncbi.nlm.nih.gov/3367280/

[3] Brown, Smith & Sun. Information Relaxations and Duality in Stochastic Dynamic Programs. Operations Research, 2010. https://pubsonline.informs.org/doi/10.1287/opre.1090.0796

[4] Choudhury & Sodhi. Better than Your Teacher: LLM Agents that learn from Privileged AI Feedback. ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/1c60ed2b01120d383eebf12dc7a0e138-Abstract-Conference.html

[5] Song et al. To Distill or Decide? Understanding the Algorithmic Trade-off in Partially Observable RL. NeurIPS 2025. https://proceedings.nips.cc/paper_files/paper/2025/hash/81343f8ebe529de8d0bb654af7523184-Abstract-Conference.html

[6] Lambrechts et al. A Theoretical Justification for Asymmetric Actor-Critic Algorithms. ICML 2025. https://proceedings.mlr.press/v267/lambrechts25a.html

[7] Yuan et al. Verifiable Process Rewards for Agentic Reasoning. arXiv 2605.10325v3. https://arxiv.org/html/2605.10325v3

[8] Wang et al. When Is a Memory Safe Enough to Act On? Action-Conditioned Verification for Long-Lived Agents. arXiv 2610.31446v1. https://arxiv.org/html/2610.31446v1

[9] Same Luck, Different Choice: Twin-World Credit Assignment for Agentic Reinforcement Learning. arXiv 2610.31544. https://arxiv.org/abs/2610.31544

[10] Lu et al. ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities. Findings NAACL 2025. https://aclanthology.org/2025.findings-naacl.65/

[11] Debenedetti et al. AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents. NeurIPS 2024 Datasets and Benchmarks. https://proceedings.neurips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets_and_Benchmarks_Track.html

[12] Qwen3-4B-Instruct-2507 official model card. https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507

[13] PyTorch official installation history. https://pytorch.org/get-started/previous-versions/

[14] TRL 0.21.0 DPO documentation and implementation. https://huggingface.co/docs/trl/v0.21.0/dpo_trainer ; https://github.com/huggingface/trl/blob/v0.21.0/trl/trainer/dpo_trainer.py

[15] Slurm官方调度与状态文档：https://slurm.schedmd.com/sbatch.html ; https://slurm.schedmd.com/squeue.html ; https://slurm.schedmd.com/sacct.html

[16] Transformers 4.55.4保存策略：https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/trainer

## 12. 受控实验执行计划（2026-10-28）

本节是后续服务器执行的操作依据。交给实验 agent 的完整任务在 [AGENT_PROMPT.md](AGENT_PROMPT.md)，唯一实验台账在 [status.md](status.md)。本轮默认执行范围是 G0 和 G1；G2–G4 是后续研究计划，不因出现在本文件中而自动启动。当前没有连接用户服务器，没有提交 GPU 作业。

### 12.1 研究假设与可证伪条件

H1：在主动信息获取有价值的前缀上，特权信息与参考动作构成的监督，可能在按正确后验聚合后诱导较差的可实施选择。
H2：明确信息可实施性约束能修复其中一部分，且该作用不能完全由参考动作覆盖不足解释。
H3：若简单约束之后仍存在有意义的效果、计算成本或迁移缺口，进一步研究信息受限过程标签/PRM 才有必要。

主机制比较固定同一个只读合法历史的 continuation policy π，gold 为精确 Q^π。小型问题里 π 可以是预先固定的 Bayes controller。若另外使用冻结 LLM 作为 continuation，其 Q^π_LLM 要另列，不把后续执行能力差异归因于信息语义。Q* 作为能力更强的参照也要单列。

### 12.2 样本设计与数据冻结

第一轮 360 个独立根场景，三个真正不同的机制各 120：
1. 静态诊断：二元故障、检测成本与可靠度、维修或暂停。
2. 动态证据：二元安全/权限状态、旧观察时间戳、查询与执行之间的状态变化。
3. 执行进度不确定：四个有限进度状态、超时后的状态核验、重试和后续补偿；状态转移需要真正不同于静态诊断。

每种机制按“信息有明确价值 / 信息无益 / 价值边界”各 40 个，分层依据 CPU gold，不依据任何教师输出。明确价值与无益分别要求最优检查与最优非检查动作的价值差 >=0.5 与 <=-0.5（沿用 +10/-30 奖励量纲）；边界绝对差 <0.5。不同任务奖励跨度另在 manifest 冻结。

每种机制 24 dev、96 audit，总计 72 dev + 288 锁定 audit；每个价值层分别抽 8 dev、32 audit。固定 split seed=17。每根场景只取一个 prefix；同一 case 的隐状态、候选动作与改写不增加独立样本数。所有隐藏世界按真实 posterior 加权，二元/四元状态完整枚举；动态转移噪声按冻结的转移模型积分或枚举。禁止默认每个世界各 1/2。不得把 dev 已覆盖的三个机制称为机制 OOD。

每个前缀固定三个第一步候选动作；后续动作由 π 执行。候选顺序按 case 固定随机排列，同 case 的所有教师条件使用相同顺序。manifest 记录数据/代码 hash、模型 commit、候选集合、后验、continuation 定义、分割、奖励跨度、主要指标和 gate。新数据只允许 data/audit/cases.jsonl 与 data/audit/manifest.json；公开 history 与 private context 必须独立字段，H 条件和学生输入禁止包含 private 字段。

### 12.3 教师条件与调用预算

| 条件 | 输入与作用 |
|---|---|
| H | 合法历史、公共规则、候选动作；每个 case 调用一次 |
| P | H + 截止当前时刻的真实隐藏状态，不默认提供未来随机结果 |
| PR | P + 目标参考动作流程所使用的 reference actions |
| PR-IC | PR + 明确要求后续决策只使用学生实际能获得的信息 |
| PR-full / PR-full-IC | 预先选定的 72 个 audit case（每机制24）上的参考完备性对照 |

PR 的输出语义必须忠实于被复现流程。原流程如果产生参考动作集合并按匹配给 +1/0/-1，就沿用这一规则；另外询问 Bayes 回报或完整排序属于“适配实验”，不能称原方法复现。采用本地 4B 教师或改动提示时称 VPR-style，不能直接宣称复现原论文模型。

full 只在同一参考生成规则与信息假设下补全等价、同样合格的参考动作；不得把信息受限 oracle 的正确动作直接补进去。无法隔离这两种干预时，报告混淆而非因果归因。

教师输出只含短 JSON 动作/评分，不要求长推理文本，max_new_tokens=128。仅 dev 允许最多两版提示（初版+一次修订）；audit 开始后不可改提示、主要指标、分层或阈值。

按前两机制 2 个隐藏世界、第三机制 4 个隐藏世界计算：
- H/P/PR/PR-IC 首轮共 360 + 3 × (120×2 + 120×2 + 120×4) = 3,240 次教师调用。
- 72-case 参考完备性对照另加 2 × (24×2 + 24×2 + 24×4) = 384 次。
- 第二版 dev 最多额外 72 + 3×(24×2 + 24×2 + 24×4) = 648 次。
- 教师调用硬计数上限 4,272，最多 546,816 个输出 token；禁止为无效结果无限重试。

普通 history-only MC 必须 rollout 同一个 π；已有解析 π 时这条对照直接用 CPU。冻结 LLM continuation 作为额外实践基线，最多 72 个预定 prefix × 3 候选 × 4 rollout × 3 后续动作 = 2,592 次生成，每次最多 64 新 token（165,888 输出 token）。4 rollout 仅是成本受控粗筛，不能充当 gold。若状态数、候选数或回合数超过上述定义，必须先重新计算预算，不能悄悄扩大。

### 12.4 主要指标与 gate

令教师给候选动作的分数为 s_c(h,a;z)，先得到 S_c(h,a)=Σ_z b(z|h)s_c(h,a;z)。使用偏好时，按实际拟用训练目标聚合偏好概率，不随意平均每个世界的 argmax。最高分并列时均匀选取；无效输出按预注册的全候选平分处理，同时单独报告无效率及 valid-only 敏感性结果。

主指标：
r_i = [max_a Q^π(h_i,a) - E_{a~π_teacher} Q^π(h_i,a)] / S_i。
S_i 是该任务预先规定的完整收益跨度，不能看过教师结果再选。唯一主对比 Δ=mean(r_PR-r_PR-IC)。按机制分层、以根 case 为单位做 2,000 次 paired bootstrap（seed=23），报告 95% 区间。独立 n=288；参考完备性子集 n=72，只作机制筛查。

同时报告：必要核验误排率、无益核验成本、任务收益与实际伤害、格式有效率、暂停比例、完整调用及 GPU 时间。dev 格式有效率要求至少98%；若 audit 组间格式差超过2个百分点，必须报告该混淆，不能只归因于信息约束。

在 audit 前冻结的工程 gate 示例：
- PR 在“信息有明确价值”层中，至少15%的 case 有 r_i>=0.05，且问题出现在至少两个机制。
- Δ>=0.01 且其配对95%区间支持正向改善。
- 完备性对照不能把全部问题解释成参考动作覆盖不足。
- 进入新方法阶段前明确简单修复后尚存的效果、成本或真实环境迁移缺口。

15%、0.05、0.01 是项目管理阈值，不是统计学定理。置信区间宽时结论为证据不足；不得看 audit 后改阈值或补有利样本。结论允许为“存在值得继续的缺口”“简单修复已足够”或“现象/机制证据不足”。后两种结果不自动扩大训练。

### 12.5 分阶段运行表

| 阶段 | 运行内容 | 规模与结束条件 | checkpoint |
|---|---|---|---|
| G0 | 环境/权限/磁盘核查、CPU gold、BF16 smoke、一次独立训练 profile | 20 steps、256 train、64 validation；profile只看工程指标 | 0 |
| G1-dev | dev 提示和吞吐检查 | 72 case，最多两版提示；冻结后才进入 audit | 0 |
| G1-audit | H/P/PR/PR-IC 完整审计 | 288锁定case，按后验聚合 | 0 |
| G1-controls | 72-case参考完备性、同π MC、bootstrap分析 | 达到调用或资源上限就结束并报告覆盖率 | 0 |
| G2（后续） | 监督影响的偏好学习 | PR、PR-IC、history-MC、拟研究方法四种标签；相同前缀与候选对；seeds 42/43/44，最多12个正式训练run | 每run最多1个最终adapter |
| G3（后续） | 学习型PRM与同优化器在线比较 | 与终局奖励保持相同任务/伤害目标；先1,000–2,000短episode；重新核算预算 | 单独登记 |
| G4（后续） | 独立工具环境与第二模型 | ToolSandbox本地子集或明确标注的AgentDojo扩展；整轨迹评测 | 单独登记 |

G2 的初始建议：4B BF16 LoRA r16，learning_rate=1e-5、beta=0.1、microbatch=1、grad_accum=8、1 epoch，输出动作<=128 tokens。优先固定参数比较标签，不做大网格搜索。训练数据须与 audit 根任务、规则组合隔离；当前1,830条toy训练pair只能用于工程验证，不能自动充当完整机制实验数据。格式、吞吐与loss正常不能替代研究gate。完整PRM结论还需要 G3/G4。

### 12.6 文件和计算预算

| 项目 | 初始 G0/G1 上限 |
|---|---:|
| 同时运行 | 2个单GPU任务；每张已分配GPU最多1个 |
| 真实队列中等待 | 最多4个本项目任务 |
| 总GPU预算 | 36 GPU-hours，含失败/重试；这是封顶，不是耗时预测 |
| 单次profile | 最多1 GPU-hour，20 optimizer steps |
| 单个教师/MC作业 | 默认最长4小时，提交前按profile估算 |
| 同逻辑run尝试 | a01 + 最多一次有明确修复的a02 |
| 新实验产物总量 | G0/G1 ≤2 GiB；后续G2连同历史产物≤10 GiB |
| 单run非模型文件 | ≤128 MiB；合并日志≤10 MiB |
| 新合成数据 | ≤128 MiB |
| 共享模型缓存 | 最多1个指定底模revision；新增缓存预算24 GiB |
| G0/G1模型保存 | 0 |
| 后续正式adapter | 每run一个，两个文件合计≤512 MiB；矩阵最多12个 |

GPU预算应计入已消耗时间、当前运行可能消耗的剩余时间以及已提交作业预留时间，不能把36小时同时分配给许多独立job。调用上限跨全部attempt累计，包含24-case试跑、无效输出和失败前已执行的调用；同提示版本下的试跑结果在72dev内复用。2GiB统计本轮runs、logs、data/audit、失败attempt和临时写入空间，并为运行及排队作业预留输出额度；单run128MiB包括外部日志，共享模型缓存单独计账。若集群提供 quota，记录实际硬限额；仅目录检查与应用层策略不能称操作系统级硬磁盘限额。预算达到上限即停止新增任务，保留已有证据，不自动删除用户文件腾空间。

固定产物：
- 训练 runs/<ATTEMPT_ID>/run.json 和 metrics.jsonl。
- 推理/审计 runs/<ATTEMPT_ID>/run.json 和 predictions.jsonl；沿用现有 evaluate_actions.py 时 summary.json 是唯一汇总文件，不另写同内容 run.json。
- stdout/stderr 合并到 logs/<ATTEMPT_ID>.log，放在训练 output-dir 外。
- 只有后续显式 --save-final-adapter 的正式训练，允许 adapter/adapter_model.safetensors 与 adapter/adapter_config.json。
- 禁止 checkpoint-*、pytorch_model*.bin、整基座副本、optimizer/scheduler/RNG/trainer_state、tokenizer副本、每步报告、每样本文件、重复csv/json镜像、W&B/TensorBoard自动目录和按时间戳不断产生的脚本副本。
- 只允许新增 audit_cases.py 与 audit_teacher.py 两个源文件；复用现有模块，提示模板放入同一审计脚本。数据与输出目录按上述白名单创建，不预创建空run目录。
- 共享缓存只准备一次；其余任务 local-files-only。不能重设 HOME 或 CODEX_HOME。启用 PYTHONDONTWRITEBYTECODE=1，避免运行时生成新 __pycache__。

目前仅adapter保存上限和输出文件行为在训练代码中做了防护；总GPU时间、调度、日志/目录总量由执行agent和服务器作业限制共同维护，尚没有独立的强制配额系统。不要声称这些限制已在用户服务器部署。

### 12.7 status.md 的维护协议

status.md 只有一个协调者写；任务与子agent只写自己的允许产物、向协调者报告。采用同目录临时文件 .status.md.tmp 写完后原子替换；不生成 status_v2.md、status_backup_*。不存在并发writer时不引入额外锁服务或常驻daemon。

固定四块：已完成实验、服务器排队/运行作业、待执行计划、预算/阻塞与决策。每条已提交记录至少包含 RUN_ID/ATTEMPT_ID、依赖、状态、原始调度状态、Job ID或主机+PID+启动时间、申请GPU数、提交/开始/结束时间、最后核验时间、进度、预算、证据路径及下一步。

PLANNED=尚未提交；READY=条件已核查但未提交；SUBMITTING=提交意图已记账/结果待确认；QUEUED=真实Job ID且当前查询为PENDING；RUNNING=有实际运行证据；VERIFYING=作业结束但产物尚未核验；SUCCEEDED=退出成功且约定产物/步数验证通过；FAILED/CANCELED/BLOCKED按证据填写。保留 scheduler_state_raw，不把 CONFIGURING/COMPLETING 等都当训练或失败。

提交前登记唯一attempt、完整命令、版本、预算和输出路径，再提交。Slurm必须把唯一project_tag/ATTEMPT_ID写入job-name或comment并在status预登记；使用 sbatch --parsable 并记录 jobid 或 jobid;cluster；收到ID后再查询 squeue，结束后用 sacct核对退出码与结果。提交断线时先查已有任务，无法确认则保持SUBMITTING/待核实，禁止盲目重交。无调度器时仅待启动清单不能称QUEUED；进程身份需主机、PID和启动时间，防止PID复用。

在启动/提交前、获得回执后、开始运行后、失败/完成后、恢复连接时和结束当前会话前更新。长作业在会话活跃期间每5–10分钟核验一次；断线后保留最后成功核验时间，另记查询失败。无人维护时文件不会自动刷新，不声称实时监控。默认不创建后台守护进程、定时任务或无限轮询。

当前没有完整续训checkpoint，resume_supported=false。最终adapter用于推理/初始化，不代表可精确续训。Slurm试验使用--no-requeue避免普通抢占后自动从头重跑；仍须核对集群强制重排设置。训练失败必须保留a01记录，修复后用a02新目录从头运行；不覆盖、不伪造resume。审计推理按固定调用键核对已有持久记录，a02只补未完成调用，结果引用a01而不复制；未知已执行调用保守计入总预算。非预期Slurm重启不提供额外尝试/预算，应先核对。当前无模型服务器访问，因此 status 中GPU任务为计划/阻塞，服务器队列写未查询，而非“0个排队”。

### 12.8 直接运行命令与阶段边界

下面仅演示普通两卡服务器、GPU索引已核实属于本项目的情况；在Slurm作业内保留调度器提供的CUDA_VISIBLE_DEVICES，不重新指定物理卡。模型commit应先解析并记录为实际值，不用占位符直接提交。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 python train_dpo.py \
  --train-file data/information/train.jsonl \
  --eval-file data/information/validation.jsonl \
  --output-dir runs/G0-PROFILE-a01 \
  --max-steps 20 --max-train-samples 256 --max-eval-samples 64 \
  --seed 42 --local-files-only
```

该profile不保存模型。实际提交时还应带 --revision 的固定commit、整个进程的walltime，并将合并日志写到已登记的 logs/G0-PROFILE-a01.log。max_steps 不限制下载、预处理或reference预计算时间。审计脚本尚待实现；不要把文档中的audit命令臆造为已存在的CLI。

第一轮交付是填好的 status.md、固定数据/提示版本、配对审计指标及“继续/简单修复足够/证据不足”的判断。不要为得到正面结果自动扩展到正式DPO、PRM或在线RL。


