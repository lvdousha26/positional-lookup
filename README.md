# Explicit Positional Encodings Turn Length Generalisation into Lookup

[![Lean proofs](https://github.com/lvdousha26/positional-lookup/actions/workflows/lean.yml/badge.svg)](https://github.com/lvdousha26/positional-lookup/actions/workflows/lean.yml)

**论文页面**：https://lvdousha26.github.io/positional-lookup/
**PDF**：[paper/main.pdf](paper/main.pdf) ·
**讲解视频**：[video/lookup.mp4](video/lookup.mp4) ·
**Lean 形式化**：[lean/](lean/)

一次端到端的自动化科研流程演示：文献 → 假设 → 实验 → 分析 → 论文 → 自我评审。

**最终主张**：显式位置编码导致长度泛化失败，不是因为位置超出训练范围，而是因为
模型学到的是**以长度为索引的查表**。判别实验：把总 token 数固定为 41、所有位置都
在训练中出现，只变有效长度——learned/sin 仍从 1.00 掉到 0.00，NoPE 保持 0.79、
ALiBi 0.87。（最初的 length-coupling 假设被证伪，论文如实写了。）

流程已固化为可复用 skill（含强制引用审计），见 [WORKFLOW.md](WORKFLOW.md)；
skill 本体在 `~/.workbuddy/skills/ai-research-pipeline/`。

## 研究问题

位置编码（PE）是否真的如 Kazemnejad et al. (NeurIPS 2023) 的聚合排名所示，
"无位置编码（NoPE）普遍最优"？还是这个效应依赖于任务属性？

我提出的条件变量是 **length-coupling**：任务的位置映射是否写成了序列长度 L 的函数。

| 任务 | 位置规则 ρ(i,L) | 类别 |
|---|---|---|
| copy | i | 解耦 |
| prefix-XOR | 1..i | 解耦 |
| sort | 无（由值决定） | 无位置 |
| swap-halves | (i+L/2) mod L | **耦合** |
| reverse | L+1-i | **耦合** |

预测：在耦合任务上，显式 PE 应占优（因为规则需要 L，而解析位置码是天然的基底）。

## 结论

**预测不成立。** 5 任务 × 5 PE × 2 种子 = 50 次训练，全部先达到 L=8 准确率 ≥0.998
才计入外推统计：

- NoPE 在所有能外推的任务上都最优或并列最优，**包括两个为击败它而构造的耦合任务**
- learned / sin / RoPE 从 L=8 的 1.00 直接跌到 L=10 的 0.00（断崖，而非渐变）
- **任务 × PE 交互极显著**（F(16,25)=32.3, p<10⁻³, partial η²=0.95）——
  异质性确实存在，但**不由 length-coupling 组织**
- 探针实验：`L_train=12` 时断崖移到 L=14（而非固定在 L=10），说明模型学的是
  "长度索引的查表"，不是可外推的规则
- 已排除两个替代解释：拟合不足（有 solved gate）、结构标记位置漂移（锚定实验仍崩）

## 目录

```
code/
  arlib.py        任务生成、5 种 PE、tiny transformer（单文件自包含）
  pilot.py        单次训练 + CLI
  run_grid.py     网格调度（可断点续跑）
  analysis.py     汇总、ANOVA、出图
  check_tasks.py  序列对齐自检（防静默 bug）
  bench.py        吞吐基准
paper/
  main.tex        12 页 PDF（8 条编号公式、3 张图）
  sections/       各章节 + Figure 1 概念图（TikZ）
  refs.bib        21 条引用，逐条对原始来源核实过
  CITATION_AUDIT.md  引用审计记录（含 4 类已修问题）
  review.md       自我评审（3/10，reject，建议转 workshop）
results/          50 个 run 的 JSON + runs.csv + summary.csv + analysis.txt
figures/          3 张论文图（pdf/png）
```

## 图表与素材来源

| 资源 | 来源 | 许可 | 用途 |
|---|---|---|---|
| SciencePlots `science.mplstyle` | github.com/garrettj403/SciencePlots | MIT | 论文级 matplotlib 样式（刻度、线宽、配色） |
| TikZ / pgf | CTAN（TeXLive 2024 自带） | LPPL | Figure 1 概念图，纯矢量、与正文同字体 |
| matplotlib | — | PSF | 其余数据图 |

采用了但**没有**用的资源，以及原因：

- **ml-visuals**（dair-ai，16k star）：100+ 张 ML 图块，但分发形式是 Google Slides / PPTX，
  进不了 LaTeX 流水线；且本文需要的是"任务映射示意"而不是网络结构图，套模板反而更慢。
- **PlotNeuralNet**：CNN/AlexNet 那类三维层叠结构图，与本文内容无关。
- **tablericons / lucide-icons**：TeXLive 2024 未收录这两个宏包；`fontawesome5` 可用，
  但概念图里用箭头和方框已经够了，加图标反而干扰。

一个技术细节：SciencePlots 默认 `text.usetex: True`，本机缺 ghostscript 会让 PDF 输出失败，
因此关闭 usetex、改用 STIX 字体。STIX 与正文的 Times（mathptmx）视觉一致，混排不会露馅。

四张图的三个设计决定（都是看了第一版之后才改的）：

1. 热图换成**白底蓝色渐变**。原来用 viridis，21/25 个格子接近零，整张图糊成一坨黑紫色，
   读者第一眼看到的是"什么都没有"，而不是"只有四格有值"。
2. learned / sin / RoPE **合并成一条曲线**。它们数值上完全一致（训练长度处 1.00，之后全 0），
   画三条重叠线会暗示这是三个独立测量，实际是同一个事实。
3. 断崖探针单独成图。它是全文唯一能区分"位置范围耗尽"与"训练长度耗尽"的证据，
   塞进六宫格会被淹没。

三张图的分工：Fig 1 判别实验（位置全部见过时各编码的表现）、Fig 2 长度曲线六宫格、
Fig 3 任务分类概念图（TikZ）。按"极简"原则删掉了热图和离散度图——它们服务的是
已被降级的 coupling 论证，留着是包装。

## 复现

```bash
cd code
python check_tasks.py                                   # 先验证对齐
python run_grid.py --workers 1 --steps 2000 --seeds 0,1 # 约 65 分钟
python analysis.py                                      # 出表出图
cd ../paper && latexmk -pdf main.tex
```

## 硬件与环境

RTX 4060 Laptop (8GB) + torch 2.7.1+cu118。
**注意**：驱动 531.61 只支持到 CUDA 12.1，cu126 版 torch 能识别设备但 kernel
直接报 `cudaErrorDevicesUnavailable`；必须用 cu118 构建。多进程跑会互相争抢导致
单 run 慢 6 倍、总吞吐反而低于单进程，因此串行。

## 诚实的评估

这篇**不够 A 会主轨**（自评审 3/10）。原因：核心结论与 Kazenjad 2023 基本一致，
新意在于"证伪自己的假设"，属于方法论贡献而非知识贡献；2 个种子、小模型、合成任务；
缺少机制层面的证据（注意力/探针）。

要够 A 会，缺的是把 ALiBi 那个例外做成主线——即"位置注入位置（表征 vs 注意力偏置）
才是真正的轴"——那是一周量级的实验，不是一次对话能完成的。
