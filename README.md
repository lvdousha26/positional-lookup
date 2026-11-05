# The Answer Stops at Eight

[![Lean proofs](https://github.com/lvdousha26/positional-lookup/actions/workflows/lean.yml/badge.svg)](https://github.com/lvdousha26/positional-lookup/actions/workflows/lean.yml)

**论文页面**：https://lvdousha26.github.io/positional-lookup/
**PDF**：[paper/main.pdf](paper/main.pdf) ·
**讲解视频**：[video/lookup.mp4](video/lookup.mp4) ·
**Lean 形式化**：[lean/](lean/)

一次端到端的自动化科研流程演示：文献 → 假设 → 实验 → 分析 → 论文 → 自我评审。

## 主张

**"显式位置编码导致长度泛化失败"这个说法，在 copy 类任务上有一半是终止决策造成的假象。**

训练长度 8、测试长度 10 时，带显式位置编码的模型并不是算不出来，而是**在第 9 个位置以
100% 概率输出 EOS**，把答案截断在 8 个 token。序列级 exact-match 因此报 0.00，而它的
token 准确率是 0.80（10 个里对 8 个），屏蔽 EOS 强制长度后恢复到 0.91–1.00。

更关键的是：**去掉 EOS 监督后排序反转**。同样无 EOS 训练，RoPE 在 L=16（两倍训练长度）
仍然 1.00，而 NoPE 掉到 0.43。

这张表是核心证据（copy，固定总长 41 的协议，2 种子均值）：

| 编码 | Acc | Tok | 强制长度 | 停止位置 |
|---|---|---|---|---|
| NoPE | **0.78** | **0.98** | 0.99 | 9.9 |
| ALiBi | 0.92 | 0.99 | **1.00** | 9.9 |
| learned | 0.00 | 0.81 | 0.83 | **8.0** |
| sin | 0.00 | 0.80 | 0.93 | **8.0** |
| q/k-abs | 0.00 | 0.80 | 0.91 | **8.0** |
| RoPE | 0.00 | 0.80 | 1.00 | **8.0** |

判据不是"绝对 vs 相对"，也不是"注入点"——而是**模型能否给答案结尾一个地址**。

## 证据链

1. **位置不是变量**。固定总 token 数 41、尾部填充，使 41 个位置在**每个**训练样本里都出现，
   测试时没有任何位置是新的。崩溃照旧（`run_padded.py`）。
2. **失败是"停"，不是"算"**。逐输出位置准确率：前 8 位 1.00，第 9 位起恰好 0.00；
   同位置 EOS 概率 1.00。屏蔽 EOS 后 token 准确率 0.80 → 0.91/0.93/1.00（`analyze_padded.py`）。
3. **不是注入点**。把同一正弦码线性投影到 q/k 而不是残差流（`qk_abs`），行为完全一致；
   RoPE 是相对码，同样截断——所以"相对性"也不足以解释。
4. **边界跟着训练长度走**。`L_tr=12` 时前缀变成 12（L=14/16/18 都是）；换到
   `L_tr∈{6,8,10}` 混合训练，sin 在 L=12 仍是 0.00，而 NoPE 是 0.99（`probe_rows.py`,
   `probe_multilen.py`）。
5. **去掉边界，差距消失且排序反转**。无 EOS 训练：RoPE 全长度 1.00，NoPE 掉到 0.43
   （`probe_noeos.py`）。

## 被证伪的假设（保留记录）

这个项目最初的问题是"PE 的效应是否依赖任务属性"，提出的条件变量是
**length-coupling**（位置映射是否写成 L 的函数），预测耦合任务上显式 PE 占优。

**预测不成立**：50 次训练（5 任务 × 5 编码 × 2 种子，全部先过 solved gate）显示 NoPE
在能外推的任务上全部最优或并列，包括两个为击败它而构造的耦合任务；交互效应极显著
（F(16,25)=32.3, partial η²=0.95）但**不由 coupling 组织**。

三个已经被否掉的中间解释，也记录在案：

- ~~位置超出训练范围~~ → 固定总长协议下所有位置都见过，崩溃不变
- ~~断崖跟随 L_tr~~ → 该探针在单一训练长度下无判别力（"位置超范围"与"长度超范围"同时发生）
- ~~相对编码更可外推~~ → RoPE 是相对码，同样在 L_tr 处截断

## 目录

```
code/
  arlib.py            任务、6 种 PE（含 qk_abs 对照）、tiny transformer
  padded.py           固定总长 41 的协议（判别实验核心）
  run_padded.py       注入点实验网格（可断点续跑）
  analyze_padded.py   逐位置准确率、EOS 概率、强制长度、停止位置
  probe_rows.py       L_tr=12 时边界是否移动
  probe_multilen.py   L_tr∈{6,8,10} 混合训练
  probe_noeos.py      去掉 EOS 监督（因果检验）
  plot_figs.py        全部论文图（不依赖 torch，可单独跑）
  run_grid.py         标准协议 5×5 网格
paper/
  main.tex            11 页 PDF
  sections/           各章节
  refs.bib            22 条引用，逐条核实
  CITATION_AUDIT.md   引用审计记录
  review.md           自我评审
results/              JSON + 汇总表
figures/              4 张论文图（pdf/png）
```

## 复现

```bash
pip install -r requirements.txt

# 1) 注入点实验 + 逐位置分析（约 40 分钟，单进程）
python code/run_padded.py --tasks copy --pes nope,alibi,learned,sin,qk_abs,rope --seeds 0,1
python code/analyze_padded.py

# 2) 边界跟随训练长度（各约 1 分钟/run）
python code/probe_rows.py --task copy --pe sin --L_train 12
python code/probe_multilen.py --pes sin,nope --seeds 0,1

# 3) 因果检验：无 EOS 监督
python code/probe_noeos.py --pes nope,sin,rope --seeds 0,1

# 4) 出图与编译
python code/plot_figs.py && cd paper && latexmk -pdf main.tex
```

## 硬件与环境

RTX 4060 Laptop (8GB) + torch 2.7.1+cu118。

两个必须记住的坑：

- **驱动 531.61 只支持到 CUDA 12.1**。cu126 版 torch 能识别设备（`cuda.is_available()`
  为 True）但第一个 kernel 就报 `cudaErrorDevicesUnavailable`；必须用 cu118 构建。
- **这台机器不能并行跑多个训练进程**。3 个进程把 8GB 显存和 Windows 页面文件一起耗尽，
  触发 `cudaErrorLaunchTimeout`（驱动 TDR 重置）和 `WinError 1455`；串行反而更快。

## 诚实的评估

见 [paper/review.md](paper/review.md)。一句话：机制清楚、可复现、结论方向明确，但**只有
copy 一个任务、2 个种子、3.2M 参数**。no-EOS 的排序反转是全文最强的一击，也是最需要
在更多任务上重复的一个数。
