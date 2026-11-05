# 工作流

本项目的流程已固化为可复用 skill（用户级，跨项目可用）：

```
~/.workbuddy/skills/ai-research-pipeline/
├── SKILL.md                          十个阶段 + 硬性规则 + 交付清单
├── references/citation-audit.md      三层引用审计（重点）
├── references/experiment-discipline.md  solved gate / 并发陷阱 / TF32
├── references/figure-standards.md    teaser / 配色 / TikZ / matplotlib 坑
└── scripts/check_citations.py        机械一致性检查（可自动跑）
```

触发方式：说"跑一遍 autoresearch"、"做一个能投 A 会的论文"、或"检查这份稿子有没有幻觉引用"。

## 十阶段

| # | 阶段 | 本次对应产物 |
|---|---|---|
| 0 | 环境探针 | 发现驱动 531.61 只能用 cu118 |
| 1 | 文献与缺口 | Kazemnejad 2023 的聚合排名 → 条件变量缺口 |
| 2 | 可证伪假设 | P1 离散度、P2 排序反转 |
| 3 | 先设计控制 | solved gate（L=8 ≥ 0.95） |
| 4 | pilot 先行 | 800 步确认有信号 |
| 5 | 全量、可续跑 | 50 个 JSON |
| 6 | 交互检验 | ANOVA F(16,25)=32.3 |
| 7 | 图表 | teaser + TikZ 概念图 + 4 张数据图 |
| 8 | 写作 | Adelson 四步法重写 intro + abstract；13 页，8 条编号公式 |
| 9 | 敌意自评审 | 3/10 reject，建议转 workshop |
| 10 | 引用审计 | 21 条逐条核实，揪出 4 类问题 |

## 本次的可复现配置

```
任务       copy, prefix_xor, sort, swap_halves, reverse
位置编码   NoPE, learned, sin, RoPE, ALiBi
种子       0, 1
模型       4 层, d=256, 8 头, ~3.2M 参数, pre-LN
训练       2000 步, batch 128, AdamW lr 5e-4, wd 0.01, cosine + 100 warmup
长度       训练 8；测试 8/10/12/16/20
探针       训练 12，验证断崖是否跟随 L_tr
评估       每长度 512 条，固定 eval 种子
```

## 引用审计怎么执行

```bash
# 1. 机械检查（假 key、重复标题、缺图、缺 venue）
python ~/.workbuddy/skills/ai-research-pipeline/scripts/check_citations.py \
        --paper paper/main.tex

# 2. 逐条对原始来源核实 —— 见 paper/CITATION_AUDIT.md
# 3. 声称核实：对每条引用问"删掉它，这句话还成立吗"
# 4. 数字核实：所有计数从 results/ 重算，不手写
```

三层里，第 1 层脚本能覆盖，第 2 层最容易被忽略也最致命：
这次 21 条引用全部真实存在，但正文有一处"使用同样的五种位置编码"是假的
（实际替换掉了 T5 relative bias）。引用列表干净，句子是错的。

第 4 层同样容易漏：贡献列表里 `5×5×3` 在种子数改成 2 之后一直没同步。
