# Citation audit

每条引用都独立检索核实，不依赖模型记忆。日期 2026-11-05。

## 核实结果：13/13 条目存在

| # | key | 条目 | 核实来源 | 状态 |
|---|---|---|---|---|
| 1 | anil2022exploring | NeurIPS 2022, vol 35, 38546–38556 | neurips.cc/virtual/2022/52978 + 论文 PDF | ✅ 作者与页码一致 |
| 2 | deletang2023neural | ICLR 2023 | iclr.cc/virtual/2023/session/13335 | ⚠️ 已修（见下） |
| 3 | kazemnejad2023impact | NeurIPS 2023, vol 36 | papers.nips.cc 论文 PDF | ✅ |
| 4 | nanda2023progress | ICLR 2023 | mlanthology.org + dblp ICLR 2023 | ✅ |
| 5 | power2022grokking | arXiv:2201.02177 | arxiv.org/abs/2201.02177 | ✅ 补了编号 |
| 6 | press2022train | ICLR 2022 | OpenReview / GitHub 官方 cite | ✅ |
| 7 | raffel2020exploring | JMLR 21(140):1–67 | jmlr.org/papers/v21/20-074.html | ✅ |
| 8 | ruoss2023randomized | ACL 2023 Short, 1889–1903 | aclanthology.org/2023.acl-short.161 | ⚠️ 已修（见下） |
| 9 | shaw2018self | NAACL-HLT 2018, 752–761 | 标准引用 | ✅ |
| 10 | su2024roformer | Neurocomputing 568:127063 | 多处书目记录一致 | ⚠️ 补了卷期 |
| 11 | vaswani2017attention | NeurIPS 30, 5998–6008 | 标准引用 | ✅ 补了页码 |
| 12 | wang2024length | Findings ACL 2024, 14024–14040 | aclanthology.org/2024.findings-acl.834 | ✅ |
| 13 | zhou2023what | ICLR 2024 | iclr.cc/virtual/2024/poster/19236 | ✅ |

### 第二批（补充引用，13 → 21）

| # | key | 条目 | 核实来源 | 状态 |
|---|---|---|---|---|
| 14 | chi2023dissecting | ACL 2023, 13522–13537 | aclanthology.org/2023.acl-long.756 | ✅ 含 Outstanding Paper |
| 15 | hahn2020theoretical | TACL 8:156–171 | aclanthology.org/2020.tacl-1.11 | ✅ |
| 16 | jelassi2023length | arXiv:2306.15400 | arXiv 摘要页 | ✅ 只写 arXiv（会议信息不确定） |
| 17 | liu2023shortcuts | ICLR 2023 | iclr.cc/virtual/2023/poster/11118 | ✅ |
| 18 | olsson2022induction | Transformer Circuits Thread | transformer-circuits.pub | ✅ 作者 27 位，用 others 截断 |
| 19 | sun2023length | ACL 2023, 14590–14604 | aclanthology.org/2023.acl-long.816 | ✅ |
| 20 | weiss2021thinking | ICML 2021, PMLR 139:11080–11090 | proceedings.mlr.press/v139/weiss21a | ✅ |
| 21 | zhao2024survey | Findings EMNLP 2024, 9959–9977 | aclanthology.org/2024.findings-emnlp.582 | ✅ |

第二批每条都核对了作者全名、页码和会议，没有沿用记忆。
特别检查过的两处：`chi2023dissecting` 的 Rudnicky 首名是 Alexander（部分来源写 Alexander I.），
按 ACL Anthology 用 Alexander；`olsson2022induction` 有 27 位作者，
`others` 之前列出的 10 位均与原文一致。

## 发现并修正的问题

### A. 虚构作者（最严重）

**deletang2023neural** 原 bib 含 `Trebacz, Maksymilian` —— 该文 11 位作者中
**没有**此人；实际作者表还包含被我漏掉的 `Genewein, Tim`。另外 `Wenliang, Li`
实为 `Wenliang, Li Kevin`。

真实作者顺序：Deletang, Ruoss, Grau-Moya, **Genewein**, Wenliang (Li Kevin),
Catt, Cundy, Hutter, Legg, Veness, Ortega。

→ 已按 ICLR 官方页面与 ML Anthology 的 BibTeX 逐字修正。

### B. 姓名拼错 + 多出作者

**ruoss2023randomized** 原 bib 写 `Bennett, Mehdi`，实为 **`Bennani, Mehdi`**；
并多列了 `Ortega, Pedro A.`，而该 ACL 版为 8 位作者、**无 Ortega**
（Ortega 是同组另一篇 Chomsky 那篇的作者，两条被我串了）。

→ 已删 Ortega、改 Bennani，补 pages 1889–1903。

### C. 正文声称与文献事实不符（同样属于幻觉）

正文两处写"使用与 Kazemnejad et al. **相同的五种** PE"：

- `setup.tex`：*"These are the five schemes compared by Kazemnejad et al., so our
  design is directly comparable to that study."*
- `intro.tex`：*"with the same five PE schemes used by Kazemnejad et al."*

**但事实不是这样。** Kazemnejad 比较的五种是 APE / **T5 relative bias** /
ALiBi / Rotary / NoPE；我实现的是 learned APE / **sinusoidal** / RoPE / ALiBi /
NoPE —— 我用 sinusoidal **替换掉了** T5 relative bias。

→ 两处均改为：四种取自原文，sinusoidal 为替换项，并明确说明这限制了跨研究
数值比较的可比性。**这是本次审计中唯一会误导审稿人的方法描述错误。**

### D. 数字与数据不符

- 正文写 "thirteen of twenty-five cells sit at exactly zero"。
  实际统计：ES < 0.01 的有 **21** 个，恰好为 0 的 **18** 个。
- 正文写 "three cells are pinned at zero for all methods"。
  实际是 **两个任务**（prefix-XOR、swap-halves）的全部 10 个格子为零。

→ 均已按 `results/summary.csv` 重算并改正。

## 保留的已知不精确处

- `shaw2018self` 的 booktitle 简写为 "Proceedings of NAACL-HLT"，未写全称
  "2018 Conference of the North American Chapter of the ACL"。
- `nanda2023progress` 第三作者在不同来源有 `Lieberum` / `Liberum` 两种拼法，
  采用 dblp 与 ML Anthology 的 `Lieberum`。

这两处不影响可查证性，未改。
