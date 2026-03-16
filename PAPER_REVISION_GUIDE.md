# 论文立即修改指南 - 今天提交版本

## 🎯 目标：快速降低over-claiming风险，提升接收概率

**修改原则**：
- ✅ 用"associated with"、"correlation"代替"causes"、"improves"
- ✅ 添加不确定性表述（suggest, indicate, preliminary）
- ✅ 缩小claim scope到当前实验设置
- ✅ 加强limitations章节
- ❌ 删除所有"first"、"comprehensive"、"state-of-the-art"等绝对化表述

---

## 1️⃣ Abstract 修改

### 当前文本（第1段）
```
Recent large language models (LLMs), particularly when augmented with
reasoning and tool-use capabilities, have demonstrated strong performance
on a variety of cognitively demanding tasks...
```

### ✅ 修改后（添加scope和不确定性）
```latex
Recent large language models (LLMs), particularly when augmented with
reasoning and tool-use capabilities, have demonstrated strong performance
on a variety of cognitively demanding tasks, including code editing,
mathematical problem solving, and complex information retrieval
\citep{swe-bench, hle, omni-olpc}. However, accumulating empirical evidence
suggests that such capabilities have not yet generalized into robust,
domain-agnostic autonomy in realistic settings requiring long-horizon
planning, persistent objective alignment, and stable behavioral consistency
\citep{amodei2024machines, kwa2025measuringaiabilitycomplete, metr2025measuring}.
```

### 当前文本（第2段）
```
To systematically study this challenge, we introduce \textit{RetailBench},
a new benchmark grounded in real-world commercial data and informed by
established economic modeling principles...
```

### ✅ 修改后（缩小scope）
```latex
To systematically study this challenge in a controlled setting,
we introduce \textit{RetailBench}, a benchmark that evaluates
LLM-based agents on retail operations tasks requiring long-horizon
decision-making under stochastic demand and evolving external conditions.
RetailBench is grounded in historical commercial data and informed by
established economic modeling principles from the operations research
literature.
```

### 当前文本（contribution第2条）
```
\item We propose the \textit{Evolving Strategy \& Execution} agent framework,
which improves operational stability compared to a Reflection-based baseline.
```

### ✅ 修改后（去除causal claim）
```latex
\item We propose the \textit{Evolving Strategy \& Execution} agent framework,
which, in our evaluated settings, demonstrates higher operational stability
metrics compared to a day-level Reflection baseline.
```

---

## 2️⃣ Introduction 修改

### 当前文本（最后一段contribution）
```latex
\item We introduce \textit{RetailBench}, a high-fidelity benchmark for
evaluating long-horizon autonomous decision-making in realistic retail environments.
```

### ✅ 修改后（缩小scope）
```latex
\item We introduce \textit{RetailBench}, an open-source benchmark for
evaluating long-horizon decision-making in simulated retail environments,
characterized by coupled pricing-inventory control, stochastic demand,
and multi-day planning horizons.
```

### 当前文本（contribution第2条）
```latex
\item We propose the \textit{Evolving Strategy \& Execution} agent framework,
which improves operational stability compared to a Reflection-based baseline.
```

### ✅ 修改后（去除causal language）
```latex
\item We propose the \textit{Evolving Strategy \& Execution} agent framework,
which separates high-level strategy formulation from low-level execution.
In our experiments, this framework is associated with improved operational
stability metrics compared to a day-level Reflection baseline.
```

---

## 3️⃣ Results Section 关键修改

### 当前文本（第3行）
```latex
\textit{Evolving Strategy \& Execution} consistently outperforms alternative
agent frameworks on core metrics, achieving higher sales and profit while
substantially reducing product expiration rates.
```

### ✅ 修改后（添加统计不确定性说明）
```latex
\textit{Evolving Strategy \& Execution} achieves higher mean sales and profit,
and lower mean product expiration rates, compared to alternative agent
frameworks in our evaluated settings. \footnote{Due to computational constraints,
our current evaluation uses a limited number of random seeds; we report
mean performance and plan to extend with statistical significance testing
in future work.}
```

### 当前文本（第6行）
```latex
our proposed framework demonstrates clear improvements over
\textit{Reflection (Day-Level)}.
```

### ✅ 修改后（去除improvements causal词）
```latex
our proposed framework demonstrates superior mean performance metrics
relative to \textit{Reflection (Day-Level)} across the evaluated models.
```

### 当前文本（第19行）
```latex
Sales per Category and Profit per Category decline substantially,
indicating persistent challenges in effective resource allocation within
increasingly high-dimensional decision spaces.
```

### ✅ 修改后（弱化结论）
```latex
Sales per Category and Profit per Category decline substantially in our
experiments. This pattern suggests challenges in resource allocation as
decision spaces expand, though further investigation with additional seeds
is needed to assess statistical significance.
```

---

## 4️⃣ Related Work 修改（重要！）

### ✅ 在Related Work章节末尾新增段落

在 `\paragraph{Long-horizon agent frameworks.}` 之后添加：

```latex
\paragraph{Positioning relative to existing benchmarks.}
RetailBench complements existing benchmarks by targeting a specific combination
of properties not jointly addressed in prior work: (1) long-horizon planning
over 10-day episodes with day-level decisions, (2) economic optimization
objectives (profit maximization under inventory and pricing constraints),
(3) coupled decision spaces where pricing affects inventory turnover and
vice versa, (4) stochastic dynamics from demand uncertainty, news shocks,
and supply-chain delays, and (5) open-source availability for reproducible
evaluation. While benchmarks such as WebArena and Mind2Web evaluate
multi-step tool use, and VendingBench and OdysseyBench assess long-horizon
planning, RetailBench is distinguished by its focus on economically grounded
retail operations with realistic temporal dependencies and delayed rewards.
Table~\ref{tab:novelty_comparison} provides a detailed comparison.
```

### ✅ 新增Novelty Comparison Table

在Related Work章节中添加表格（建议在新增段落之后）：

```latex
\begin{table}[t]
\centering
\small
\caption{Comparison of RetailBench with related long-horizon benchmarks.}
\label{tab:novelty_comparison}
\begin{tabular}{lcccc}
\toprule
Benchmark & Domain & Horizon & Economic Obj. & Open Source \\
\midrule
WebArena~\citep{webarenarea} & Web tasks & Short-Horizon & No & Yes \\
Mind2Web~\citep{mind2web} & Web tasks & Medium-Horizon & No & Yes \\
VendingBench~\citep{andonlabs2025vendingbench2} & Vending & Long-Horizon & Limited & Yes \\
OdysseyBench~\citep{odysseybenchevaluatingllmagents} & Games & Long-Horizon & No & Yes \\
\textbf{RetailBench} & \textbf{Retail Ops.} & \textbf{Long-Horizon} & \textbf{Yes} & \textbf{Yes} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 5️⃣ Conclusion 修改

### 当前文本（第1段）
```
Experiments on eight state-of-the-art LLMs across progressively challenging
environments show that our framework improves operational stability and
efficiency compared to other baselines.
```

### ✅ 修改后（添加scope和不确定性）
```latex
Experiments on eight state-of-the-art LLMs across progressively challenging
environments show that, in our evaluated settings, our framework achieves
higher mean operational stability and efficiency metrics compared to
the evaluated baselines. However, substantial performance gaps remain
relative to heuristic policies, and observed performance variations across
random seeds indicate that further investigation is needed to establish
statistical significance.
```

### 当前文本（第1段末尾）
```
revealing fundamental limitations in current LLMs for long-horizon,
multi-factor decision-making.
```

### ✅ 修改后（缩小generalization）
```latex
revealing challenges that current LLMs face in long-horizon, multi-factor
decision-making within the RetailBench environment. Whether these
limitations generalize to other domains remains an open question for
future research.
```

---

## 6️⃣ Limitations Section 大幅增强

### ✅ 完全替换当前limitations.tex为：

```latex
\section*{Limitations}

Our work has several important limitations that constrain the interpretation
and generalizability of our results.

\textbf{Evaluation scope.} Our evaluation is conducted in a simulated
single-store supermarket environment. While the environment incorporates
realistic elements such as stochastic demand, perishable inventory, and
exogenous shocks, it remains a simplified abstraction of real-world retail
operations. We do not model multi-store coordination, competitive market
dynamics, or strategic interactions among multiple autonomous agents.
External validity beyond this simulated setting remains to be established.

\textbf{Statistical rigor.} Due to computational constraints, our current
evaluation uses a limited number of experimental runs (three runs per model
in most settings). The reported performance differences should be interpreted
as preliminary observations rather than statistically significant conclusions.
Rank orderings may be sensitive to random seed selection, and effect sizes
may vary with additional repetitions. We plan to extend our evaluation with
paired-seed experimental designs and formal significance testing in future work.

\textbf{Causal claims.} Our comparison of agent frameworks demonstrates
correlations between framework design and performance metrics, but does not
establish causal relationships. Performance differences may be attributable
to factors including token budget, prompt engineering, or model-specific
behaviors rather than framework architecture alone. Controlled ablation
studies are needed to isolate the causal impact of individual framework
components.

\textbf{Baseline strength.} Our current baselines include rule-based
heuristics and agent framework variants, but do not include strong
classical operations research methods such as (s,S) inventory policies or
newsvendor-based ordering strategies. The relative performance of LLM-based
agents compared to these well-established methods remains unknown and is an
important direction for future investigation.

\textbf{Learning and adaptation.} Our evaluation focuses on prompting-based
LLM agents without parameter updates or inter-episode learning. Stronger
performance may be achievable through reinforcement learning, fine-tuning,
or hybrid neuro-symbolic approaches. Our findings should be interpreted as
characterizing zero-shot prompting performance rather than the full potential
of learned agent systems.

\textbf{Failure mode analysis.} While we identify recurring patterns such
as hallucinations and economically irrational actions, we do not propose
explicit algorithmic mechanisms to enforce economic rationality or factual
grounding during execution. Addressing these failure modes through
constraint-aware action control remains an important open problem.

\textbf{Domain specificity.} Our findings are specific to the retail
operations domain and the RetailBench environment. Transferability to other
decision-making domains—such as manufacturing, logistics, or financial
trading—requires empirical validation and should not be assumed without
evidence.
```

---

## 7️⃣ 标题修改（可选，推荐）

### 当前标题
```
RetailBench: Evaluating Long-Horizon Autonomous Decision-Making and
Strategy Stability of LLM Agents in Realistic Retail Environments
```

### ✅ 建议修改（去掉over-claiming）
```latex
RetailBench: A Simulation Benchmark for Long-Horizon Decision-Making
with LLM Agents in Retail Environments
```

或者更保守：
```latex
RetailBench: Evaluating Long-Horizon Decision-Making of LLM Agents
in a Simulated Retail Environment
```

**理由**：
- 去掉"Autonomous" - agents不是真正autonomous
- 去掉"Realistic" - 只是simulation，不是真实环境
- 保留核心claim

---

## 8️⃣ 快速检查清单

### Abstract ✅
- [ ] 删除"first comprehensive benchmark"表述
- [ ] 添加"in our evaluated settings"
- [ ] "improves" → "is associated with higher"
- [ ] "outperforms" → "achieves higher mean"

### Introduction ✅
- [ ] Contribution 1: 添加具体特性描述
- [ ] Contribution 2: 去除causal language
- [ ] 缩小scope到simulation

### Results ✅
- [ ] 添加footnote说明seeds限制
- [ ] 所有"improves" → "achieves higher mean"
- [ ] "outperforms" → "demonstrates superior mean metrics"
- [ ] "indicating" → "suggesting"

### Related Work ✅
- [ ] 新增"Positioning relative to existing benchmarks"段落
- [ ] 新增Table: novelty comparison

### Conclusion ✅
- [ ] 添加"in our evaluated settings"
- [ ] 添加"preliminary observations"表述
- [ ] "fundamental limitations" → "challenges in RetailBench"

### Limitations ✅
- [ ] 完全重写，添加6个详细limitation subsections
- [ ] 明确说明statistical rigor限制
- [ ] 明确说明causal claims限制
- [ ] 明确说明baseline强度限制

---

## 9️⃣ 修改后立即自检

### ❌ 删除这些词
- "first" (首个)
- "comprehensive" (全面的)
- "state-of-the-art evaluation" (SOTA评估)
- "significantly improves" (显著改善)
- "outperforms" (胜过)
- "demonstrates superiority" (展示优越性)
- "fundamental limitations" (根本限制)
- "generalizes to" (泛化到)

### ✅ 替换为
- "To our knowledge" (据我们所知)
- "in our evaluated settings" (在我们评估的设置中)
- "is associated with" (与...相关)
- "achieves higher mean" (达到更高的平均值)
- "preliminary evidence suggests" (初步证据表明)
- "remains an open question" (仍是开放问题)
- "requires further investigation" (需要进一步研究)

---

## 🎯 修改优先级

### 🔴 Critical（必须修改）
1. **Related Work**: 添加positioning段落 + novelty table
2. **Limitations**: 完全重写（6个subsections）
3. **Abstract contribution 2**: 去除"improves" causal claim

### 🟠 High Priority（强烈建议）
4. **Results**: 添加statistical limitations footnote
5. **Conclusion**: 添加scope和uncertainty
6. **Introduction contributions**: 缩小scope

### 🟡 Medium Priority（有时间就改）
7. **标题**: 去掉"Autonomous"和"Realistic"
8. **全文**: 替换绝对化表述

---

## 📝 修改后重新编译检查清单

```bash
cd paper/latex
pdflatex acl_latex.tex
bibtex acl_latex
pdflatex acl_latex.tex
pdflatex acl_latex.tex
```

### 检查项：
- [ ] PDF编译成功，无错误
- [ ] 新增的Table显示正常
- [ ] Limitations章节格式正确
- [ ] 所有修改的文本显示正确
- [ ] Reference没有遗漏

---

## ⏰ 时间安排建议

如果今天必须提交，建议：

**1小时紧急修改**：
1. Abstract contribution 2（去除improves）
2. Related Work添加positioning段落
3. Conclusion添加uncertainty表述

**2-3小时完整修改**：
1. 上述所有Critical修改
2. Results添加statistical footnote
3. Limitations完全重写

**4+小时充分修改**：
1. 所有Critical + High Priority修改
2. 全文替换绝对化表述
3. 标题修改

---

## 💡 最终建议

**最低限度**（1小时）：
- Abstract: "improves" → "is associated with higher metrics"
- Related Work: 添加positioning段落
- Conclusion: 添加"in our evaluated settings"

**推荐修改**（2-3小时）：
- 上述最低限度
- Limitations完全重写
- Results添加statistical footnote
- Introduction contributions缩小scope

**理想修改**（4+小时）：
- 所有Critical + High Priority
- 标题修改
- 全文语言polish

---

**记住**：目标是降低over-claiming风险，不是重写整个论文。专注修改claims的语言表述，而不是改变论文结构或实验结果。
