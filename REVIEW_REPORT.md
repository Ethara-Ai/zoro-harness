# RetailBench ACL 2026 Submission - Comprehensive Review Report

**Review Date**: 2026-03-16
**Reviewer**: GPT-5.4 (xhigh reasoning)
**Venue**: ACL 2026

---

## Executive Summary

### Overall Verdict
- **Current Status**: **Weak Reject** (Score: 2.5/5)
- **Acceptance Likelihood**: 20-35% as-is; 55-70% with targeted fixes
- **Primary Issues**: Over-claimed novelty, insufficient statistical rigor, weak causal evidence for architecture claims

### Bottom Line
This is a **promising benchmark/system paper** that falls below the ACL main-track acceptance bar in its current form. The work has valuable contributions (open-source retail simulation, multi-agent evaluation), but key claims are not adequately supported. With focused revisions on statistics, ablations, and baseline strength, this can become competitive.

---

## Critical Issues (Severity-Ordered)

### 🔴 CRITICAL: Novelty Over-Claiming
**Problem**: "First comprehensive benchmark" is not defensible without direct comparison to existing benchmarks.

**Impact**: Undermines credibility; reviewers will immediately flag this.

**Fix Required**:
- Add novelty matrix comparing to AgentBench, ToolBench, WebArena, ALFWorld, etc.
- Reframe from "first comprehensive" to specific unique combination of properties
- See Claim Wording section below for concrete examples

### 🟠 MAJOR: Causal Architecture Claims Under-Supported
**Problem**: "Step-Reflection +31%" and "Strategy-Execution stability" presented as mechanism claims without component ablations.

**Impact**: Architecture contribution is speculative rather than scientific.

**Fix Required**: Minimal ablation package (see Section 5)

### 🟠 MAJOR: Insufficient Statistical Rigor
**Problem**: Only 3 runs per model is inadequate for stochastic LLM-agent evaluation.

**Impact**: Reported rankings may be unstable; significance uncertain.

**Fix Required**: Seed expansion to 10-15 paired seeds for key comparisons

### 🟠 MAJOR: Weak Baselines
**Problem**: Missing strong classical OR/control baselines; hard to assess LLM value.

**Impact**: Cannot determine if benchmark is too easy or LLMs are genuinely capable.

**Fix Required**: Add (s,S) and newsvendor-style baselines under identical constraints

---

## Detailed Assessment by Section

### 1. Significance & Novelty

**What's Strong** ✅
- Interactive environment with multi-day horizon
- Tool-calling with business-relevant objective (profit)
- Open-source benchmark direction
- Cost-performance analysis

**What's Weak** ❌
- Novelty is mostly domain + packaging, not methodological leap
- Retail domain value not adequately justified
- Missing positioning relative to existing benchmarks

**Recommendation**:
- Center contribution on: **economic simulator + constrained tool ecosystem + multi-day planning tradeoffs**
- Argue retail tests: delayed rewards, perishability, inventory-price coupling, partial observability
- Use novelty matrix to show unique **combination** of properties

### 2. Methodological Rigor

**Metrics**: Partially adequate but incomplete
- Current: profit, sales, turnover, perishable rate
- **Missing**: stockout rate, service level, holding cost decomposition, ordering frequency

**Experimental Design**:
- 3 runs/model: **too weak** for stochastic systems
- No paired seeds across models/agents
- No confidence intervals or effect sizes
- No multiple-comparison correction

**Baselines**: Insufficient for top-tier venue
- Current: rule-based, Plan-and-Act variants
- **Missing**: Classical OR baselines, stronger agentic baselines under matched budgets

### 3. Missing Experiments (Priority-Ordered)

| Priority | Experiment | Why It Matters | Est. Cost |
|----------|-----------|----------------|-----------|
| **P0** | Seed expansion (10-15 paired seeds) | Validates significance | Medium |
| **P0** | Step-Reflection ablation | Supports causal claims | Low-Med |
| **P0** | Budget-matched comparisons | Prevents unfair compute advantages | Low |
| **P1** | Strong non-LLM baselines | Shows LLM value beyond OR | Med |
| **P1** | Horizon scaling (10/20/30 days) | Tests long-term planning | Med |
| **P1** | Scenario generalization | Tests OOD robustness | Med |
| **P2** | Human/expert baseline | Anchors practical significance | Med-High |
| **P2** | Cross-benchmark transfer | Tests broader generalization | High |

---

## Minimal Experiment Package (Highest ROI)

If you have limited compute (~2-3 weeks, ~50-100 GPUh), target this **core package**:

### Package A: Step-Reflection Causal Ablation (~40-60 runs)
**Model**: GPT-4o (cheapest strong model)
**Scenario**: Hard mode
**Conditions**:
1. `A0`: Plan-and-Act (no reflection)
2. `A1`: Step-Reflection (full)
3. `A2`: Sham reflection (same tokens, no critique)
4. `A3`: Sparse reflection (once/day)

**Seeds**: Start with 10 paired seeds; extend to 15 if CI crosses 0

### Package B: Top-Model Ranking Stabilization (~24 runs)
**Models**: GPT-5.2 vs DeepSeek-V3 (both Step-Reflection)
**Scenario**: Hard mode
**Seeds**: 12 paired seeds

### Package C: Classical Anchors (~20 runs, mostly CPU)
**Baselines**:
1. `(s,S) + rule pricing`
2. `Newsvendor + elasticity pricing`

**Seeds**: 10 each

**Total**: ~85-105 runs total

---

## Ablation Design: Step-Reflection

### Minimal Sufficient Design (4 Conditions)

| Condition | Description | Purpose |
|-----------|-------------|---------|
| A0 | Plan-and-Act (no reflection) | Baseline |
| A1 | Step-Reflection (full) | Test overall gain |
| A2 | Sham reflection | Test if gain from reflection quality vs compute |
| A3 | Sparse reflection | Test dose/frequency effects |

### Metrics to Defend Causal Claim
- **Primary**: Final cumulative profit
- **Secondary**: Stockout rate, perishable loss, inventory turnover, tool-error rate
- **Process**: Action-revision rate, failure-recovery next day

### Statistical Criteria
- Paired tests (bootstrap/permutation)
- Holm correction for multiple contrasts
- Report effect size + CI, not just p-values

---

## Non-LLM Baseline Implementation

### Must-Have #1: (s,S) Inventory + Markdown Heuristic
```
Forecast: Exponential smoothing from sales history
Reorder: If inventory_position < s, order to S
Pricing: Markdown near-expiry, markup on stockout risk
```

### Must-Have #2: Newsvendor + Discrete Price Optimization
```
Demand: Estimate distribution per SKU
Order: Newsvendor quantile using underage/overage costs
Price: Grid search maximizing expected one-day margin
```

### Implementation Constraints
- Use **identical tool APIs** as LLM agents
- Same tool-call budget and observation limits
- No privileged state access
- Output tool calls in same format as agent logs

### Success Criteria
**LLM adds value** if:
- Statistically significant profit gain over best classical baseline
- Acceptable waste/stockout tradeoff
- Under matched compute budgets

---

## Claim Wording Fixes (Before → After)

### Novelty/Benchmark
❌ **Before**: "RetailBench is the first comprehensive benchmark for LLM sequential decision-making."
✅ **After**: "RetailBench is an open-source retail-focused benchmark for long-horizon tool-using decision-making, with coupled pricing and inventory control under stochastic demand."

❌ **Before**: "We provide state-of-the-art evaluation."
✅ **After**: "We provide a controlled, reproducible evaluation protocol for LLM agents in a retail operations simulator."

### Architecture Findings
❌ **Before**: "Step-Reflection significantly improves performance by 31%."
✅ **After**: "In our current setting, Step-Reflection is associated with a +31% mean profit improvement over Plan-and-Act; ablations test whether this gain is attributable to reflection rather than added compute."

❌ **Before**: "Strategy-Execution is robust and stable."
✅ **After**: "Strategy-Execution shows higher day-to-day strategy consistency (78% in our metric), though robustness depends on scenario and model family."

### Evaluation Claims
❌ **Before**: "GPT-5.2 is best across difficulty levels."
✅ **After**: "GPT-5.2 attains the highest mean profit in our runs; we report confidence intervals and pairwise win rates to reflect ranking uncertainty."

❌ **Before**: "GPT-4o delivers 80% performance at 30% cost."
✅ **After**: "Under our token-pricing assumptions and protocol, GPT-4o achieves ~80% of GPT-5.2 mean profit at ~30% inference cost."

### Generalization
❌ **Before**: "Our results generalize to real-world sequential decision-making."
✅ **After**: "RetailBench isolates key properties of sequential economic control; external validity beyond retail simulation remains to be established."

❌ **Before**: "Our methods broadly transfer to other domains."
✅ **After**: "Transfer to other domains is a hypothesis for future work; current evidence is limited to retail scenarios in RetailBench."

---

## Mock ACL Review

### Summary (2-3 sentences)
This paper introduces RetailBench, an open-source simulator for evaluating LLM agents on multi-step retail operations decisions with tool use. The authors benchmark multiple frontier models and propose agent variants (Plan-and-Act, Step-Reflection, Strategy-Execution), reporting sizable differences in profit and strategy stability. The topic is timely and potentially impactful, but key claims currently outpace the strength of evidence.

### Strengths (3-4 bullets)
- ✅ Practical and relevant task: long-horizon, economically meaningful objective, tool-mediated interaction
- ✅ Open-source benchmark direction valuable for reproducibility and future comparisons
- ✅ Includes multiple model families and reports cost/performance tradeoffs
- ✅ Architecture variants interesting and could yield actionable insights

### Weaknesses (4-5 bullets)
- 🔴 **CRITICAL**: Novelty claim ("first comprehensive benchmark") insufficiently substantiated against existing benchmarks
- 🟠 **MAJOR**: Causal architecture claims not adequately supported without tighter ablations (including compute-matched controls)
- 🟠 **MAJOR**: Statistical rigor limited (few seeds), making rankings and gains potentially unstable
- 🟠 **MAJOR**: Baseline set underpowered; missing strong non-LLM OR/control baselines
- 🟡 **MINOR**: Generalization language extends beyond demonstrated scope

### Questions for Authors (3-4 questions)
1. Can you provide paired-seed significance tests and confidence intervals for main model and architecture comparisons?
2. How much of Step-Reflection's gain remains under compute/token-matched controls (e.g., sham reflection)?
3. How do strong classical baselines ((s,S), newsvendor-style policies) perform under identical tool and information constraints?
4. Which properties are unique to RetailBench relative to AgentBench/ToolBench/WebArena, and which are inherited?

### Score
- **2.5 / 5** (borderline Weak Reject)
- **Confidence**: 0.82

### What Would Move Toward Accept (Checklist)
- [ ] Add robust statistics: paired seeds (≥10-15), CIs, effect sizes, multiple-comparison tests
- [ ] Add minimal causal ablation package for Step-Reflection
- [ ] Add at least two strong non-LLM baselines under identical constraints
- [ ] Replace broad novelty claims with precise novelty matrix
- [ ] Reframe conclusions with scoped, variance-aware claims

---

## Novelty Matrix Design

### Benchmarks to Compare
- AgentBench
- ToolBench
- WebArena
- ALFWorld
- ScienceWorld
- Mind2Web (optional)
- GAIA (optional)

### Comparison Dimensions
| Dimension | Description |
|-----------|-------------|
| Interactive multi-step environment | Yes/No |
| Typical/max horizon length | Number of steps |
| Delayed scalar reward | Yes/No |
| Economic objective (profit/cost) | Yes/No |
| Stochastic dynamics | Yes/No |
| Coupled decisions (pricing + inventory) | Yes/No |
| Tool-call requirement & budget | Yes/No |
| Unstructured text signal integration | Yes/No |
| Open-source simulator | Yes/No |
| Cost-aware evaluation | Yes/No |

### Defensible Claim Framing
✅ **Use**: "To our knowledge, first open-source benchmark combining [property A] + [property B] + [property C]"
❌ **Avoid**: "First comprehensive benchmark for LLM sequential decision-making"

---

## Paper Structure Redesign

### Recommended Section Outline

1. **Introduction**
   - Research question: Which design choices improve long-horizon economic decisions under tool constraints?
   - Contributions: benchmark + rigorous protocol + architecture study

2. **Related Work + Positioning**
   - Benchmark matrix vs prior work
   - Precise novelty scope (no broad "first comprehensive")

3. **RetailBench Environment**
   - Dynamics, tools, reward structure, difficulty modes
   - Why retail is a hard sequential testbed

4. **Agent Designs and Hypotheses**
   - Plan-and-Act, Step-Reflection, Strategy-Execution
   - **Explicit hypotheses per design choice**

5. **Experimental Protocol**
   - Paired seeds, statistical tests, budget matching, reproducibility

6. **Results**
   - RQ1: Model ranking with uncertainty
   - RQ2: Architecture ablations
   - RQ3: Classical baseline comparison
   - RQ4: Cost-performance frontier

7. **Analysis**
   - Failure taxonomy, tool-use breakdown, where agents fail

8. **Limitations and Scope**
   - Domain specificity, simulator realism limits, external validity boundaries

9. **Conclusion**

### Figure/Table Plan
- **Table 1**: Novelty matrix vs prior benchmarks
- **Figure 1**: Environment/tool interaction loop
- **Table 2**: Benchmark statistics and difficulty settings
- **Table 3**: Main results with confidence intervals
- **Figure 2**: Rank stability / win-rate plots
- **Figure 3**: Ablation results
- **Table 4**: Non-LLM vs LLM comparison
- **Figure 4**: Cost-performance frontier
- **Figure 5**: Failure-case timeline

### Balance: Benchmark-First
- **Benchmark**: ~65% (environment, protocol, results)
- **Architecture**: ~35% (designs, ablations, analysis)

---

## Claims Matrix for Different Outcomes

### If Step-Reflection Shows No Significant Gain
**Can still claim**:
- "Reflection variant is competitive but not reliably superior"
- Benchmark sensitivity and need for stronger architectural inductive biases

**Must drop**:
- Causal claims about reflection mechanism

### If Non-LLM Baselines Match/Beat LLMs
**Keep**:
- Benchmark contribution (environment, protocol)

**Reframe to**:
- "Current LLM agents underperform strong OR heuristics in structured retail control"
- Hybrid methods as main insight

### If Seed Expansion Shows Unstable Rankings
**Drop**:
- Strict leaderboard ordering

**Replace with**:
- Performance tiers, win-rates, variance-aware conclusions
- Contribution shifts to "robust evaluation protocol in stochastic agent benchmarks"

### If All Three Are Positive
**Can claim**:
- Strong empirical evidence for architecture benefit
- Benchmark discriminative power

---

## Practical 3-Week Revision Plan

### Week 1: Infrastructure & Baselines
- [ ] Implement two classical baselines ((s,S), newsvendor)
- [ ] Build evaluation harness for paired seeds
- [ ] Set up statistical analysis pipeline (CIs, effect sizes, paired tests)

### Week 2: Core Experiments
- [ ] Run Step-Reflection ablations (A0-A3, 10 seeds each)
- [ ] Run top-model stabilization (GPT-5.2 vs DeepSeek, 12 seeds)
- [ ] Run classical baselines (10 seeds each)

### Week 3: Analysis & Writing
- [ ] Statistical analysis with proper corrections
- [ ] Rewrite all claims with scoped language
- [ ] Build novelty matrix table
- [ ] Rebuild figures with error bars
- [ ] Add ablation section
- [ ] Strengthen limitations section

---

## Priority Action Items (Ranked)

### Immediate (This Week)
1. **Rewrite novelty claim** - Add comparison matrix to existing work
2. **Add confidence intervals** to all main results (even with current data)
3. **Tighten all claim language** - Use "associated with" instead of "causes"

### High Priority (Week 1-2)
4. **Implement (s,S) baseline** - ~1-2 days coding
5. **Implement newsvendor baseline** - ~1-2 days coding
6. **Step-Reflection ablations** - Core causal evidence

### Medium Priority (Week 2-3)
7. **Seed expansion** for key comparisons
8. **Add missing metrics** (stockout rate, service level)
9. **Failure case analysis** with taxonomy

### If Time Permits
10. Horizon scaling experiments
11. OOD scenario tests
12. Human/expert baseline

---

## Final Recommendations

### What to Keep ✅
- Open-source benchmark direction
- Multi-agent evaluation across model families
- Cost-performance analysis
- Retail domain focus (with better justification)

### What to Change 🔧
- All claim language (tighten scope)
- Statistical approach (add rigor)
- Baseline set (add classical OR)
- Novelty framing (be precise)

### What to Add ➕
- Novelty matrix table
- Ablation studies
- Confidence intervals everywhere
- Stronger limitations section
- Failure case taxonomy

### What to Remove ❌
- "First comprehensive benchmark" language
- Broad generalization claims
- Unsubstantiated causal statements
- Over-confident ranking claims

---

## Summary

This paper has **solid raw material** but needs methodological tightening to meet ACL standards. The good news is that the core contributions (benchmark + multi-agent evaluation) are valuable; the bad news is that current claims outpace evidence.

**Key insight**: Shift from "impressive leaderboard numbers" to "defensible scientific evidence + scoped claims." With focused work on statistics, ablations, and baseline strength (≈60-85 additional runs), this can become a competitive ACL submission.

**Most critical fix**: Add robust statistics and classical baselines. Everything else is secondary.

---

**Thread ID for future reference**: `019cf5ae-55ec-7002-bf99-0bbebbf4ddaa`
