# Paper Improvement Log

## Score Progression

| Round | Score | Verdict | Key Changes |
|-------|-------|---------|-------------|
| Round 0 (original) | 5.0/10 | No | Baseline: overclaiming, weak baselines, under-specified problem |
| Round 1 | 5.5/10 | No | Fixed claim calibration, added data sources, acknowledged limitations |
| Round 2 | 5.5/10 | No | Further refinements needed (empirical issues remain) |

## Round 1 Review & Fixes

<details>
<summary>GPT-5.4 xhigh Review (Round 1) - Score: 5/10</summary>

**CRITICAL Issues Identified:**
- W1: "high-fidelity" and "real-world data" claims not validated with concrete sources
- W2: Empirical evidence for framework superiority too weak (only 3 runs, no statistical significance)
- W3: Baselines too weak - no classical OR/control methods

**MAJOR Issues Identified:**
- W4: MDP formulation under-specified, unclear if POMDP
- W5: No primary benchmark objective defined
- W6: Failure analysis descriptive, not quantitative

**Writing Issues:**
- Overclaiming terms throughout ("extensive", "high-fidelity", "principled")
- Narrative split between benchmark and method paper
- Inconsistent certainty levels between sections

</details>

### Fixes Implemented in Round 1

**CRITICAL Fixes:**
1. **Added Benchmark Validity Details** (Environment Construction section):
   - Added "Design Rationale and Data Sources" paragraph
   - Explicitly mentioned Dominick's dataset for demand distributions
   - Cited OR literature for supply-chain parameters
   - Described template-based news generation
   - Added acknowledgment that environment is "simplified abstraction"

2. **Softened Empirical Claims** (throughout paper):
   - Abstract: "high-fidelity" → "simulation benchmark", "realistic" → "commercial scenarios"
   - Added "in our evaluated settings" qualifiers
   - Added "due to limited experimental runs" for statistical limitations
   - Introduction: "extensive experiments" → "experiments across eight models"
   - Conclusion: "principled testbed" → "testbed"

3. **Acknowledged Weak Baselines** (Related Work + Limitations):
   - Added "Classical operations research baselines" paragraph to Related Work
   - Explicitly listed missing methods: base-stock policies, newsvendor, dynamic pricing, MPC, approximate DP
   - Updated Limitations to mention specific missing OR methods

**MAJOR Fixes:**
4. **Clarified Problem Formulation** (Environment Construction):
   - Renamed section to "State Space and Observations"
   - Added explicit POMDP discussion
   - Added detailed tool access description with 6 specific tools

5. **Defined Primary Evaluation Objective** (Experiment section):
   - Added "Primary Evaluation Objective" paragraph
   - Explained multi-objective nature and metric tradeoffs
   - Specified Days and Daily Income as primary indicators

6. **Added Quantitative Failure Diagnostics** (Analysis section):
   - Decision Coverage Ratio for scalability analysis
   - Information Coverage Rate for information gathering
   - Strategy Churn Metrics (3 measures) for temporal instability
   - Action Validity Metrics (2 measures) for hallucinations

**Writing Improvements:**
- Improved consistency between abstract and conclusion
- Made claims more cautious with appropriate qualifiers
- Added specific details about data sources and validation
- Removed overclaiming terms throughout

## Round 2 Review

<details>
<summary>GPT-5.4 xhigh Review (Round 2) - Score: 5.5/10</summary>

**Strengths Noted:**
- Claim calibration substantially improved; wording more defensible
- Partial observations and tool access now clear
- Explicit diagnostic metrics are scientific improvement
- Paper now reads as more self-aware and credible

**Remaining CRITICAL Issues:**
- Framework gains still under-supported (3 runs, no statistical significance)
- Strong OR/control baselines still missing (acknowledgment not enough)

**Remaining MAJOR Issues:**
- Benchmark validity improved but not quantitatively validated
- No official evaluation protocol/leaderboard defined
- New diagnostic metrics need systematic reporting

**Verdict:** Still not submission-ready for top-tier venues, but improvements made

</details>

### Remaining Issues Requiring Empirical Work

The Round 2 review identified issues that require additional experiments and implementation beyond writing improvements:

1. **Statistical Significance**: Need more runs (currently 3), confidence intervals, paired significance tests
2. **OR Baselines**: Need to implement base-stock, newsvendor, MPC, or approximate DP baselines
3. **Benchmark Validation**: Need quantitative validation comparing simulator to real retail statistics
4. **Evaluation Protocol**: Need official leaderboard protocol with clear scoring

These would require substantial additional work (weeks to months) and are beyond the scope of writing-only improvements.

## PDFs Generated

- `acl_latex_round0_original.pdf` — Original paper before improvements
- `acl_latex_round1.pdf` — After Round 1 fixes (current version)
- `acl_latex.pdf` — Latest version (same as Round 1)

## Compilation Status

✅ LaTeX compiles successfully with 0 errors
⚠️ 6 overfull box warnings (mostly in figures/tables)
⚠️ 192 underfull box warnings (loose spacing, generally acceptable)

## Summary for Tomorrow's Submission

**What Was Improved:**
- Claim calibration is now much more defensible
- Data sources and design rationale are now explicit
- Problem formulation (MDP/POMCP) is clarified
- Limitations are acknowledged concretely
- Writing is more consistent and cautious

**What Remains for Top-Tier Acceptance:**
- Statistical rigor (more experimental runs, significance testing)
- Stronger baselines (classical OR methods)
- Benchmark validation (quantitative comparison to real data)
- Clear evaluation protocol

**Recommendation:**
The paper is substantially improved in terms of writing quality and claim calibration. However, the empirical limitations identified by the reviewer would need to be addressed for acceptance at ACL/ICLR/NeurIPS level venues. Consider:
- Targeting a venue that accepts preliminary work or shorter papers
- Adding the empirical work as future work in the submission
- Being very explicit about limitations in the submission
- Considering this as a solid draft for ArXiv/workshop while continuing empirical work

## Key Changes by File

- **abstract.tex**: Softened "high-fidelity", "realistic", "extensive experiments"
- **introduction.tex**: Removed "real-world commercial data" overclaim, softened "extensive experiments"
- **environment_construction.tex**: Added data sources paragraph, clarified POMDP, added tool details
- **experiment.tex**: Added primary evaluation objective explanation
- **analysis.tex**: Added quantitative diagnostic metrics for all failure modes
- **related_work.tex**: Added classical OR baselines acknowledgment paragraph
- **limitations.tex**: Expanded baseline limitations with specific OR methods
- **conclusion.tex**: Softened "principled testbed", added specific qualifiers
