# Literature review

## Research question

Does permitting the editor to modify and execute its parent-selection rule improve held-out prediction of historical conference decisions at a fixed validation budget? The registered target uses a class-balanced sample. Reviewer quality has additional dimensions beyond this measured agreement. [Preregistration](preregistration.md) · [Audit of evaluation and generalization](reviews/independent-scientific-audit.md)

## What prior work establishes

- DGM accumulates executable variants in an archive and can branch from ancestors. HGM estimates how productive an ancestor is from validation outcomes throughout its subtree and uses Thompson sampling. Its implementation also scales both Beta parameters as the remaining budget decreases. This motivates a strong fixed selector comparator. [DGM](https://arxiv.org/abs/2505.22954) · [HGM paper](https://arxiv.org/abs/2510.21614) · [HGM sampler](https://github.com/metauto-ai/HGM/blob/013872d95da978483f5b540e531db063d23890da/hgm.py#L360-L376)
- HyperAgents integrates the task agent and its editor in each workspace. Appendix E.5 initializes a uniform selector, uses the latest eligible agent's selector, and edits its chosen parent. The authors describe UCB-like rules in each editable run. Their joint paper-review/robotics scores and statistical qualifier are reproduced below from the original publication. [Appendix E.5 and Algorithm 5](https://arxiv.org/pdf/2603.19461v2#page=60) · [Released dispatch](https://github.com/facebookresearch/HyperAgents/blob/59a68f672dfb92c74aeb7e61535d776fb36e172d/generate_loop.py#L100-L175)
- RQGM uses a frozen evaluator in each epoch, replacement against a fixed human anchor, and erasure of dependent measurements after replacement. October Algorithm 1 samples expansion parents using clade counts and evaluation targets using individual node counts. Paper writers receive a title and abstract. Human conference decisions anchor reviewers. [Method](https://arxiv.org/html/2606.26294v3#A2.SS4) · [Benchmark](https://arxiv.org/html/2606.26294v3#A3.SS6)
- APRES studies peer review and revision. The HyperAgents release provides paper text, an identifier, and the decision. The released paper-review utility passes only paper text to the reviewer. An upstream issue reports test labels in a meta-agent workspace. We use a trusted splitter, label-free inputs, training-only feedback, and isolated selector execution. Public pretraining exposure remains possible. [APRES](https://arxiv.org/abs/2603.03142) · [Domain data and utilities](https://github.com/facebookresearch/HyperAgents/tree/59a68f672dfb92c74aeb7e61535d776fb36e172d/domains/paper_review) · [Access report](https://github.com/facebookresearch/HyperAgents/issues/39)

## Design and interpretation

Pair independent search seeds, hold paper identities common across conditions, and resample both crossed dimensions for uncertainty. Record operational failures and all preregistered seeds. A single pilot can establish that a selector is executable and affects a choice. Estimating a performance advantage requires further searches and sealed scoring. [Variance in ML benchmarks](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html) · [Crossed bootstrap](https://artowen.su.domains/reports/pbs2.pdf) · [ML pitfalls](https://arxiv.org/abs/2108.02497) · [NeurIPS checklist](https://neurips.cc/public/guides/PaperChecklist)

We distinguish diagnostic replay on a fixed archive from a causal intervention that changes future descendants. The paired searches and midpoint fork test the latter. Keep interpretations conditional on the sampled papers, search budgets, inference model and decoding procedure. [Schulman's experimental guidance](http://joschu.net/blog/opinionated-guide-ml-research.html) · [Research process](https://www.lesswrong.com/posts/hjMy4ZxS5ogA9cTYK/how-i-think-about-my-research-process-explore-understand) · [Claims and evidence](https://arxiv.org/abs/1807.03341)

<!-- GENERATED: literature -->
## Published comparison data

HyperAgents reports joint paper-review and robotics test performance across 5 searches per condition. The authors report p > 0.05 for differences between methods. [Appendix E.5](https://arxiv.org/pdf/2603.19461v2#page=60)

| Published selector | Joint test score | Authors’ reported interval |
| --- | --- | --- |
| fixed uniform | 0.428 | 0.407–0.432 |
| editable | 0.491 | 0.387–0.512 |
| handcrafted | 0.532 | 0.384–0.586 |

OpenRQGM's coding reconstruction records 12,288 validation outcomes, 4 evaluator replacements and 2,182 erased records. The original record sets its direct paper-comparison validity flag to false. [Pinned reconstruction result](https://raw.githubusercontent.com/DreamFallenFlowers/OpenRQGM/feecb32d9fbdd1c60749ade9ecb47f971e12666e/docs/results/paper-matched-v5-summary.json) · [Extracted values and checksums](../results/published-source-evidence.json)
