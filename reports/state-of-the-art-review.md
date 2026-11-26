# Source and implementation audit

| Source | Reuse or evidence | Difference material to this experiment |
|---|---|---|
| [OpenRQGM at feecb32](https://github.com/DreamFallenFlowers/OpenRQGM/tree/feecb32d9fbdd1c60749ade9ecb47f971e12666e) | Apache-licensed archive, utility records, evaluator objects, posterior quantiles, scheduling, persistence and audit tests | Its engine calls the clade sampler for expansion and for evaluation. This matches [June Algorithm 1](https://arxiv.org/pdf/2606.26294v2#page=19). [October Algorithm 1](https://arxiv.org/html/2606.26294v3#A2.SS4) uses own-node counts for evaluation and demand-driven training feedback on expansion. |
| [OpenRQGM published reconstruction](https://github.com/DreamFallenFlowers/OpenRQGM/blob/feecb32d9fbdd1c60749ade9ecb47f971e12666e/docs/results/paper-matched-v5-summary.json) | One coding search with archived validation outcomes, replacements and erased records. [Reproduced source values](literature-review.md#published-comparison-data) | The artifact marks direct comparison with the paper as invalid. The independent audit reran the pinned archive, scheduler, engine, statistics and resume fixtures. [Test evidence](../results/audit-observation.json) |
| [HyperAgents released rule](https://github.com/facebookresearch/HyperAgents/blob/59a68f672dfb92c74aeb7e61535d776fb36e172d/utils/gl_utils.py#L571-L584) | Dynamic midpoint from the three highest available scores, sigmoid slope ten, exponential child penalty | [Appendix A.2](https://arxiv.org/pdf/2603.19461v2#page=24) prints a reciprocal penalty. Use the released version for end-to-end search and replay both from identical snapshots. |
| [kingkillery independent RQGM](https://github.com/kingkillery/red-queen-godel-machine/blob/2cea930b09e1df0e564efb0d6e77ac5154e52be7/src/rqgm/search.py#L197-L221) | Cross-check of posterior implementation and epoch logic | It also uses subtree evidence for evaluation. OpenRQGM supplies the primary implementation and serialized RNG state. |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) and [Ollama](https://docs.ollama.com/api/chat) | Frozen local Q4_K_M model, streaming and token counts | The hardware and model differ from both published experiments. Verify GPU residency, context coverage and actual compute. |

## Contribution under test

The released HyperAgents weighting formula uses a candidate set with available upstream domain scores. Our adapter scores every eligible node using two role posterior means. The specific comparison therefore concerns the local adapter. [Released candidate set](https://github.com/facebookresearch/HyperAgents/blob/59a68f672dfb92c74aeb7e61535d776fb36e172d/utils/gl_utils.py#L523-L584) · [Local formula](../src/rqstudy/policies.py) · [Independent source correspondence](reviews/independent-scientific-audit.md)

The experiment joins the newest-agent selector of HyperAgents E.5 to the co-evolving writer and reviewer of RQGM. Its contribution is a specified comparison against native RQGM, diagnostic replay and a matched-state intervention. Neither cited publication reports this exact local APReS comparison. Upstream coding scores and joint robotics scores use different outcome scales.

## Sources of uncertainty

Historical human labels can be noisy. Conference status strings and duplicated versions can leak outcomes. Long manuscripts require a reading procedure whose information loss may vary by paper. Some papers may already have appeared in model pretraining. Repeated decisions on the same paper share uncertainty. Track cleaning, input route, paired search seeds and crossed-paper uncertainty explicitly. [RQGM data protocol](https://arxiv.org/html/2606.26294v3#A3.SS6) · [HyperAgents discussion](https://arxiv.org/pdf/2603.19461v2#page=63)
