# Run matrix

All arms prescribe frozen local Qwen3.5-4B and common splits, anchor, context, replacement rule and validation budget. The editable arm can propose selector edits and execute them. The fixed arms restrict edits to the other roles and use their assigned selector. Actual call and token counts can differ. [Registration](preregistration.md) · [Completeness clarification](amendments.md) · [Editing interface](../src/rqstudy/domain.py)

| Stage | Seeds | Arm | Scheduled validation per run | Endpoint set |
|---|---|---|---|---|
| Root | 0 | Initial program, no search | 0 | Pilot first; sealed root after freeze |
| Pilot | 0 | Editable, native clade, uniform, released handcrafted | 64 | 24 pilot-only manuscripts |
| Confirmatory primary | 1–8 | Editable and native clade | Common timing-selected budget from 64, 128, 256 | Same 100 sealed manuscripts |
| Secondary comparison | 1–5 | Fixed uniform and fixed handcrafted | Same budget | Same 100 |
| Midpoint intervention | 1 | Continuation of editable state with native override | Remaining half of common budget | Same 100 |
| Historical formula replay | All expansions | Released exponential vs paper reciprocal penalty | No model calls | Parent-ID disagreement |

The local dashboard updates search validation observations during a run. Published panels also show verified endpoint evaluations. Pilot endpoints use the pilot split. The joint freeze binds all available confirmatory endpoints before test scoring, and records missing configurations. [Campaign](../src/rqstudy/campaign.py) · [Endpoint checks](../src/rqstudy/endpoints.py)
