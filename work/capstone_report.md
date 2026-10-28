# Content Opportunity Scoring: A Machine Learning Approach to Prioritizing Content Review

**Author:** Vasavi Joshi  
**Track:** FlyRank AI Machine Learning Internship  
**Lane:** Refresh / Content Opportunity Scoring  
**Repository:** [Vasavijoshi/flyrank-ml-internship](https://github.com/Vasavijoshi/flyrank-ml-internship)  
**Date:** October 28, 2026

---

## Abstract

This study investigates how search, engagement, freshness, and content characteristics can help prioritize pages for content-team review. A transparent rule-based baseline was compared with a Random Forest classifier using an anonymized 30,000-page starter dataset and a client-grouped holdout. The Random Forest measured Precision@10, Precision@50, and Precision@100 of 0.50, 0.64, and 0.61, respectively, compared with 0.30 at all three cutoffs for the baseline. A separate random-row-split evaluation produced higher metrics, but its potential client overlap makes it an optimistic reference rather than the primary result. The resulting ranked queue is intended as directional decision support for human review, not as proof of future decline or evidence that refreshing a page will cause better search performance.

---

## 1. Introduction and problem statement

Content teams may manage many pages and have limited time to investigate them. A practical question is which pages deserve attention first for refreshing, expansion, protection, pruning, or monitoring.

This project asks:

**Which content pages should be reviewed first using measurable search visibility, engagement, freshness, and content characteristics?**

The unit of analysis is one content page. The intended output is a ranked review queue with a model-based decline probability, a reason code, and a suggested review action.

A rule-based score is useful because it is transparent and easy to interpret. However, it combines only a small number of predefined signals. A Random Forest can combine multiple inputs and capture nonlinear relationships, potentially improving prioritization.

The purpose is not to automate editorial decisions. It is to help reviewers decide where to investigate first.

---

## 2. Data and data safety

### Dataset and release

The FlyRank ML Internship provides a search-intelligence warehouse release described as approximately 79 million rows, along with a smaller anonymized starter dataset.

**The reported modeling and validation results in this paper use the 30,000-page starter dataset, containing 44 columns. The Random Forest was not trained on the full 79-million-row warehouse.**

The larger warehouse was used in earlier data-contract work to investigate the intended daily search-performance grain and the February 2026 feature window.

### Data-contract verification

The earlier data-contract work examined `fact_content_daily_performance` and used `dim_content` when content metadata was required.

The February 1–28, 2026 feature window was paired with a decision cutoff at the end of February. The warehouse checks covered 7,351,108 daily-grain rows; GSC data was available for 2,621,783 of those rows.

These warehouse checks established context for the feature contract. They are distinct from the starter-dataset model evaluation reported here.

### Modeling features

The Random Forest used 12 features:

- `days_since_last_update`
- `impressions_90d`
- `clicks_90d`
- `sessions_90d`
- `ctr`
- `avg_position`
- `engagement_rate`
- `scroll_rate`
- `content_age_days`
- `word_count`
- `search_volume`
- `cpc`

### Target and exclusions

The target proxy was defined as:

`is_declining_label = (trend_direction == "down")`

This is a dataset-derived proxy, not a guaranteed future outcome.

The fields `trend_direction` and `trend_pct` were excluded from model inputs because they are outcome-derived. The target itself was also excluded from the features. Pseudonymous identifiers `content_id` and `client_id` were not model inputs; `client_id` was used only to group rows for validation.

Future-window information and label-derived fields were excluded to reduce leakage risk.

### Public safety

This paper presents aggregate results and methodology. It does not publish client names, private search queries, or identifying client-page URLs. The ranked queue is intended for internal human review, not public identification of individual pages.

---

## 3. Baseline

The first approach was a transparent rule-based action score combining page freshness and search visibility.

### Scoring rule

Staleness points:

- 0 points for `days_since_last_update <= 90`
- 1 point for 91–180 days
- 2 points for more than 180 days

Visibility points:

- 0 points for fewer than 250 impressions
- 1 point for 250–499 impressions
- 2 points for at least 500 impressions

The two scores were added, giving a total from 0 to 4. Pages scoring at least 3 received the `Refresh` action; the remaining pages received `Monitor`.

The baseline was evaluated on the same client-grouped holdout as the Random Forest.

| Metric | Week-4 baseline |
|---|---:|
| Precision@10 | 0.30 |
| Precision@50 | 0.30 |
| Precision@100 | 0.30 |

The declining-label rate in the grouped test set was 51.1%. This base rate is useful context when interpreting the ranking results.

---

## 4. Model and methodology

### Model

The primary model was a Random Forest classifier with 300 trees, `random_state=42`, `min_samples_leaf=5`, and median imputation.

It used the 12 features listed in the Data section to estimate the probability associated with the declining-label proxy. Pages could then be ranked by this score for review.

The model was chosen because it can combine search performance, freshness, engagement, and content characteristics without requiring a single linear relationship between inputs and the target.

### Validation design

The primary evaluation used a client-grouped holdout, keeping each client's pages in one partition.

| Split statistic | Value |
|---|---:|
| Training rows | 23,837 |
| Training clients | 25 |
| Test rows | 6,163 |
| Test clients | 7 |
| Client overlap | 0 |
| Test-set declining-label rate | 51.1% |

This design better represents evaluation on unseen clients than a random row split, where pages from the same client may appear in both training and test data.

The main metric was Precision@K at K=10, 50, and 100 because the intended use is prioritizing a limited number of pages for human review.

### Leakage checks

The final feature audit excluded `trend_direction`, `trend_pct`, `is_declining_label`, `content_id`, and `client_id` from model inputs.

The audit found no forbidden features, no target included as an input, no missing final features, and no outcome-like feature names.

An earlier leakage experiment deliberately introduced a future outcome-derived feature and produced an artificially perfect score of 1.0000. This illustrates why future-window and label-derived information must remain excluded.

---

## 5. Results

### Random Forest versus baseline

Both methods were compared on the same client-grouped holdout using Precision@K.

| Metric | Baseline | Random Forest, grouped split |
|---|---:|---:|
| Precision@10 | 0.30 | **0.50** |
| Precision@50 | 0.30 | **0.64** |
| Precision@100 | 0.30 | **0.61** |

The Random Forest measured higher Precision@K at all three cutoffs on this holdout. At K=10, measured precision increased from 0.30 to 0.50; at K=50, from 0.30 to 0.64; and at K=100, from 0.30 to 0.61.

These are observed measurements on this dataset and split. They do not establish that the same performance will hold for every client or future time period.

### Validation sensitivity

The earlier random-row-split evaluation measured:

| Metric | Random row split | Grouped-client split |
|---|---:|---:|
| Precision@10 | 1.00 | 0.50 |
| Precision@50 | 0.96 | 0.64 |
| Precision@100 | 0.94 | 0.61 |

The random row split allowed pages from the same clients to appear in both partitions, making it an optimistic reference. The grouped-client split had zero client overlap and is the primary result for unseen-client decision support.

The difference demonstrates that validation design materially affects measured performance.

### Figure

The primary comparison chart is generated by the capstone notebook and saved at:

`work/figures/precision_at_k_baseline_vs_random_forest.png`

It plots the Week-4 baseline against the Random Forest on the grouped-client split. The random-row-split metrics are retained as a sensitivity comparison, not as the headline result.

---

## 6. Interpretation and limitations

### Feature interpretation

Permutation importance from the Random Forest evaluation provided a diagnostic view of the model's inputs.

| Feature | Mean permutation importance |
|---|---:|
| `impressions_90d` | 0.0605 |
| `avg_position` | 0.0350 |
| `content_age_days` | 0.0114 |
| `clicks_90d` | 0.0100 |
| `ctr` | 0.0095 |
| `scroll_rate` | 0.0044 |
| `engagement_rate` | 0.0026 |
| `sessions_90d` | 0.0021 |
| `cpc` | -0.0011 |
| `search_volume` | -0.0035 |
| `word_count` | -0.0131 |
| `days_since_last_update` | -0.0186 |

These values are model diagnostics, not causal effects. For example, negative permutation importance for `days_since_last_update` means that permuting this feature did not improve predictive performance on this evaluation set. It does not mean freshness causes worse outcomes.

The measured ranking performance suggests that combining multiple signals can be more useful than relying on the simple baseline alone on this particular holdout.

### Limitations

**Dataset scope.** The reported model results use the 30,000-page starter dataset, not the full warehouse. Earlier warehouse checks were data-contract work, not full-warehouse model evaluation.

**Proxy target.** The target is derived from an observed trend label and is not a guaranteed future outcome.

**Generalization.** The primary evaluation uses one grouped holdout containing seven test clients. It does not establish performance across all clients, future periods, content types, or search environments.

**Metric scope.** Precision@K measures the proportion of declining-label pages in the top-ranked set. It does not directly measure business value, editorial quality, or the impact of a refresh.

**No causal evidence.** No randomized refresh intervention was conducted. The analysis cannot establish that refreshing a page causes better search performance.

**Human review.** A highly ranked page may not need action, while a lower-ranked page may still be important. Reviewers must consider search intent, factual accuracy, business importance, seasonality, recent changes, and strategic content value.

The model must not autonomously publish, rewrite, delete, redirect, or otherwise modify content.

---

## 7. Ranked recommendations: Action playbook

The output is a ranked queue for human investigation, not an automatic content-change system. The model's decline probability is used for ranking, with search visibility as a secondary ordering signal.

### Priority levels

| Priority | Score threshold | Recommended next step |
|---|---|---|
| Priority review | `>= 0.70` | Investigate first and verify context |
| Review | `>= 0.50` and `< 0.70` | Investigate after the priority queue |
| Monitor | `< 0.50` | Continue monitoring unless other evidence appears |

These thresholds are operational prioritization choices, not calibrated guarantees of future decline.

### Reason codes

| Reason code | Trigger | Suggested action |
|---|---|---|
| `STALE_HIGH_VISIBILITY` | More than 180 days since update and at least 500 impressions | Priority refresh review |
| `STALE_PAGE` | More than 180 days since update | Investigate whether a refresh is appropriate |
| `HIGH_VISIBILITY_REVIEW` | At least 500 impressions and elevated model ranking | Review search performance and intent alignment |
| `AGING_PAGE` | More than 90 days since update | Monitor or investigate freshness |
| `MONITOR` | No higher-priority rule applies | Continue monitoring |

### Recommended workflow

**Rank → investigate → validate context → decide → act → monitor**

Before acting, a reviewer should verify search intent, factual accuracy, business importance, seasonality, recent changes, ranking context, signal reliability, and whether a page is strategically protected.

### Cost and value

The queue is intended to help reviewers use limited time more efficiently by identifying pages to investigate first. Its main cost is reviewer time spent on pages that ultimately need no action. It may also miss pages that need attention but rank lower.

Performance and queue concentration should be monitored over time. New labels or material changes in data should trigger fresh grouped validation before the model is relied upon again.

### No-go list

The model must not independently publish or rewrite content, delete pages, change redirects or canonicals, make factual or editorial decisions, or treat a score as proof that a refresh will improve performance.

---

## 8. Reproducibility

The project repository is:

[https://github.com/Vasavijoshi/flyrank-ml-internship](https://github.com/Vasavijoshi/flyrank-ml-internship)

### Notebooks

- [ML-04 — Data Contract](https://github.com/Vasavijoshi/flyrank-ml-internship/blob/main/work/notebooks/w03_data_contract.ipynb)
- [ML-07 — Baseline Action Score](https://github.com/Vasavijoshi/flyrank-ml-internship/blob/main/work/notebooks/w04_baseline_score.ipynb)
- [ML-08 — Capstone Modeling Lane](https://github.com/Vasavijoshi/flyrank-ml-internship/blob/main/work/notebooks/w05_model.ipynb)
- [ML-09 — Validation and Research Claim Audit](https://github.com/Vasavijoshi/flyrank-ml-internship/blob/main/work/notebooks/w06_validation_audit.ipynb)
- [ML-10 — Content Action Playbook](https://github.com/Vasavijoshi/flyrank-ml-internship/blob/main/work/notebooks/w07_action_playbook.ipynb)
- [ML-11 — Capstone](https://github.com/Vasavijoshi/flyrank-ml-internship/blob/main/work/notebooks/capstone.ipynb)

### Re-running the work

From a fresh clone:

```bash
git clone https://github.com/Vasavijoshi/flyrank-ml-internship.git
cd flyrank-ml-internship
pip install -r requirements.txt
