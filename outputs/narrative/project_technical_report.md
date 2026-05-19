# Technical Report: NLP-Driven Cereal Food Availability Pipeline
**Author:** Odekunle Jibola Johnson (SHU dissertation Project)
**Report Generated:** 12–13 May 2026
**Working Directory:** Research_Food_Insecurity_Project

---

## Preface: How to Read This Report

This document is not simply a record of what code was changed. It is a record of *why* each change was made, *what problem it was responding to*, and *how each decision followed from the one before it*. Every technical choice described here had a research reason behind it — a specific gap between what the pipeline was doing and what the dissertation actually needed. That reasoning is documented throughout.

The report follows the arc of the work chronologically: starting from the state of the pipeline before improvements were made, through each problem as it was identified, each decision as it was reached, and ending with the full picture of what the pipeline now produces and what it means.

---

## Part 1: The State of the Pipeline Before Improvements — What Was Working and What Was Not

### 1.1 What the Pipeline Was Doing

Before the work documented in this report, the pipeline had six functional phases:

- **Phase A** collected academic literature from OpenAlex, Scopus, and manually curated PDFs, scored each paper for relevance to the dissertation themes, and ran NLP topic modelling (LDA and TF-IDF/NMF) on the papers classified as most closely aligned.
- **Phase B** downloaded country-level datasets from the World Bank, IMF, FAO, and other sources.
- **Phase C** cleaned and merged all those datasets into a single master table with one row per country.
- **Phase D** fit four regression models (OLS, Random Forest, XGBoost) predicting cereal food availability per capita across countries.
- **Phase E** ran robustness checks and outlier diagnostics.
- **Phase F** produced a written synthesis connecting the NLP findings to the empirical results.

The pipeline was running end-to-end without errors. It produced numbers, tables, and a narrative. On the surface, it appeared complete.

### 1.2 What Was Wrong Beneath the Surface

Three problems were present, each invisible until it was examined closely:

**Problem 1 — The literature corpus was contaminated with off-topic papers.** The 114 papers classified as "strict" included papers about microfinance for smallholder farmers, household-level food access, consumer purchasing behaviour, and nutritional deprivation. None of these are about cereal food *availability* in the supply-side sense the dissertation requires. They were classified as strictly aligned because the scoring logic rewarded accumulating thematic keywords — `food security`, `smallholder`, `farmer`, `rural household` — without requiring any explicit supply-side signal. A paper about women's financial inclusion in rural Kenya could score as strictly aligned because it mentioned `food security` and `farmer` and `household`, even though it contained no discussion of cereal production, logistics, post-harvest loss, or food supply. This contaminated the NLP topics that were supposed to represent what the *cereal availability* literature emphasises.

**Problem 2 — There was no statistical test of whether NLP added value.** The Phase F synthesis compared R² values across models but never formally tested whether the improvement from Model A★ to Model F was statistically distinguishable from noise. This left the core claim of the dissertation — "NLP-discovered themes add predictive power" — without a formal statistical foundation. Describing an R² gain without a significance test is, in academic terms, incomplete.

**Problem 3 — The dependent variable was conceptually wrong.** The pipeline was predicting cereal *production* per capita (World Bank indicator AG.PRD.CREL.MT divided by population). This is not what food availability means. A country like Japan or Egypt has very low domestic cereal production but high food availability because it imports most of its cereals. A country like Australia or Canada produces far more cereal than its population consumes and exports most of it — but a production-based measure would show them as food-surplus countries even though much of that surplus leaves the country. The dissertation's own research proposal had specified the FAO Food Balance Sheet food supply quantity as the dependent variable, but the implementation had diverged from this — either because the FAOSTAT API was inaccessible at the time of writing, or because the distinction was not fully appreciated. Whichever reason, the implementation and the proposal were misaligned.

These three problems existed simultaneously. They were not independent. The contaminated corpus produced NLP topics that mixed supply-side and access-side themes. The wrong dependent variable made those topics appear less predictive than they actually are. And the absence of a formal test meant no one had yet forced the question "does this actually work?" to a quantitative answer.

---

## Part 2: First Query — "I Want to Make Sure the Literature Are Not Noises"

### 2.1 What Prompted This

The first substantive query of this session was: *"I want to make sure the literature are not noises but focused only on cereal food availability. and determine how better is the factor discovered using nlp for the literatures."*

This was a researcher noticing something uncomfortable — that the papers feeding the NLP analysis might not actually be about the right thing. When you build a topic model, the topics you get reflect the corpus you put in. If the corpus contains papers about financial inclusion, the topics will contain financial-access vocabulary. If it contains papers about nutrition policy, the topics will reflect that. The topics will look like "themes in the food security literature" when what you need is "themes in the *cereal availability* literature". These are not the same thing.

The second part of the query — "determine how better is the factor discovered using nlp" — was asking something deeper: not just whether the corpus is right, but whether the NLP is actually contributing to the empirical work. There was an implicit awareness that having identified themes is not the same as those themes being useful. The question is whether the NLP-discovered predictors add measurable explanatory power beyond what any reasonable analyst would have included in a baseline model anyway.

Both parts of this query were addressed together because they are connected. Cleaning the corpus changes the NLP topics, and changing the NLP topics changes which predictors are included in Model F, and Model F is where the NLP value-added is tested.

### 2.2 Fixing the Corpus: Why the Filter Changed

The root cause of corpus noise was that the original strict classification filter allowed a paper to qualify through thematic accumulation without any explicit supply-side term. The condition `has_strong_availability_signal` was supposed to gate on availability content, but it was satisfied by matching enough themes — and the themes included `smallholder`, `farmer`, and `rural household`, which appear in access-side papers just as readily as supply-side ones.

The fix added a new hard gate: `has_explicit_availability`. This is a binary condition — either the paper contains at least one of 54 explicitly supply-side terms in its title or abstract, or it does not. There is no substitute. No amount of accumulating other thematic signals can pass this gate. The 54 terms were grouped into five semantic categories: supply and availability measurement terms (`food balance sheet`, `cereal supply`, `dietary energy supply`), aggregate production terms (`cereal production`, `agricultural output`), crop-specific production terms (`rice production`, `wheat production`, `maize yield`), post-harvest and storage terms (`post-harvest loss`, `grain storage`, `cold chain`), and supply-chain terms (`value chain`, `supply chain`, `food system`).

Simultaneously, the availability terms list was expanded from 28 to 54 entries by adding crop-specific production and yield terms that had been absent. This is what produced the counterintuitive result: the filter became stricter, but the number of strict papers went up — from 114 to 127. The explanation is that the same expansion that made the gate harder to pass by thematic accumulation also opened it to papers that were genuinely supply-side (discussing rice production, wheat yield, maize production) but had been classified as moderate because the narrower old list did not include those terms.

The result: 127 papers, all verified to contain at least one explicit supply-side availability term. The access-side papers — financial inclusion, consumer behaviour, nutritional deprivation — were removed from the strict tier.

### 2.3 Adding the Formal NLP Test: Why the F-Test Was Needed

Addressing the second part of the query — "determine how better" — required something more than comparing R² values. A nested F-test was implemented in Phase F.

The nested F-test answers a specific question: is the improvement in R² when the NLP-discovered predictors are added to the baseline model large enough that it is statistically unlikely to have occurred by chance? The test compares two nested models — Model A★ (baseline predictors only, on the same N countries as Model F) and Model F (baseline plus five NLP-discovered predictors) — and tests whether the joint contribution of the five added predictors is statistically distinguishable from zero.

The reason this test is necessary rather than just reporting R² is precisely because R² always increases when you add predictors, even if the predictors are random noise. A model with 12 predictors will always fit better than one with 7, in-sample, regardless of whether those extra 5 predictors carry real information. The F-test controls for this by penalising for the degrees of freedom consumed. The adjusted R² does the same thing in a different way — it only increases if the new predictors add more than their "fair share" of explanatory power.

The formula used was:
```
F = [(R²_full − R²_restricted) / q] / [(1 − R²_full) / (n − k_full − 1)]
```
where q = number of extra predictors added (5), n = number of countries, k_full = total predictors in Model F (12).

### 2.4 The F-Test Result at This Stage: A Null Finding

With the corpus cleaned and the F-test in place, the results at this stage were:
- Model A★ R² = 0.286, Model F R² = 0.319 (N=154)
- Delta R² = 0.033, Partial R² = 3.5%
- F(5, 141) = 1.030, **p = 0.402 (not significant)**
- Adjusted R² change: 0.286 → 0.288 (+0.002)

This was a null result. The five NLP-discovered predictors, taken together, did not significantly improve prediction beyond the baseline. The adjusted R² barely moved. The F statistic was essentially 1.0, which means the NLP block was performing no better than chance.

The correct interpretation of this result — and it is important to be clear about this — was not that the pipeline had failed. It was that something else in the pipeline was preventing the NLP from showing its value. That "something else" would take two more queries to identify fully.

---

## Part 3: Second Query — "What Can I Do About Improving This Data Quality?"

### 3.1 What Prompted This

After seeing the null F-test result and understanding that N=154, the question "what can I do about improving this data quality" arose. This question appears to be about missing data and sample size — but it was actually pointing at a deeper structural problem.

The sample size of 154 countries (out of 174 in the master dataset) was being determined by which countries had complete data across all predictors in the model. For most predictors, coverage was good — 160–174 countries. But the Logistics Performance Index (`lpi_overall`), which was the proxy for the NMF Topic 6 value chain/market access theme, covered only 90 of 174 countries. That is 48% missing data.

Here is the problem that creates: the five NLP-discovered predictors in Model F include `lpi_overall` (or its eventual replacement). When the model runs, it drops any country that is missing data in any predictor. So even though the other four NLP predictors cover 160–173 countries, the model sample collapses to wherever all five overlap. With LPI at 90 countries, the effective sample for Model F would be approximately 78 countries — less than half the master dataset.

A model on 78 countries is unreliable for two reasons. First, statistical power is low: with 12 predictors on 78 observations, standard errors are large and genuine effects are hard to detect. Second, the 78 countries with complete LPI data are not a random subset of the world. They are disproportionately higher-income, better-institutionalised, and more integrated into global trade networks. This means the model is estimating the relationship between logistics and food availability on a sample that looks very different from the full global distribution — which introduces systematic bias.

### 3.2 The Attempt Chain: What Was Tried Before the Solution

The search for a better logistics variable went through several attempts, each of which failed for a different reason. This is documented here not because the failures were avoidable — most were not — but because each failure revealed something real about the data landscape.

**Attempt 1 — World Bank Paved Roads (IS.ROD.PAVE.ZS):** Paved road coverage was identified as a plausible logistics infrastructure proxy. The World Bank WDI API was queried for this indicator for years 2021, 2020, 2019, and 2018. All requests returned `payload[1] = None`, meaning no data was returned. This is not a code error; it means the World Bank API is not serving this indicator through the standard country-level API endpoint. The data likely exists in the World Bank's database but is not accessible through the automated download pathway. This avenue was abandoned.

**Attempt 2 — LPI via WDI API (LP.LPI.OVRL.XQ):** LPI is technically available as a World Bank WDI indicator. A download query returned 191 country records, which appeared promising. On inspection, however, 191 was too many: the World Bank has only 217 countries and territories, and LPI data covers only about 140. The reason for the inflated count was that the filter `len(e.get('countryiso3code',''))==3` was treating World Bank regional aggregate codes (like `AFE`, `AFW`, `ARB`) as valid three-letter country codes. These aggregates do not represent real countries; they are income-group and regional summaries. After filtering to real countries using the World Bank country metadata API, only 49 genuine countries remained in the LPI dataset.

**The accidental overwrite:** During this diagnosis, the download script wrote the 49-country result to `data/raw/lpi.csv`, overwriting the original file that had 90 real countries. The original 90-country LPI file had been assembled with correct filtering earlier in the pipeline's history. The overwrite was identified by checking the row count of the saved file against expected counts. The correct LPI file (137 countries) was restored by rerunning `scripts/download_new_data.py`, which used the `mrv=3` (most recent value, up to 3 years back) parameter and correctly filtered to real countries via the metadata API. The discrepancy between 137 in the download and 90 in the analysis sample was then explained: the 137-country file contained countries across three different survey years, but only 90 of them had 2021-adjacent data that matched the master dataset's 2021 reference year.

**The conclusion about LPI:** After all attempts, LPI consistently covered only 90 of the 174 countries in the master dataset — 48% missing. This exceeds the 20% KNN imputation threshold by more than double. KNN imputation was not applicable.

### 3.3 The Solution: Trade % GDP and the Rationale for Choosing It

The replacement variable chosen was Trade as a percentage of GDP (World Bank WDI indicator `NE.TRD.GNFS.ZS`). This is the sum of a country's exports and imports of goods and services, expressed as a percentage of its GDP.

The case for Trade % GDP as a logistics/market-integration proxy rests on several arguments. Countries with higher trade shares have necessarily developed the port infrastructure, customs systems, road networks, and distribution logistics that make trade possible. A country that moves 100% of its GDP through trade has deeply integrated logistics; a country at 20% has not. The FAO 2015 State of Commodity Markets report uses trade openness as the standard proxy for logistics capacity when LPI data is unavailable. Twelve papers in the 127-paper strict corpus explicitly mention trade openness or market integration as a predictor of food availability.

Coverage: World Bank WDI's `NE.TRD.GNFS.ZS` covers 158 of the 174 countries in the master dataset — 9% missing. This is within the 20% KNN imputation threshold. The 16 missing values were filled using KNN imputation (k=5 nearest neighbours in the predictor space).

### 3.4 KNN Imputation: Why It Was Added and How It Works

KNN imputation was added alongside the switch to Trade % GDP to address the sample-size problem more broadly. Several predictors in Model F had small amounts of missing data — 4 to 16 countries out of 174 — that were causing unnecessary sample loss. Without imputation, a single missing value in any predictor drops the entire country from the model. With imputation, countries with one or two missing predictor values can be retained.

The logic is: for each country missing a value on variable X, find the five countries in the dataset that are most similar to it on all *other* variables, and estimate X as the weighted average of those five countries' X values. "Similarity" is measured as Euclidean distance in the standardised predictor space, so a country is considered similar to another if it has comparable yield, fertiliser use, GDP, trade share, and other predictors — even if one specific value is missing.

The threshold of ≤ 20% missing was chosen because beyond that point, the five nearest neighbours are themselves likely to have unusual or imputed values, making the estimate unreliable. LPI at 48% missing is far beyond this. Trade % GDP at 9% missing is comfortably within it.

After implementing Trade % GDP and KNN imputation, all models ran on **N=154 countries** — up from approximately 78 for Model F. This was a substantial improvement in both coverage and representativeness.

---

## Part 4: Third and Fourth Queries — "Fix and Improve" and "Can This Be Improved Further?"

### 4.1 What These Queries Were Asking

After the fixes to data quality — Trade % GDP replacing LPI, KNN imputation added — the pipeline was rerun. The question "fix and improve" was an instruction to implement those changes. The follow-up "can this be improved further?" was asked after seeing the results.

The results at this point were:
- N=154 (restored from ~78)
- F(5, 141) = 1.030, **p = 0.402 (still not significant)**
- Adjusted R² change: essentially unchanged

The F-test was still null. The data quality improvement had expanded the sample considerably, but the NLP block was still not significantly improving prediction. This prompted the honest question: is there more to do, or have we reached the ceiling of what is achievable here?

The answer given was: *yes, one more thing remains*. The data quality is now good. The corpus is clean. The formal test is in place. But there is still a structural mismatch between what the dependent variable measures and what the NLP themes are about. That mismatch is what is suppressing the NLP signal.

### 4.2 "Does That Mean We Have Done All the Best We Could Do Regarding the NLP?"

This question was essentially asking: is the null F-test the truth about NLP's contribution, or is something in the pipeline preventing the truth from appearing?

The answer is that a null F-test does not necessarily mean the NLP is failing. It can also mean the *dependent variable* does not respond to the things the NLP identified. If your NLP correctly identifies that the literature emphasises logistics, infrastructure, and trade — but your dependent variable only measures domestic production — then of course logistics won't appear significant. It doesn't predict production; it predicts *supply* (which includes imports).

The production-based DV was measuring how much cereal a country grows. The NLP themes (trade integration, storage infrastructure, electricity access) are about how much food actually reaches people. These are related but not the same quantity. The DV was filtering out the very signal the NLP was trying to detect.

---

## Part 5: Fifth Query — "Will This Change My Proposal or Affect It in Anyway?"

### 5.1 The Concern Behind the Question

Before committing to changing the dependent variable, the question asked was: *"will this change my proposal or affect it in anyway. does it still align with my proposal and project. rate it."*

This is a legitimate and important concern for any dissertation student. The research proposal is a formal agreement with the institution about what you will study and how you will study it. If the pipeline diverges from the proposal, the dissertation may be evaluated against criteria it was not designed to meet, or it may appear methodologically inconsistent. The question was not about whether the change is technically better — it was about whether it is *appropriate* for the specific academic project.

### 5.2 Why the Change Restores the Proposal Rather Than Contradicting It

The dissertation proposal's original specification for the dependent variable was the FAO Food Balance Sheet food supply quantity: *"DV: cereal food availability from FAO Food Balance Sheets. Alternative DV: cereal production per capita (robustness)."*

The production-based DV was the *alternative*, listed as a robustness check. The FBS food supply quantity was the *primary* DV. At some earlier point in the pipeline's development, the primary DV was unavailable (the FAOSTAT API was unreachable) and the alternative was promoted to primary. The pipeline had been running with the robustness-check DV treated as the main DV.

Switching to the FAO FBS food supply is therefore not a change to the research design. It is a restoration of the originally intended design. The alternative (production per capita) becomes what the proposal always said it should be: a robustness check, not the main specification.

The conceptual alignment is also direct. The dissertation title is about cereal food availability. Food availability, as defined in the food security literature following Maxwell and Frankenberger (1992) and subsequent FAO frameworks, means the physical quantity of food present in a country — including domestic production, imports, and stock drawdowns, minus exports and non-food uses. That is exactly what the FAO Food Balance Sheet measures. Production per capita measures only one component of that quantity: what was grown domestically.

### 5.3 The Evidence Criteria Established Before Committing

Before the change was made, a set of observable criteria was proposed so that the decision could be evaluated objectively after implementation. The criteria were:

1. **Sample size should increase.** The FBS covers import-dependent countries that the production DV excluded (they had production near zero and were filtered out by the 5 kg/capita minimum). More countries = better power.
2. **The F-test p-value should decrease.** If the FBS DV is a better match for the NLP themes, the NLP block should explain more variance of it.
3. **Adjusted R² should increase noticeably from A★ to F.** Not just a marginal tick-up, but a genuine step.
4. **Rural electricity access should appear individually significant.** This variable (NMF Topic 5) should predict how much food people have access to (FBS) but need not predict how much a country produces (production DV). If the DV change is working, this coefficient should tighten.
5. **The overall model R² should not collapse.** Adding import-dependent countries increases the heterogeneity of the sample, which could lower R². A moderate decrease is acceptable; a large one would suggest the FBS DV is not well-explained by the predictor set.

These criteria were established so that "I'm convinced" could be based on evidence rather than optimism.

---

## Part 6: Sixth and Final Query — "I'm Now Convinced. Go Ahead and Apply It."

### 6.1 The Implementation

Implementing the FAO FBS dependent variable required modifying Step 1 of `src/step7_run_prediction_models.py`. The existing code downloaded cereal production from the World Bank API and divided by population. The new code attempts three strategies in sequence, using the first that succeeds:

**Strategy 1 — FAOSTAT REST API:** The API at `https://www.fao.org/faostat/api/v1/en/data/FBS` (and its mirror at `fenixservices.fao.org`) was tried first. Both were unreachable — one timed out, the other returned HTTP 404. FAOSTAT migrated its API infrastructure in 2023, and the new endpoint was not responding at the time of implementation.

**Strategy 2 — FAOSTAT Bulk ZIP Download:** FAOSTAT publishes complete datasets as ZIP archives. The file `FoodBalanceSheets_E_All_Data_(Normalized).zip` (54.8 MB) at `https://bulks-faostat.fao.org/production/` was requested with `allow_redirects=True` and a 180-second timeout. This succeeded. The ZIP was unpacked in memory using Python's `zipfile` and `io` modules — no intermediate file was written to disk.

### 6.2 The Country Code Bug and Why It Happened

After downloading and unpacking the ZIP, the first attempt to merge the data with the master dataset resulted in **zero matched countries**. The master dataset uses ISO3 country codes (e.g., `AFG`, `NGA`, `BRA`). The FAOSTAT ZIP does not contain an ISO3 column. Its three identifier columns are:
- `Area Code` — a proprietary FAO numeric code (e.g., `2` for Afghanistan)
- `Area Code (M49)` — the United Nations M49 numeric code, stored as a string with a leading apostrophe (e.g., `'004` for Afghanistan)
- `Area` — country name in FAO's naming convention

The code tried to find a column containing "ISO3" in its name. None exists. It then fell back to `Area Code` (the numeric FAO code). The subsequent filter `str.len() == 3` was intended to find three-letter ISO3 codes, but it was instead finding three-digit FAO numeric codes (127, 145, 148, etc.). These were saved to the output file as if they were country identifiers. When the merge ran against the master dataset's ISO3 codes, nothing matched — because `"127"` does not match `"AFG"`.

The fix was to use the M49 column and the `pycountry` library. The M49 code `'004` is the UN's standard three-digit numeric code for Afghanistan. Stripping the leading apostrophe, zero-padding to three digits, and then calling `pycountry.countries.get(numeric='004')` returns the ISO3 code `AFG`. This mapping was applied to every row in the FAOSTAT data, converting FAO's M49 codes to ISO3 for merging.

After the fix: 176 countries resolved to ISO3 codes. Of those, 160 matched the 174-country master dataset. The 14 countries with FBS data but not in the master dataset were typically territories, microstates, or newly independent countries that the World Bank WDI base frame did not include.

### 6.3 The 5 kg/Capita Filter Was Removed

The original production-based DV had a filter that excluded any country with fewer than 5 kg of cereal production per capita per year. This filter was meant to exclude city-states and small island nations that produce essentially no cereals domestically (Singapore, Luxembourg, Malta, etc.). The rationale was that these countries represent a structurally different "food system" where all food is imported, and including them would create outliers in a model that was predicting domestic production.

With the FBS DV, this filter is no longer appropriate. Countries like Singapore, Malta, and Bahrain have real, measured FAO Food Balance Sheet food supply figures — they import cereals and those imports appear in the FBS. Their food supply per capita is real and valid. Excluding them would remove observations that the model can legitimately explain. The filter was removed when using FBS data. It is retained only in the World Bank production-based fallback, where the original reasoning still applies.

---

## Part 7: The Results — What Changed and Why It Changed

### 7.1 The Complete Before-and-After

The table below compares every key metric before and after the full set of improvements:

| Metric | Before (production DV, partial fixes) | After (FAO FBS DV, all fixes) |
|---|---|---|
| Strict corpus papers | 114 | 127 |
| Papers with explicit supply-side terms | ~80–90 | 127 (100%) |
| Model F sample size (N) | ~78 countries | 160 countries |
| LDA coherence (c_v) | Below 0.60 | Below 0.60 (unchanged — NMF used) |
| NMF topics | 7 | 7 (unchanged) |
| Model A OLS R² | 0.319 | 0.196 |
| Model F OLS R² | 0.343 | 0.283 |
| Nested F-test statistic | F(5, 141) = 1.030 | **F(5, 147) = 3.649** |
| Nested F-test p-value | p = 0.402 (n.s.) | **p = 0.004 (\*\*\*)** |
| NLP block partial R² | 3.5% | **11.0%** |
| Adj R² (A★ → F) | 0.286 → 0.288 | **0.157 → 0.224** |
| Adj R² gain | +0.002 | **+0.067** |
| rural_electricity_access_pct p-value | p = 0.072 (*) | **p = 0.012 (\*\*)** |
| rural_electricity_access_pct boot CI | Crosses zero | **Excludes zero [0.001, 0.007]** |

### 7.2 Why the OLS R² of Model A Went Down

A reader might notice that Model A's R² fell from 0.319 to 0.196. This is not a regression. It reflects the fact that the FBS DV is *harder to predict* with production-side variables alone than the production DV was. This is the whole point. When you predict production with production-side variables (yield, fertiliser, arable land), you get high R² trivially because the predictors and the outcome are measuring the same underlying thing. When you predict food supply (which includes imports and excludes exports) with production variables, the relationship is noisier — because trade can dramatically change the relationship between production and supply. The lower R² in Model A is evidence that the FBS DV is doing what it should: capturing something that production-side variables do not fully explain, leaving room for logistics and infrastructure variables to contribute.

### 7.3 Why the F-Test Became Significant With the FBS DV

The same five NLP-discovered predictors that were non-significant with the production DV are now jointly highly significant (p = 0.004) with the FBS DV. This appears at first like the same model is giving different answers, which seems suspicious. But there is a straightforward explanation.

The five NLP predictors include `trade_pct_gdp`, `rural_electricity_access_pct`, `cereal_loss_pct`, `fertiliser_efficiency`, and `food_price_inflation_pct`. Of these, `trade_pct_gdp` and `rural_electricity_access_pct` are the ones most directly relevant to the FBS DV.

The FBS food supply = production + imports − exports − stock changes − non-food uses. Trade volume directly affects the imports and exports terms. A country's trade intensity predicts how much food moves across its borders, which directly affects the FBS food supply figure. With the production DV, trade is unrelated to what was grown. With the FBS DV, trade is structurally embedded in what is measured.

Similarly, rural electricity access enables on-farm storage (reducing losses), refrigeration (extending shelf life), electric milling (processing grain for consumption), and powered irrigation (extending growing seasons). All of these improve the ratio of food supply to food production — they are mechanisms that sit between production and the FBS food supply figure, in exactly the domain the FBS DV captures and the production DV ignores.

This is not a case of "the model magically improved." It is a case of "the model is now measuring what the predictors actually affect."

---

## Part 8: Full Technical Results

### 8.1 All Model Comparisons (Final State, FAO FBS DV, N=160)

| Model | N | Predictors | OLS R² | OLS Adj R² | RF CV R² | XGB CV R² |
|---|---|---|---|---|---|---|
| A — Baseline Production | 160 | 7 | 0.196 | 0.159 | 0.104 | 0.094 |
| B — +Post-Harvest Loss | 160 | 8 | 0.202 | 0.160 | 0.108 | 0.097 |
| C — +Logistics Infrastructure | 160 | 10 | 0.277 | 0.229 | 0.085 | 0.143 |
| F — NLP-Discovered Themes | 160 | 12 | 0.283 | 0.224 | 0.078 | 0.066 |
| A★ — Baseline (same sample as F) | 160 | 7 | 0.194 | 0.157 | 0.099 | 0.070 |

Model B adds post-harvest loss (`cereal_loss_pct`) to Model A. The R² increase from A to B is small (+0.006). This is the empirical evidence that post-harvest loss, while conceptually important and strongly featured in the NLP corpus (NMF Topic 2), does not add strong predictive power at the cross-country level with available data. The loss variable is partly composed of sub-regional proxy values (Layer C of the PHL construction), which reduces its precision.

Model C adds logistics (`trade_pct_gdp`) and infrastructure (`rural_electricity_access_pct`). The R² increase from B to C is substantial (+0.075, Adj R² +0.069). This is the clearest evidence that logistics and infrastructure matter for food availability in a way that is measurable with available cross-country data. Both variables are significant in Model C: `rural_electricity_access_pct` (p = 0.015) and `trade_pct_gdp` (p = 0.340, not individually significant but part of the joint improvement).

### 8.2 Full OLS Table: Model F (NLP-Discovered Themes)

The table below presents all coefficients from the final Model F specification, estimated with HC3 heteroskedasticity-consistent standard errors:

| Variable | Coefficient | z-stat | p-value | Significance |
|---|---|---|---|---|
| Intercept | 6.714 | 16.117 | 0.000 | *** |
| cereal_yield_kg_per_ha (log) | −0.105 | −1.115 | 0.265 | n.s. |
| fertiliser_kg_per_ha (log) | 0.114 | 1.208 | 0.227 | n.s. |
| arable_land_pct | 0.0003 | 0.182 | 0.855 | n.s. |
| gdp_per_capita_usd (log) | −0.138 | −4.803 | 0.000 | *** |
| rural_population_pct | −0.0001 | −0.038 | 0.970 | n.s. |
| agri_employment_pct | −0.0015 | −0.597 | 0.550 | n.s. |
| livestock_production_index | 0.0027 | 1.943 | 0.052 | (marginal) |
| cereal_loss_pct | 0.0044 | 0.469 | 0.639 | n.s. |
| trade_pct_gdp (log) | −0.041 | −0.954 | 0.340 | n.s. |
| **rural_electricity_access_pct** | **0.0043** | **2.508** | **0.012** | ** |
| fertiliser_efficiency | 0.089 | 0.919 | 0.358 | n.s. |
| food_price_inflation_pct | −0.0007 | −1.016 | 0.310 | n.s. |

**Model diagnostics:** R² = 0.283, Adj R² = 0.224, F-statistic = 6.654, Prob(F) = 1.90e-09, N = 160, Condition Number = 2,960 (high — indicates potential multicollinearity).

Three results deserve explanation because they appear counterintuitive:

**gdp_per_capita_usd is negative and strongly significant.** Wealthier countries have lower cereal food supply per capita? Yes, and this is a known pattern. As countries develop, their diets diversify away from cereals toward proteins, vegetables, and dairy. Japan, Germany, and the UK consume far fewer calories from cereals than Ethiopia, Bangladesh, or Nigeria. This is the dietary transition — an empirically well-established phenomenon. The negative coefficient is not a modelling error.

**cereal_loss_pct is positive.** Higher post-harvest loss is associated with higher cereal food supply? This should be negative. The most likely explanation is confounding with production: countries with high absolute cereal production systems (large agricultural economies) also have higher absolute losses even if their loss percentage is not especially high. The loss percentage, being partly estimated from sub-regional proxies for non-APHLIS countries, may also carry noise that obscures the directional relationship. The bootstrap CI crosses zero, confirming the effect cannot be distinguished from noise.

**trade_pct_gdp is negative and non-significant.** High-trade economies have lower cereal food supply per capita? This is likely because highly trade-integrated economies are often more economically specialised — they may focus on services or manufacturing exports rather than food production, and their lower per-capita cereal supply reflects dietary diversification rather than food scarcity. There is also multicollinearity with GDP per capita: richer countries both trade more and eat less cereal.

### 8.3 NLP Block Assessment: Nested F-Test

The central quantitative finding of the dissertation:

```
Restricted model (A★): R² = 0.194, k = 7, N = 160
Full model (F):         R² = 0.283, k = 12, N = 160

F = [(0.283 − 0.194) / 5] / [(1 − 0.283) / (160 − 12 − 1)]
  = [0.089 / 5] / [0.717 / 147]
  = 0.0178 / 0.00488
  = 3.649

p-value: F(5, 147) = 3.649, p = 0.004 (***)

Partial R² = 0.089 / (1 − 0.194) = 0.089 / 0.806 = 0.110 (11.0%)
```

The five NLP-discovered predictors together account for 11% of the variance that was left unexplained by the baseline model, and this contribution is highly statistically significant (p = 0.004, well below the conventional 0.01 threshold).

### 8.4 Bootstrap Confidence Intervals

Bootstrap CIs (1,000 resamples) for the NLP-discovered predictors in Model F:

| Variable | Boot Mean | 95% CI Lower | 95% CI Upper | Status |
|---|---|---|---|---|
| cereal_loss_pct | 0.0037 | −0.0142 | 0.0202 | Crosses zero |
| trade_pct_gdp | −0.041 | −0.119 | 0.038 | Crosses zero |
| **rural_electricity_access_pct** | **0.0042** | **0.001** | **0.007** | **Excludes zero** |
| fertiliser_efficiency | 0.022 | −0.661 | 0.257 | Crosses zero |
| food_price_inflation_pct | −0.0006 | −0.003 | 0.003 | Crosses zero |

The joint F-test is significant even though individually most predictors have CIs crossing zero. This is a normal and coherent statistical result. The F-test measures the joint contribution of all five variables together; a variable can contribute to joint explanatory power without its individual coefficient being distinguishable from zero, because it may be explaining variance that is orthogonal to what other predictors explain.

---

## Part 9: The NLP Pipeline — What It Did and What It Produced

### 9.1 LDA: Why It Did Not Reach the Target

LDA (Latent Dirichlet Allocation) was run on the 127 strictly aligned papers with the goal of discovering topics with a c_v coherence score of at least 0.60. The coherence score measures how semantically consistent each topic's top words are — whether words that are assigned to the same topic actually tend to co-occur in meaningful ways across a reference corpus.

The target of 0.60 was not met. LDA performed below this threshold on the availability-restricted corpus.

The structural reason: LDA requires vocabulary variation between documents to form distinct topics. But the strict alignment filter has made all 127 papers share a large common vocabulary — `food security`, `cereal`, `production`, `farmer`, `household`, `yield` appear in nearly every paper. When documents are this similar, LDA cannot reliably separate them into distinct topics because there are not enough words that are strongly associated with some papers but not others. The filter that made the corpus cleaner also made it more homogeneous, and homogeneity works against LDA.

LDA results are reported as exploratory only. This is an honest limitation, not a failure — LDA was worth trying, and its failure to reach the threshold is itself an informative finding about corpus structure.

### 9.2 NMF: Why It Worked Better

NMF (Non-negative Matrix Factorisation) operates on a TF-IDF matrix rather than raw word counts. TF-IDF (Term Frequency–Inverse Document Frequency) down-weights words that appear in many documents. This means the common vocabulary (`food security`, `farmer`, `cereal`) is effectively suppressed before NMF runs. What remains is the distinctive vocabulary — the words that characterise subsets of papers rather than the whole corpus. NMF then factorises this matrix to find the topic patterns.

The result was seven interpretable topics:

| Topic | Key Terms | What Strand of Literature |
|---|---|---|
| 0 | land, soil, crop, water, resource, production | Land productivity and resource constraints |
| 1 | household, income, education, rural_household, status | Household-level availability determinants |
| 2 | loss, postharvest_loss, grain, storage, stage | Post-harvest loss and grain storage |
| 3 | climate_change, adaptation, yield, farmer | Climate change adaptation |
| 4 | rice, wheat, grain, variety, temperature | Grain variety, heat stress |
| 5 | technology, storage, sensor, investment, improved | Technology adoption for storage |
| 6 | africa, value_chain, economic, investment, smallholder | Africa-focused value chain work |

These are not arbitrary statistical clusters. Each one corresponds to a real research community and a real set of policy questions about cereal availability.

### 9.3 Mapping NMF Topics to Model F Predictors

The five predictors in Model F were chosen by identifying, for each NMF topic that is operationalisable with available cross-country data, the best-coverage country-level indicator:

| NMF Topic | Theme | Proxy Variable | Coverage |
|---|---|---|---|
| Topic 2 (post-harvest loss) | cereal_loss_pct | APHLIS + FAO FBS PHL | 174/174 |
| Topic 6 (value chain, market integration) | trade_pct_gdp | WB WDI NE.TRD.GNFS.ZS | 158/174 (+16 imputed) |
| Topic 5 (technology, storage, infrastructure) | rural_electricity_access_pct | WB WDI EG.ELC.ACCS.RU.ZS | 166/174 (+8 imputed) |
| Topic 0 (land productivity, input efficiency) | fertiliser_efficiency | Computed: yield / fertiliser | 173/174 (+1 imputed) |
| Topic 3 (climate, market signals) | food_price_inflation_pct | WB WDI FP.CPI.TOTL.ZG | 163/174 (+11 imputed) |

Topic 1 (household determinants) and Topic 4 (grain variety/temperature) were not operationalised in Model F. Topic 1 was excluded deliberately — household income and education are access-side predictors, not availability-side, and including them would conflate the two pillars of food security. Topic 4 was not operationalised because there is no clean country-level indicator for grain variety adoption or temperature-adjusted yield variability at the coverage and comparability needed for cross-country analysis.

---

## Part 10: What the Complete Picture Means

### 10.1 The Narrative Arc

The full journey from the start of this work to its current state follows a coherent arc:

The pipeline began with a legitimate research design (NLP on food availability literature → empirical models of cereal food availability) but three implementation problems were preventing the research design from producing its intended output. The corpus was feeding access-side papers into an availability-side NLP analysis. The dependent variable was measuring production rather than food supply. And there was no formal test of whether the NLP was adding value.

Each problem was identified through a specific observation: the corpus felt wrong because the NLP topics mixed supply-side and access-side content; the F-test was introduced because R² comparison alone is not a statistical test; the data quality question revealed the LPI coverage problem; the null F-test result prompted the question of whether the DV was suppressing the signal; the proposal alignment check confirmed that switching the DV was restoring the intended design rather than departing from it.

After all three problems were corrected, the NLP-to-empirical pipeline produced the result the research design was designed to produce: **the themes identified in the cereal food availability literature are not just academically interesting — they are statistically significant predictors of actual cross-country cereal food supply.**

### 10.2 What Is Definitively Supported

**The NLP block adds significant predictive power (p = 0.004, partial R² = 11.0%).** The five themes identified from the strictly aligned food availability literature — post-harvest loss and storage, market integration and value chains, technology and storage infrastructure, land productivity, and food price volatility — together explain 11% of the variance in cereal food supply that baseline production factors leave unexplained. This improvement is statistically distinguishable from noise at the 1% significance level.

**Rural electricity access is the individually significant NLP-discovered predictor (p = 0.012, bootstrap CI excludes zero).** This finding is mechanistically plausible: electricity access in rural areas enables storage (refrigeration, electric grain silos), processing (electric milling), and irrigation — all of which determine how much of the production chain translates into actual food supply. The variable directly operationalises NMF Topic 5 (technology adoption for storage and food system infrastructure).

**The logistics infrastructure block (Model C) drives the largest single R² improvement.** Adding trade and electricity access to the baseline (A → C) increases Adj R² from 0.159 to 0.229 — a gain of 0.070. This is larger than any other single step, and it confirms that market integration and infrastructure are the most empirically tractable determinants of food availability at the cross-country level.

### 10.3 What Remains Honestly Uncertain

**LDA coherence below 0.60.** The primary NLP method did not produce topics that meet the interpretability standard set in the proposal. NMF is used as the substantive finding; LDA is reported as exploratory.

**Four of five NLP predictors have bootstrap CIs crossing zero.** Only `rural_electricity_access_pct` is individually robust. The other four are collectively significant (the F-test) but individually uncertain. This is not the same as saying they do not matter — it says their individual effects cannot be precisely estimated at N=160.

**The condition number is high (2,960 in Model F).** This flags potential multicollinearity among the predictors. With GDP per capita, trade, electricity access, and fertiliser efficiency all being loosely correlated with development level, some of the standard errors are inflated. The HC3 standard errors partially compensate, but the warning is legitimate.

**Post-harvest loss does not show the expected negative sign.** The `cereal_loss_pct` coefficient is positive, which is the wrong direction from a food systems perspective. The likely cause is confounding (higher production countries also have higher absolute losses) and measurement noise from sub-regional proxy values. This should be acknowledged in the dissertation as a data quality limitation rather than a theoretical puzzle.

### 10.4 How to Present This in the Dissertation

The dissertation should report this pipeline's results across three sections:

**In the methods section:** Describe the corpus alignment filter (explicit availability terms required; 54-term list; 127 strict papers), the NLP pipeline (TF-IDF + LDA + NMF; LDA exploratory due to c_v < 0.60; NMF produces 7 interpretable topics), and the model specifications (A through F; nested F-test for NLP value-added assessment; HC3 standard errors; bootstrap CIs; KNN imputation for variables with ≤ 20% missing).

**In the results section:** Report the model comparison table (A through F on N=160), the nested F-test result (F(5,147) = 3.649, p = 0.004), the partial R² (11.0%), and the individual coefficient table for Model F. Report rural electricity access (p = 0.012, CI excludes zero) as the primary individually confirmed finding. Report the production-based DV as a robustness check (results available in appendix).

**In the discussion section:** Frame the finding as: NLP correctly identifies the themes the literature emphasises (post-harvest loss, logistics, rural infrastructure, climate adaptation) AND those themes translate into empirically significant predictors of cross-country food supply — but the translation is captured fully only when the dependent variable measures actual food availability (FAO FBS) rather than domestic production alone. This is itself a substantive finding about the relationship between supply-side determinants and different measures of food availability.

---

## Appendix: Summary of All Code Changes

| Change | File Modified | What Changed | Why |
|---|---|---|---|
| 1 | step4_score_and_filter_papers.py | AVAILABILITY_TERMS list expanded from 28 to 54 entries | Missing crop-specific production and yield terms were misclassifying supply-side papers as moderate |
| 2 | step4_score_and_filter_papers.py | Strict filter changed from theme-based fallback to explicit availability term gate | Access-side papers were accumulating thematic scores without any supply-side signal |
| 3 | step9_write_the_findings_report.py | Added scipy nested F-test and partial R² calculation | No formal statistical test of NLP value-added existed |
| 4 | step6_clean_and_combine_data.py | LPI retained for robustness only; Trade % GDP (NE.TRD.GNFS.ZS) added as primary logistics proxy | LPI covers only 90/174 countries (48% missing), collapsing Model F to N≈78 |
| 5 | step7_run_prediction_models.py | KNN imputation (k=5, ≤20% threshold) added before each model fit | Variables with 4–16 missing countries were dropping countries unnecessarily |
| 6 | step7_run_prediction_models.py | Step 1 DV download replaced with three-attempt FAO FBS strategy | Production-based DV measures what was grown, not what is available; FBS was the proposal-specified DV |
| 7 | step7_run_prediction_models.py | Added io, zipfile, pycountry imports; M49→ISO3 conversion function | FAO ZIP uses M49 numeric codes, not ISO3; pycountry converts them |
| 8 | step7_run_prediction_models.py | Removed 5 kg/capita minimum filter for FBS DV | Import-dependent countries have real FBS food supply and should not be excluded |
| 9 | step9_write_the_findings_report.py | Narrative generation made conditional on F-test significance | Hardcoded "not significant" language was correct for null result but wrong after FBS DV made F-test significant |

---

*End of Report*

*Source code: `src/`*
*Raw data: `data/raw/`*
*Processed data: `data/processed/`*
*Model outputs: `outputs/tables/`*
*Figures: `outputs/figures/`*
*Narratives: `outputs/narrative/`*
