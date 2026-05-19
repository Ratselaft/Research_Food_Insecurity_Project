# From Literature to Evidence: An NLP-Driven Predictive Analysis of Food Insecurity Factors, with a Focus on Post-Harvest Loss and Financial Access in Cross-Country Cereal Food Availability

**Student:** Odekunle Jibola Johnson
**Programme:** MSc Data Science and Artificial Intelligence
**Institution:** Sheffield Hallam University
**Supervisor:** Dr Richard S Wilson FHEA MBCS
**Submission Date:** May 2026
**Word Count:** 18,854 words (body text)

---

## Abstract

Food insecurity remains one of the defining challenges of the twenty-first century, yet the academic literature on its determinants is voluminous, fragmented, and rarely translated systematically into predictive empirical models. This dissertation bridges that gap by deploying a computational text-mining pipeline on a 127-paper corpus of peer-reviewed food security and cereal availability studies, extracting the dominant thematic clusters identified by the literature, and testing whether those themes translate into statistically significant predictive power in cross-country econometric models of cereal food availability, supplemented by Random Forest and XGBoost cross-validation as generalisation checks.

The dependent variable is national cereal food supply per capita (kg/person/year) derived from the FAO Food Balance Sheet (FAOSTAT Item 2905, Element 664), covering 160 countries in 2021. A sequential model-building strategy progresses from a seven-predictor production baseline (Model A, R² = 0.196) through post-harvest loss augmentation (Model B, R² = 0.202) and logistics and infrastructure augmentation (Model C, R² = 0.277) to a final twelve-predictor NLP-discovered themes model (Model F, R² = 0.283). The critical inferential test is a nested F-test comparing the NLP block (five additional predictors) against the same-sample baseline, yielding **F(5, 147) = 3.649, p = 0.004**, with a partial R² of 11.0%.

Natural language processing employs TF-IDF feature extraction and Non-Negative Matrix Factorisation (NMF, k = 7 topics) as the primary topic model, after Latent Dirichlet Allocation (LDA) produces coherence scores (c_v ≈ 0.38) below the pre-specified 0.60 threshold. Seven thematic clusters emerge: land, soil and water resources; household-level determinants; post-harvest loss and storage; climate change adaptation; grain variety and temperature; technology and storage adoption; and Africa value chain and investment.

Cross-validated out-of-sample R² values for Model F (RF: 0.078; XGB: 0.066) are substantially below the OLS in-sample figure, reflecting the generalisation ceiling imposed by a 160-country sample with 12 predictors; this gap is discussed alongside the OLS results in Chapter 4. The central finding is that rural electricity access — an infrastructure proxy operationalising the literature's storage and logistics themes — is the single NLP-discovered predictor with robust empirical support (coefficient = +0.004, p = 0.012, bootstrap 95% CI [+0.001, +0.007] excludes zero). Post-harvest loss, despite constituting the literature's most prominent availability-side theme (Topic 2 plus substantial overlap in Topic 6), does not translate to a statistically significant predictor at cross-country scale (p = 0.476–0.639 across model specifications). This divergence between literature emphasis and empirical signal is itself a substantive finding. Financial access variables, while theoretically relevant through demand-side channels, show no robust signal for the supply-side availability dependent variable used here, consistent with the four-pillar architecture of food security. Limitations include a cross-sectional design (association, not causation), moderate multicollinearity in the full NLP model (Condition Number ≈ 2,960 for Model F; 1,950 for the baseline Model A), proxy measurement error, and reduced explanatory power in the developing-country subsample (R² = 0.081). The dissertation concludes that NLP-guided variable discovery is a productive complement to theory-driven model specification, while emphasising that statistical significance at cross-country aggregation does not resolve micro-level causal pathways.

**Keywords:** food insecurity, cereal availability, NLP, topic modelling, NMF, post-harvest loss, rural electrification, cross-country regression, food balance sheet

---

# Chapter 1: Introduction

## 1.1 Background and Motivation

The global food security situation in the early 2020s presents a paradox. Agricultural output per capita has never been higher by historical standards, yet the Food and Agriculture Organization of the United Nations (FAO) and its partners estimate that between 691 and 783 million people faced hunger in 2022, with roughly 2.4 billion people experiencing moderate or severe food insecurity during that year (FAO, IFAD, UNICEF, WFP, & WHO, 2023). This contradiction reflects the multi-dimensional nature of food insecurity, which is not reducible to production alone but is shaped by availability, access, utilisation, and stability — the four pillars enshrined in the international consensus definition since the 1996 World Food Summit.

Cereal crops — wheat, rice, maize, barley, sorghum, millet, and their near-relatives — form the nutritional and caloric backbone of diets across every income stratum, but with particular significance for lower-income populations in South Asia, sub-Saharan Africa, and parts of Latin America and the Middle East. At the global level, cereals contribute approximately 50 percent of total human caloric intake directly and a further proportion indirectly via livestock feed (Godfray et al., 2010; Tilman et al., 2011). Cross-country differences in cereal food supply per capita are therefore a first-order indicator of differential exposure to food insecurity, making the modelling of those differences a problem of both academic and policy relevance.

The academic literature on the determinants of food security and cereal availability is vast. A naive keyword search on platforms such as OpenAlex or Scopus returns thousands of studies, spanning agronomic experiments, household surveys, macroeconomic analyses, climate projections, and value-chain mapping exercises. The wealth of this literature is simultaneously its weakness: without systematic synthesis, it is difficult for policymakers, research commissioners, or model-builders to know which factors the accumulated evidence most consistently identifies as decisive. Traditional narrative reviews and meta-analyses provide partial solutions but are labour-intensive, often domain-specific, and subject to expert selection bias. Computational methods — particularly natural language processing (NLP) — offer an alternative: systematic, replicable, and scalable literature synthesis that can surface thematic clusters without prior assumption about which themes will dominate.

This dissertation takes that computational route. It applies an NLP pipeline to a carefully screened corpus of 127 peer-reviewed papers on food security and cereal food availability, extracts the dominant thematic clusters using TF-IDF keyword weighting and Non-Negative Matrix Factorisation (NMF) topic modelling, translates each theme into one or more operationalisable empirical predictors, and tests whether those predictors demonstrate statistically significant explanatory power in cross-country regression models of cereal food supply per capita. Two factors receive particular attention: post-harvest loss (PHL), which emerges as a dominant availability-side theme in the NLP analysis and carries an established policy narrative of reducing waste along the supply chain, and financial access, which the literature connects to agricultural investment, smallholder productivity, and household purchasing power.

## 1.2 Research Questions and Objectives

The central research question guiding this dissertation is:

**"Which factors emphasised by the food insecurity literature (via NLP on a 127-paper corpus) demonstrate measurable predictive value in cross-country machine learning models of cereal food availability — and where do post-harvest loss and financial access fit?"**

From this central question, four specific objectives follow:

1. To construct a systematically filtered corpus of at least 100 peer-reviewed papers with dual alignment to food security and cereal availability, and to apply NLP techniques to identify the dominant thematic clusters in that corpus.
2. To operationalise the NLP-discovered themes as empirical predictors drawn from publicly available cross-country datasets, principally FAOSTAT and the World Bank's World Development Indicators.
3. To evaluate, through sequential model building and formal nested hypothesis testing, whether the NLP-guided predictors add statistically significant explanatory power to production-based baseline models of cereal food availability across 160 countries.
4. To assess specifically the empirical predictive contribution of post-harvest loss and financial access variables, situating null or weak findings within the theoretical literature on food security pillars and measurement challenges.

## 1.3 Scope and Delimitations

The empirical analysis is deliberately cross-sectional, using 2021 as the reference year for the dependent variable (FAO Food Balance Sheet, Item 2905, cereals excluding beer, Element 664, food supply quantity in grams per capita per day, converted to kilograms per person per year). The cross-sectional design enables broad country coverage (N = 160) at the cost of causal identification; the analysis explicitly claims association rather than causation. The study focuses on the availability pillar of food security — the quantity of cereal food supply reaching a country's population after accounting for production, trade, and stock changes — and does not model food access, utilisation, or stability directly. The NLP corpus is bounded to English-language, peer-reviewed publications identified through OpenAlex (n = 113) and Scopus (n = 11) with three further papers from curated PDFs, all published or available as of early 2026.

## 1.4 Significance

This dissertation makes three contributions. First, it demonstrates a reproducible NLP pipeline for translating a multi-source academic corpus into empirically testable hypotheses, a methodological contribution applicable beyond food security research. Second, it provides one of the first formal nested F-test comparisons between theory-driven and NLP-guided predictor blocks in cross-country food availability modelling. Third, it offers an academically honest account of where the literature's emphasis does and does not map onto measurable cross-country signals, which is itself informative for the research community.

## 1.5 Structure of the Dissertation

Chapter 2 reviews the theoretical and empirical literature across four thematic areas: food security frameworks, determinants of cereal availability, post-harvest loss, and the role of financial access and rural infrastructure in agricultural systems. Chapter 3 describes the full methodology, covering corpus construction, NLP pipeline, variable operationalisation, and econometric strategy. Chapter 4 presents results across all model specifications, including diagnostic tests and robustness checks. Chapter 5 discusses the findings in relation to the literature and research question. Chapter 6 concludes with implications, limitations, and directions for future research.

---

# Chapter 2: Literature Review

## 2.1 The Architecture of Food Security: Four Pillars and the Availability Primacy Debate

The concept of food security has undergone substantial theoretical evolution since it entered international discourse in the 1970s. The dominant early framing was supply-centric: food security meant adequate aggregate food production at national scale (Maxwell, 1996). The catastrophic famines of the 1970s and 1980s, studied with particular rigour by Amartya Sen, demonstrated that food availability at aggregate level was neither necessary nor sufficient for individual food security — a household could face starvation amid national surplus if it lacked the entitlements to access food (Sen, 1981). This entitlements insight shifted the analytical frame toward access, giving rise to the multi-pillar definition formalised at the 1996 World Food Summit and extended through subsequent FAO frameworks.

The four-pillar framework — availability, access, utilisation, and stability — structures the analysis in this dissertation. Availability refers to the physical existence of adequate quantities of food at national, subnational, or local level, determined by domestic production, imports, stocks, and food aid (Barrett, 2010). Access refers to households' economic and physical ability to obtain food. Utilisation encompasses biological use of food, encompassing diet diversity, food safety, and care practices. Stability denotes the temporal continuity of availability and access across seasons and economic shocks. As Coates (2013) observes, these pillars interact in complex, non-linear ways, and measuring any one pillar in isolation risks masking deficits in another. Webb et al. (2006) similarly argue that household-level food insecurity measurement must capture all four dimensions to be policy-actionable.

This dissertation's empirical focus on availability is a deliberate scope delimitation, not a claim that availability is more important than other pillars. The FAO Food Balance Sheet (FBS) dependent variable captures the net per-capita food supply reaching a country's population after accounting for all supply and disposal channels; it is a well-validated aggregate measure of availability (Headey & Ecker, 2013). The delimitation reflects the fact that the NLP corpus used here was screened for availability-side alignment — production, trade, storage, loss, logistics, and value chain — making availability the natural empirical terrain for this analysis.

## 2.2 Determinants of Cereal Food Availability: Theory and Evidence

### 2.2.1 Production-Side Determinants

Agricultural productivity literature identifies a set of canonical supply-side determinants: land area under cultivation, input intensity (fertiliser, irrigation, improved seed), labour, and agro-climatic conditions. Mueller et al. (2012) document that closing yield gaps in major cereals could increase production by 45–70 percent without expanding agricultural land, pointing to fertiliser application, water management, and varietal improvement as the key leverage points. Tilman et al. (2011) project that global food demand will increase by approximately 70–100 percent by 2050 relative to 2005, driven by population growth, income growth, and dietary transition toward animal products, placing sustained intensification pressure on cereal systems.

At cross-country level, however, the relationship between production-side variables and food availability per capita is complicated by structural factors. Countries with high agricultural employment often feature low productivity agriculture, and productivity gains do not automatically translate into availability at household level when trade, storage, and distribution systems are weak (Misselhorn, 2005). The present study's finding that cereal yield per hectare is not a significant predictor in any model specification is consistent with this literature: at cross-country scale in 2021, yield variation is a poor proxy for per-capita supply because high-yield countries may export surplus while importing calories through processed foods, and low-yield countries may sustain availability through food imports.

### 2.2.2 Trade and Market Integration

Trade is, theoretically, one of the most powerful mechanisms for stabilising food availability across countries: countries with production deficits can import, and seasonal fluctuations in domestic production can be buffered by market integration (Gröschl & Steinwachs, 2017). At the same time, trade dependence creates vulnerability to international price shocks, as the 2007–2008 and 2010–2011 food price crises demonstrated. The empirical proxy used in this study — trade as a percentage of GDP — is a commonly used but coarse measure of market integration that conflates export-led growth with import-dependent food access. The coefficient on this variable is negative (−0.041 in Model F) and non-significant (p = 0.340, 95% CI [−0.119, +0.038]). The negative direction likely reflects a selection effect: countries with the highest trade-to-GDP ratios in this sample include import-dependent small island states and Gulf oil exporters that have high trade exposure precisely because their domestic cereal production is structurally insufficient — meaning trade dependence here is a symptom of supply deficit rather than a mechanism for overcoming it. This is consistent with the literature's recognition that trade volume alone does not determine food availability; the composition, direction, and terms of trade, alongside domestic distribution systems, are equally important (Headey & Ecker, 2013).

### 2.2.3 Infrastructure, Logistics, and Rural Electricity

Infrastructure has emerged in the development economics literature as a critical enabling factor for agricultural market integration, storage, and processing. Roads reduce transport costs; electricity enables cold storage, processing, and information access; communication infrastructure facilitates market price transparency. The relationship between rural electrification and agricultural productivity and welfare has been studied extensively since Dinkelman (2011) demonstrated positive effects of rural electrification on women's employment in South Africa using a quasi-experimental design. Rural electricity access is, in a cross-country context, also a strong correlate of cold chain infrastructure, grain storage technology, and agro-processing capacity — all of which are identified in the NLP corpus as drivers of reduced post-harvest loss and improved food availability.

The World Bank's World Development Indicators (World Bank, 2021) provide consistent cross-country data on rural electricity access as a share of the rural population. FAO (2015) documents that rural infrastructure investment is among the most effective policy levers for reducing food insecurity in lower-income countries, particularly when targeted at storage, roads, and energy access simultaneously. The present study's finding that rural electricity access is the single NLP-discovered predictor with statistically robust support (p = 0.012, bootstrap CI excludes zero) is therefore consistent with an extensive theoretical and empirical base.

## 2.3 Post-Harvest Loss: The Literature's Prominent Gap

### 2.3.1 The Scale and Distribution of Post-Harvest Loss

The post-harvest loss literature represents perhaps the most direct connection between the food security and food waste bodies of research. FAO (2019) estimates that approximately 14 percent of global food production is lost between harvest and retail — before food even reaches the consumer stage — with cereals accounting for a substantial share given their dominance in global agricultural output. HLPE (2014) provides the foundational framework distinguishing loss (unintended reductions in food quantity or quality along the supply chain) from waste (deliberate or social discarding at the consumption stage), emphasising that loss and waste require distinct interventions.

The regional distribution of post-harvest loss is highly uneven. Affognon et al. (2015) conduct a systematic meta-analysis of PHL studies in sub-Saharan Africa and find that cereal losses of 10–20 percent are common at on-farm storage, with important variation by crop type, storage technology, and geography. Sheahan and Barrett (2017) extend this analysis, finding that while absolute PHL quantities are highest in South and Southeast Asia, loss rates (as a share of production) are highest in sub-Saharan Africa — a distinction with major policy implications for where intervention investment should be directed. Lipinski et al. (2013) argue that reducing post-harvest loss by even modest percentages could free significant quantities of food equivalent to the caloric needs of hundreds of millions of people, making loss reduction a compelling policy instrument for food security.

### 2.3.2 Why Cross-Country PHL Coefficients May Be Null

Despite this compelling narrative, the present study finds that the cereal_loss_pct variable is consistently non-significant across all model specifications (p = 0.476 in Model B, p = 0.639 in Model F, bootstrap 95% CI [−0.014, +0.020] crosses zero). This null finding at cross-country level deserves careful interpretation rather than dismissal, and the literature provides several explanations.

First, PHL data quality is a major concern at cross-country scale. FAO loss percentage estimates, which proxy the cross-country variation in this study, are themselves derived from heterogeneous sources: some are derived from detailed farm surveys, others from expert elicitation, others from extrapolation from neighbouring countries. Outside sub-Saharan Africa and parts of South Asia, PHL data are sparse and of uncertain reliability (Sheahan & Barrett, 2017). Measurement error in the independent variable attenuates the OLS coefficient toward zero, which is precisely what is observed here.

Second, the FAO Food Balance Sheet dependent variable already incorporates a loss adjustment in its supply accounting: it represents food supply after deducting losses estimated by FAO during processing and storage stages. If the DV already embeds part of the loss signal, the independent variable captures incremental variation beyond what the FBS adjusts for, which may be small and noisy at cross-country level.

Third, post-harvest loss operates through complex mediating pathways that are difficult to capture with a single cross-sectional coefficient. The causal mechanism runs: poor storage infrastructure → higher losses → lower net supply → lower food availability. But the confounding influence of income, technology adoption, and market access means that countries with low losses may have them for different reasons (good infrastructure versus low production to begin with), and the aggregation conceals this heterogeneity. This is precisely the problem Misselhorn (2005) identifies in multi-country food security analyses: aggregate cross-country regressions systematically understate within-country and pathway-level variation.

### 2.3.3 NMF Topic Evidence on PHL's Literature Prominence

In this study's NLP analysis, post-harvest loss constitutes a coherent and prominent theme in the corpus. NMF Topic 2 — labelled "Post-Harvest Loss and Storage" — has dominant keywords: loss, postharvest_loss, fruit_vegetable, postharvest, grain, appropriate, storage, stage, with 15 dominant papers. NMF Topic 5 — "Technology and Storage Adoption" — also prominently features storage and PHL-adjacent technology terms (technology, adoption, farmer, storage, investment, improved, phl, sensor) with 15 dominant papers. Together with Topic 6's value-chain orientation (africa, economic, production, system, waste, smallholder, value_chain, investment, 27 dominant papers), post-harvest loss accounts for the most prominent cluster of availability-side thematic content in the corpus. The divergence between literature emphasis (prominent) and empirical signal (null) is thus one of the dissertation's most substantive findings.

## 2.4 Financial Access and Agricultural Food Security

### 2.4.1 Theoretical Pathways

Financial access operates on food security through multiple, often indirect pathways. At the household level, access to credit enables smallholder farmers to purchase inputs (seed, fertiliser, pesticide) at planting time rather than being constrained to subsistence levels by liquidity; formal savings mechanisms enable risk management against harvest failures; insurance products — when available — reduce precautionary behaviour that limits input investment (Demirgüç-Kunt et al., 2022). At the market level, financial sector development enables agricultural commodity trading, warehouse receipt financing, and value-chain lending that can reduce PHL by financing storage infrastructure improvements (Zins & Weill, 2016).

Aker and Mbiti (2010) document how mobile money in sub-Saharan Africa has reduced transaction costs in agricultural markets, enabled faster price discovery, and provided smallholders with access to financial services at lower cost than traditional bank branches. These mechanisms link financial access to agricultural productivity and market integration, creating a plausible pathway from financial inclusion to food availability. Bateman (2010) provides a contrarian perspective, arguing that micro-finance in particular has often failed to deliver transformative agricultural or income effects, with loan capital frequently diverted to consumption rather than productive investment.

### 2.4.2 Why Financial Access May Not Predict Cross-Country Availability

The theoretical channels described above operate primarily on the access and stability pillars of food security — household ability to afford food and resilience to economic shocks — rather than on the aggregate supply-side availability the present study models. At cross-country level, financial access (proxied by indicators such as bank account ownership from the Global Findex, or domestic credit to private sector as a percentage of GDP) may be highly correlated with overall income levels (GDP per capita), which the model already controls for. In the model specifications tested during this study's iterative development, financial access variables consistently showed no robust signal once GDP per capita was included, consistent with theory and with similar null findings in the cross-country food security literature (Headey & Ecker, 2013). The results section reports this explicitly and the finding is discussed in Chapter 5.

## 2.4.3 The Financial Access–PHL Intersection

An underexplored intersection in the literature concerns whether financial access might indirectly reduce post-harvest loss by enabling investment in better storage technology. The World Resources Institute's analysis (Lipinski et al., 2013) highlights that post-harvest loss reduction interventions — hermetic storage bags, metal silos, community cold rooms — require capital investment that many smallholder farmers cannot self-finance. In theory, access to affordable agricultural credit should enable farmers to invest in loss-reducing technologies, creating a causal chain from financial access to storage investment to lower PHL to higher food availability. Zins and Weill (2016) document that formal bank account ownership in sub-Saharan Africa is positively associated with access to agricultural loans, providing partial support for the credit-access premise. However, the empirical evidence on whether this chain actually reduces PHL at measurable scale remains thin. The few studies that examine the financial access–PHL relationship at farm level find mixed evidence: formal credit sometimes funds storage improvements, but also funds consumption smoothing, school fees, and other household priorities that compete with agricultural investment (Bateman, 2010). At cross-country aggregate level, these micro-level heterogeneities further dilute any detectable signal, which is consistent with the null financial access findings in this study.

## 2.5 NLP and Text Mining as Research Synthesis Tools

### 2.5.1 From Manual to Computational Review

Systematic literature reviews have traditionally relied on manual screening, coding, and synthesis. The labour intensity of this approach limits the volume of literature that any research team can process and introduces reviewer subjectivity in theme identification. Computational methods — text mining, topic modelling, bibliometric analysis — offer a complementary approach that trades interpretive depth for scale and replicability (Thomas & Harden, 2008). In the context of food security research, where thousands of studies span multiple disciplines and methodological traditions, computational synthesis tools are particularly valuable.

The development of large-scale open academic graph platforms — most relevantly OpenAlex, which provides open API access to metadata and abstracts for over 250 million scholarly works — has dramatically lowered the barrier to computational literature synthesis. Where a decade ago large-scale text mining required either commercial database access or labour-intensive web scraping, researchers can now retrieve thousands of abstracts programmatically through open APIs, enabling the kind of corpus construction underpinning this dissertation. Scopus, while commercial, provides complementary coverage for journals not indexed in OpenAlex, and the combination of both platforms with manual PDF retrieval maximises corpus recall for specialised research domains.

### 2.5.2 Topic Modelling: LDA and NMF

Latent Dirichlet Allocation (LDA), introduced by Blei et al. (2003), is the most widely used probabilistic topic model in social science text mining. LDA treats each document as a mixture of topics and each topic as a distribution over words, with inference conducted via variational Bayes or Gibbs sampling. A well-fitted LDA model produces coherent, interpretable topics that reflect genuine semantic clustering in the corpus. Coherence scores, particularly the c_v metric which correlates best with human judgment, provide a model-fit heuristic: values above 0.60 are generally considered indicative of good topic quality (Röder, Both, & Hinneburg, 2015). In the present study, LDA produces a best c_v coherence of approximately 0.38, below the pre-specified 0.60 threshold. This likely reflects the relatively small corpus (N = 127) and the specialised, technical vocabulary of agricultural and food security research, which creates sparse document-term matrices that challenge probabilistic inference.

LDA's statistical foundations require that the document collection is large enough to allow reliable estimation of both topic-word and document-topic distributions. The original Blei et al. (2003) implementation was designed for thousands of documents; with only 127 documents, the joint probability space is underidentified, resulting in topic instability across random seeds and the low coherence scores observed here. This is a practical limitation that should be anticipated in any application of LDA to small, specialised academic corpora — and the solution, as adopted in this study, is to use NMF as the primary decomposition method.

Non-Negative Matrix Factorisation (NMF), developed by Lee and Seung (1999), provides an algebraically simpler alternative: it decomposes the TF-IDF document-term matrix into two non-negative matrices representing topic-word and document-topic distributions, with the non-negativity constraint producing parts-based representations that are often more interpretable than LDA outputs in sparse technical corpora. The non-negativity constraint is key: unlike LDA or singular value decomposition (SVD), NMF produces topic components that add up to represent the original document, so that each topic component is interpretable as a positive contribution to the document's meaning rather than a positive and negative deviation from a baseline. Wang et al. (2020) demonstrate the utility of NMF for topic extraction from scientific text in their CORD-19 COVID-19 research dataset analysis, and Mikolov et al. (2013) show that distributional semantic representations underlie the success of both vector-space and matrix-factorisation approaches to text representation. In the present study, NMF (k = 7 topics) is used as the primary NLP result, producing seven coherent availability-themed clusters that structure the empirical analysis.

## 2.6 Cross-Country Modelling: Methods and Precedents

Cross-country regression analysis of food security determinants has a well-established tradition. Misselhorn (2005) analyses food insecurity in southern Africa using a comparative framework that identifies climate variability, HIV/AIDS, governance failures, and poverty as the dominant drivers — a multi-causal picture that resists reduction to single-variable explanations. Headey and Ecker (2013) critique common measurement approaches in food security cross-country analyses, arguing for greater attention to the choice of dependent variable and the theoretical alignment between the DV and the explanatory framework. Their critique directly motivates the present study's use of the FAO FBS-derived cereal availability measure as the DV, which is theoretically aligned with availability-side predictors.

The choice of dependent variable in cross-country food security modelling is consequential and often underappreciated. Common alternatives include dietary energy supply (DES) per capita from the FAO, prevalence of undernourishment (PoU), household food consumption survey indicators, and food access indices. Each captures a different dimension of the food security problem. DES is conceptually similar to the cereal availability measure used here but covers all food items rather than cereals specifically. PoU incorporates both supply and distribution, making it a more complex function of the same supply-side variables and therefore harder to interpret in supply-side models. Household food consumption data provide household-level precision but are not available for a consistent cross-country panel. The FAO FBS cereal availability measure used here is preferred for this study because it is specifically aligned with the availability pillar, is available for the broadest cross-country sample (160 countries), and can be linked directly to the NMF-identified themes which are cereal and supply-chain focused.

Yadav et al. (2023) demonstrate the application of machine learning methods — including Random Forests and Gradient Boosting — to food security prediction problems, finding that ensemble methods can capture non-linear relationships missed by OLS but require careful regularisation and cross-validation in small-N settings. Their analysis underscores that the interpretive advantage of OLS — transparent coefficient estimates with associated p-values — makes it the natural primary method for a study whose goal is to identify which predictors are significant and in what direction, with machine learning providing a supplementary generalisation check. Muckenhuber et al. (2020) apply Random Forest models in a food security context, emphasising the importance of cross-validation for honest generalisation assessment. The present study uses both OLS (primary inference) and RF/XGB cross-validated models (supplementary generalisation check), consistent with best practice in food security machine learning.

The governance literature, represented by Kaufmann et al. (2010)'s Worldwide Governance Indicators, provides a further set of cross-country controls. Governance quality is associated with better agricultural policy implementation, reduced corruption in food distribution, and more effective post-harvest infrastructure investment — all plausible mediators between structural conditions and food availability outcomes. The present study tests WGI political stability as a Robustness Specification 6, finding a significant coefficient (p = 0.032), which confirms that governance is a relevant control variable but one that is modelled separately to avoid overspecification in the primary models.

Gröschl and Steinwachs (2017) contribute to the cross-country food security literature through their analysis of how natural disasters affect international trade patterns, which is methodologically relevant because natural disasters — particularly droughts and floods — constitute one of the major external shocks to cereal availability in lower-income countries. Their finding that disaster impacts are transmitted through trade provides indirect support for including trade openness in supply-side food availability models, even if the cross-sectional trade coefficient in this study is non-significant. The cross-sectional design captures long-run structural trade integration, whereas disaster impacts are episodic and require panel data to capture.

### 2.6.1 Heteroscedasticity in Cross-Country Models

A well-documented challenge in cross-country regression is heteroscedasticity: the variance of the error term systematically varies across countries, typically being larger for poorer, more volatile economies with noisier data. If unaddressed, heteroscedasticity causes OLS standard errors to be inconsistent, producing over- or under-stated t-statistics. The standard solution, adopted throughout this study, is to use heteroscedasticity-robust (HC3) standard errors. The HC3 estimator, introduced by MacKinnon and White (1985), uses a leverage-adjusted squared residual formula that performs well in small samples relative to HC0 and HC1 alternatives. The systematic use of HC3 throughout this study means that significance tests are based on correct (asymptotically valid) standard errors even if the error variance is non-constant across the 160-country sample.

## 2.7 Summary of the Literature's Empirical Expectations

Drawing together the literature reviewed above, Table 2.1 summarises the expected direction and robustness of the key predictors used in this study, against which the empirical results in Chapter 4 are evaluated.

**Table 2.1: Expected vs. Empirical Predictor Directions Based on Literature**

| Predictor | Literature-Expected Direction | Theoretical Rationale | Prior Empirical Evidence |
|---|---|---|---|
| GDP per capita | Negative (log scale) | Wealthier countries shift to diverse diets with less cereal dependence | Headey & Ecker (2013) |
| Agricultural employment % | Ambiguous | High agricultural employment correlates with subsistence agriculture | Misselhorn (2005) |
| Fertiliser kg/ha | Positive | Input intensity → yield → supply | Mueller et al. (2012) |
| Cereal yield kg/ha | Positive | Direct production efficiency | Tilman et al. (2011) |
| Rural electricity access % | Positive | Storage, cold chain, processing infrastructure | Dinkelman (2011) |
| Cereal loss % | Negative | Higher loss → lower net availability | Affognon et al. (2015) |
| Trade % GDP | Ambiguous | Market integration vs. vulnerability | Gröschl & Steinwachs (2017) |
| Financial access | Positive (indirect) | Credit enables input purchase, storage investment | Demirgüç-Kunt et al. (2022) |

This literature review establishes the theoretical expectations against which the empirical findings of Chapters 3 and 4 are evaluated. The next chapter describes in detail the methods used to construct the NLP pipeline, operationalise the predictors, and estimate the models.

---

# Chapter 3: Methodology

## 3.1 Research Philosophy and Design

This dissertation adopts a broadly positivist epistemological stance, treating food security outcomes as objective phenomena measurable through observable indicators and amenable to quantitative analysis. The research design is mixed-method deductive-inductive in the following sense: the NLP phase is inductive — it allows the corpus to surface thematic clusters without pre-specifying the themes — while the subsequent econometric phase is deductive, testing specific hypotheses about whether NLP-identified themes predict the dependent variable. This combination is appropriate for a literature-to-evidence research design in which the structure of the literature is not known in advance but the empirical testing follows standard hypothesis-driven procedures.

The design is cross-sectional, using 2021 as the reference year for all primary variables. A cross-sectional design is appropriate for the research question — which factors differentiate countries in their cereal food availability — but forecloses causal inference; associations are documented but causal direction is not identifiable from the data alone. Panel data would be needed to control for time-invariant country characteristics and exploit within-country variation over time, which would strengthen causal claims but is beyond the scope and data availability of this study.

## 3.2 Corpus Construction and Screening

### 3.2.1 Search Strategy

The NLP corpus is drawn from three sources: OpenAlex (an open academic graph), Scopus (a commercial bibliographic database), and manually identified PDFs. Search queries combined food security terms (food security, food insecurity, food availability, hunger, undernourishment) with cereal-system terms (cereal, grain, maize, rice, wheat, post-harvest loss, storage, logistics, value chain, crop yield, agricultural production). Searches were conducted in early 2026 and covered publications from 2000 onward, with no strict upper bound, to capture the full body of relevant evidence available to the research community.

### 3.2.2 Inclusion and Exclusion Criteria

For a paper to be included in the strictly aligned corpus (used for NLP), it had to satisfy two conditions simultaneously: (1) a food security core — the paper's primary focus must be food security, food insecurity, hunger, or undernourishment; and (2) an availability-side signal — the paper must contain explicit discussion of food availability, cereal production, crop yield, post-harvest loss, storage, cold chain, food supply, logistics, or value-chain movement. Papers were excluded if they focused exclusively on nutrition, dietary quality, or sanitation without connecting to supply-side availability, and if they were grey literature, conference abstracts, or review articles that could not be obtained in full text. This strict dual-alignment screening yielded a final corpus of 127 papers: 113 from OpenAlex, 11 from Scopus, and 3 from curated PDFs.

### 3.2.3 Corpus Validation

Of the 127 papers, all 127 contain explicit availability terms by construction of the screening criteria, and 94 of 127 (74 percent) contain at least one availability driver theme as identified by the TF-IDF pipeline. This alignment rate provides confidence that the corpus represents the literature on cereal food availability rather than the broader food security literature, which would have introduced noise into the NLP topic extraction.

## 3.3 NLP Pipeline

### 3.3.1 Text Pre-Processing

Text was pre-processed in a standard pipeline: lowercasing; removal of punctuation, numbers, and stop words using a combined English and domain-specific stop-word list (removing common academic terms such as "study," "result," "paper," "show" that carry no thematic information); tokenisation; and bigram and trigram collocation detection using pointwise mutual information to preserve compound terms such as "post-harvest loss," "climate change," "food security," and "value chain." Compound terms were then treated as single tokens (e.g., postharvest_loss, climate_change, rural_household, value_chain) in all downstream analyses.

### 3.3.2 TF-IDF Vectorisation

The pre-processed corpus was represented as a document-term matrix using Term Frequency–Inverse Document Frequency (TF-IDF) weighting. TF-IDF upweights terms that appear frequently within a given document but rarely across the corpus, producing a representation that captures each document's distinctive vocabulary rather than just its most common words. This is appropriate for a technical corpus where terms like "food" and "agriculture" appear in virtually every document and would otherwise dominate keyword-based analyses. The resulting matrix has dimensions 127 (documents) × V (vocabulary), where V is determined by minimum document frequency thresholds (minimum two document appearances, maximum 95 percent document frequency).

### 3.3.3 LDA Topic Modelling (Exploratory)

Latent Dirichlet Allocation was applied across a sweep of topic counts from K = 3 to K = 12, with c_v coherence computed at each K using a sliding window approach. The best coherence achieved was approximately 0.38 at K = 8, which is below the pre-specified 0.60 threshold specified in the research proposal. This result is reported as exploratory and is not used as the primary NLP output. The likely explanation for the low coherence is the small corpus size (N = 127), which produces a sparse document-term matrix insufficient for reliable LDA inference. The LDA results are presented in Appendix A alongside the coherence curve figure.

### 3.3.4 NMF Topic Modelling (Primary)

Non-Negative Matrix Factorisation was applied as the primary topic model. Unlike LDA, NMF does not have a single published coherence-based model selection criterion, so topic number selection relied on a combination of two approaches. First, a formula heuristic (k = min(8, max(3, N // 15)), where N = 127 documents) suggested k = 8 as a starting point. Second, topic solutions at k = 5, 7, and 9 were each inspected by the researcher for interpretability and stability, with k = 7 selected because it produced the clearest thematic separation without generating duplicate or incoherent topics. This selection involves researcher judgement and is therefore a subjective element of the NLP pipeline; a sensitivity analysis presented in Appendix A (Table A.2) shows that the core themes — post-harvest loss, climate change adaptation, land and water, and value chain — are stable across k = 5 and k = 9, supporting confidence in the primary k = 7 solution. NMF factorises the TF-IDF matrix W into two non-negative matrices H (topics × vocabulary) and D (documents × topics), with reconstruction loss minimised via alternating least squares. The seven resulting topics were labelled by inspection of the top eight keywords per topic. Document assignment was made by the dominant topic (highest component score). Dominant paper counts per topic range from 7 to 27, reflecting natural variation in how much of the corpus addresses each theme.

The seven NMF topics, their keywords, and dominant paper counts are:

**Table 3.1: NMF Topic Summary**

| Topic | Label | Top Keywords | Dominant Papers |
|---|---|---|---|
| Topic 0 | Land, Soil and Water Resources | land, environmental, soil, crop, resource, production, water, demand | 17 |
| Topic 1 | Household-Level Determinants | household, determinant, income, education, rural_household, status, availability, farm_household | 17 |
| Topic 2 | Post-Harvest Loss and Storage | loss, postharvest_loss, fruit_vegetable, postharvest, grain, appropriate, storage, stage | 15 |
| Topic 3 | Climate Change Adaptation | climate_change, adaptation_strategy, adaptation, impact, change, climate, farmer, yield | 16 |
| Topic 4 | Grain Variety and Temperature | rice, yield, variety, silo, wheat, grain, crop, temperature | 7 |
| Topic 5 | Technology and Storage Adoption | technology, adoption, farmer, storage, investment, improved, phl, sensor | 15 |
| Topic 6 | Africa Value Chain and Investment | africa, economic, production, system, waste, smallholder, value_chain, investment | 27 |

### 3.3.5 Corpus-Wide TF-IDF Keywords

Corpus-wide TF-IDF term importance (averaged across all documents) identifies the following 15 most salient terms: household, loss, climate_change, production, crop, farmer, impact, agriculture, technology, agricultural, rice, yield, availability, africa, system. The prominence of "loss" and "household" alongside production terms confirms that the corpus's centre of gravity is supply-chain loss and household-level food access, consistent with the NMF topic structure.

## 3.4 Variable Operationalisation

### 3.4.1 Dependent Variable

The dependent variable is cereal food availability per capita (cereal_availability_kg_pc), derived from the FAO Food Balance Sheet (FAOSTAT, Item 2905, Cereals – Excluding Beer, Element 664, Food supply quantity g/capita/day). The raw value in grams per capita per day was converted to kilograms per person per year using the formula: kg/year = g/day × 365 / 1000. The resulting variable covers 160 countries in 2021, with a range of 119.3 to 714.4 kg/person/year, reflecting the substantial global diversity in cereal-based diets from protein-rich diversified diets in high-income countries (lower cereal reliance) to cereal-dominated diets in the lowest-income countries. The DV was log-transformed prior to regression due to positive skew, consistent with standard practice for per-capita consumption variables.

The FBS measure is theoretically appropriate for the research question because it captures net food supply at the point of dietary absorption — after production, imports, exports, stock changes, processing losses, and non-food uses (seed, feed, industrial) have been accounted for. This is the appropriate dependent variable for testing whether supply-chain factors (logistics, storage, loss) predict food availability, rather than a production-only measure such as cereal output per capita which would not capture the trade, storage, and distribution dimensions.

### 3.4.2 Independent Variables

Variables were operationalised from FAOSTAT and the World Bank World Development Indicators (WDI, 2021 data). Table 3.2 summarises the full variable set across all model specifications.

**Table 3.2: Variable Definitions and Sources**

| Variable | Description | Source | NLP Theme |
|---|---|---|---|
| cereal_yield_kg_per_ha | Cereal yield (kg/hectare) | FAOSTAT | Production |
| fertiliser_kg_per_ha | Fertiliser consumption (kg/hectare of arable land) | WDI | Land/Inputs |
| arable_land_pct | Arable land as % of land area | WDI | Land Resources |
| gdp_per_capita_usd | GDP per capita, current USD | WDI | Baseline control |
| rural_population_pct | Rural population as % of total | WDI | Household |
| agri_employment_pct | Employment in agriculture as % of total employment | WDI/ILO | Household |
| livestock_production_index | FAO livestock production index | FAOSTAT | Production |
| cereal_loss_pct | Estimated cereal loss % (FAO loss data) | FAOSTAT | PHL Theme |
| trade_pct_gdp | Trade (exports + imports) as % of GDP | WDI | Value Chain/Logistics |
| rural_electricity_access_pct | Rural population with electricity access, % | WDI | Infrastructure/Storage |
| fertiliser_efficiency | Cereal yield per kg of fertiliser applied | Derived | Technology |
| food_price_inflation_pct | Food CPI inflation rate | WDI | Economic Access |

Skewed predictors (cereal_yield_kg_per_ha, fertiliser_kg_per_ha, gdp_per_capita_usd, livestock_production_index) were log-transformed prior to regression. All models include HC3 heteroscedasticity-robust standard errors (MacKinnon & White, 1985) to account for the non-constant variance typically observed across countries with widely varying income levels.

### 3.4.3 Data Quality and Missing Value Treatment

A practical challenge in assembling the cross-country variable set is differential data availability across countries. FAOSTAT and World Bank WDI cover the majority of the 160-country sample, but several variables have meaningful missing-data rates. Fertiliser kg per hectare is missing for some Pacific island states and small territories. Agricultural employment percentage is missing or severely lagged for several Gulf states and some African countries where Labour Force Survey data are unreliable. Food price inflation rates are missing for a subset of countries not tracked by the IMF consumer price index system.

The missing data strategy has two tiers. For variables with up to 20 percent missing observations, K-Nearest Neighbour imputation (KNN, k = 5, RANDOM_SEED = 42) is applied before model fitting. This recovers countries missing only one or two predictors due to data reporting lags rather than structural data absence, and is implemented using scikit-learn's KNNImputer on the log-transformed predictor matrix. Variables with more than 20 percent missing observations are not imputed; countries without valid values for such variables are excluded via listwise deletion. In practice, KNN imputation affects trade_pct_gdp (14 missing values), rural_electricity_access_pct (8 missing values), and fertiliser_efficiency (1 missing value) in the primary models. The resulting sample sizes (N = 157 in some robustness specifications, N = 160 in primary models) reflect this combined treatment. Readers should note that KNN-imputed observations are synthetic estimates; results are robust to their exclusion, as confirmed by the reduced-N robustness specifications in Section 4.6. All country-level variable values are from the 2021 reference year where available, with the most recent available year used for countries with missing 2021 data (in practice, fewer than five countries for any single variable).

## 3.5 Sequential Model Building Strategy

The primary empirical strategy involves sequential model building, in which predictor blocks are added incrementally to track their marginal contribution to explanatory power. This approach follows the logic of hierarchical regression used in social science research, providing transparent evidence of where explanatory gains occur.

**Model A (Production Baseline):** Seven predictors representing conventional production-side and structural controls: cereal yield, fertiliser, arable land, GDP per capita, rural population, agricultural employment, livestock production index. This model represents the status quo of production-focused modelling.

**Model B (+ Post-Harvest Loss):** Model A predictors plus cereal_loss_pct. This tests whether PHL adds explanatory power beyond the production baseline, directly testing the literature's PHL emphasis.

**Model C (+ Logistics and Infrastructure):** Model B predictors plus trade_pct_gdp and rural_electricity_access_pct. This tests the logistics and infrastructure channel identified in NMF Topics 5 and 6.

**Model A★ (Baseline on NLP Sample):** Model A estimated on precisely the same 160-country sample used by Model F, enabling a fair same-sample nested F-test comparison. This is a critical methodological safeguard against confounding sample changes with model changes.

**Model F (NLP-Discovered Themes):** Model A★ baseline plus five NLP-discovered predictors: cereal_loss_pct, trade_pct_gdp, rural_electricity_access_pct, fertiliser_efficiency, and food_price_inflation_pct. This is the full NLP-guided model and the subject of the nested F-test.

### 3.5.1 Predictor Transformation and Scaling

Skewed continuous predictors are log-transformed before entering the regression. The decision rule is: apply log transformation if the variable's skewness statistic exceeds 1.5. Variables receiving log transformation in the primary models are cereal_yield_kg_per_ha, fertiliser_kg_per_ha, gdp_per_capita_usd, and livestock_production_index. The dependent variable log-cereal_availability_kg_pc is also log-transformed. Log transformation reduces the influence of extreme outliers and linearises multiplicative relationships that are often more appropriate for economic variables across the large income range spanned by the 160-country sample (GDP per capita ranges from under USD 500 to over USD 70,000 in the sample). All other continuous predictors enter untransformed, as their distributions do not exhibit sufficient skewness to require transformation.

No standardisation (z-scoring) is applied, as the substantive interpretation of coefficients in their natural units is preferred for a study where practical significance — not just statistical significance — is relevant to policy interpretation. The coefficient on rural_electricity_access_pct, for example, is directly interpretable as the change in log cereal availability associated with a one-percentage-point increase in rural electricity access.

## 3.6 Nested F-Test Procedure

The nested F-test (also known as the incremental F-test or partial F-test) provides a formal hypothesis test of whether the NLP predictor block adds statistically significant explanatory power beyond the baseline. The null hypothesis is that all five NLP predictor coefficients are jointly zero (H₀: β₈ = β₉ = β₁₀ = β₁₁ = β₁₂ = 0). The test statistic is:

**F = [(R²_full − R²_restricted) / q] / [(1 − R²_full) / (n − k − 1)]**

where q is the number of restrictions (5 NLP predictors), n is the sample size (160), and k is the total number of predictors in the unrestricted model (12). Under H₀, this statistic follows an F(5, 147) distribution. The same-sample comparison using Model A★ and Model F on N = 160 countries is essential for a valid test; comparing models estimated on different samples would conflate sample variation with model specification.

### 3.6.1 Multiple Testing Considerations

The sequential model-building strategy involves multiple hypothesis tests across multiple model specifications, which in principle inflates the family-wise Type I error rate. With five NLP predictors tested individually in Model F plus seven robustness specifications, there are numerous opportunities for false positives. The nested F-test addresses this for the NLP block as a whole — testing the joint null that all five coefficients are zero simultaneously — and is the primary inferential test. For individual predictor significance, the analysis relies on a combination of: (1) p-value thresholds applied consistently (p < 0.05 as the conventional threshold), (2) bootstrap CIs that provide a non-parametric cross-check, and (3) persistence across robustness specifications as the criterion for concluding a finding is robust rather than specification-dependent. Only rural electricity access meets all three criteria, and this is clearly stated in Chapter 4. The pre-specification of the research design in the research proposal, including the nested F-test as the primary inferential procedure, further guards against ex post significance shopping.

## 3.7 Robustness Checks

Seven robustness specifications were estimated to assess the sensitivity of the baseline results:

- **Spec 1 (Baseline replication):** Standard baseline with Spec 1 controls, N = 157 (three countries dropped for missing data compared to primary models).
- **Spec 2 (+ Precipitation):** Adds average annual precipitation as a climate control.
- **Spec 3 (Level DV):** Uses untransformed (level) cereal availability rather than log-transformed, as a functional form check.
- **Spec 4 (No Cook outliers):** Drops eight countries flagged by Cook's Distance as influential observations (DRC, Cabo Verde, Myanmar, PNG, Rwanda, Bhutan, Hong Kong SAR, Congo Rep.).
- **Spec 5 (No ISO outliers):** Drops 16 countries flagged by Isolation Forest as multivariate outliers (including UAE, Bangladesh, Kuwait, Qatar, India).
- **Spec 6 (+ WGI Governance):** Adds the World Governance Indicators political stability score.
- **Spec 7 (Developing only):** Restricts the sample to 108 countries classified as developing by World Bank income criteria, testing whether results hold in the most policy-relevant subsample.

## 3.8 Machine Learning Supplementary Analysis

Random Forest (RF) and XGBoost (XGB) regression models were estimated using five-fold cross-validation for each primary model specification, providing out-of-sample R² estimates that complement the in-sample OLS R² values. The CV R² values provide a check against OLS in-sample overfitting, with the expectation that tree-based ensemble methods may capture non-linearities but may also struggle with the low predictor-to-observation ratio (160 countries, up to 12 predictors) that characterises cross-country datasets.

## 3.9 Bootstrap Confidence Intervals

Bootstrap confidence intervals (1,000 resampling iterations with replacement) were computed for all five NLP predictor coefficients in Model F, providing non-parametric uncertainty bounds that do not rely on normality assumptions for the error distribution. Sampling is stratified by income quartile — countries are assigned to four strata based on their GDP per capita rank and sampled independently within each stratum — so that every bootstrap replicate preserves the proportion of low-, lower-middle-, upper-middle-, and high-income countries present in the observed sample. A 95% CI that excludes zero is interpreted as robust evidence of a non-zero effect.

## 3.10 Ethical Considerations

All data used in this study are publicly available aggregated country-level statistics. No individual-level data are used, and no primary data collection involving human subjects was conducted. There are no ethical approval requirements for secondary analysis of publicly available national statistics. The computational analysis is fully reproducible: the Python scripts used to construct the corpus, run the NLP pipeline, estimate all regression models, and produce all tables and figures are included in the repository (src/ directory) and referenced in Appendix D.

---

# Chapter 4: Findings

## 4.1 Overview of Results

This chapter presents results in the sequence of the research design: first the NLP corpus and topic modelling outputs, then the sequential regression model results, followed by the nested F-test, bootstrap confidence intervals, robustness specifications, and machine learning cross-validation results. All statistics reported here match the computational outputs produced by the Python pipeline scripts described in Chapter 3.

## 4.2 NLP Results

### 4.2.1 Corpus Characteristics

The final strictly aligned corpus contains 127 papers: 113 sourced from OpenAlex, 11 from Scopus, and 3 from curated PDFs. All 127 papers satisfy the dual alignment criteria (food security core plus availability-side signal). Of these, 94 papers (74 percent) contain at least one availability driver theme identifiable by the TF-IDF pipeline, confirming that the corpus is substantively oriented toward the supply-side availability dimensions of food security.

### 4.2.2 LDA Coherence Results

LDA was estimated across K = 3 to K = 12 topics. The best c_v coherence achieved across the sweep was approximately 0.38, occurring at K = 8. This is substantially below the 0.60 threshold pre-specified in the research proposal. The coherence curve (illustrated in Appendix A, Figure A.1) shows flat or marginally rising coherence across K values, with no clear elbow indicating an optimal topic count, consistent with a corpus that is too small and specialised for reliable LDA inference. Given this result, LDA is presented as exploratory context only and is not used as the primary NLP output. This limitation is acknowledged explicitly: the threshold was pre-specified in the research proposal before the final corpus size and composition were determined, and a c_v of 0.38, while below the threshold, does not indicate random or meaningless topic extraction — rather, it reflects the inherent challenge of LDA on small technical corpora.

### 4.2.3 NMF Topic Modelling Results (Primary)

NMF with k = 7 topics produces coherent, interpretable availability-themed clusters. Table 4.1 presents the full NMF topic results including keywords, dominant paper counts, and assigned labels.

**Table 4.1: NMF Topic Extraction Results (k = 7, Primary NLP Output)**

| Topic ID | Label | Top 8 Keywords | Dominant Papers (N) | Share of Corpus (%) |
|---|---|---|---|---|
| 0 | Land, Soil and Water Resources | land, environmental, soil, crop, resource, production, water, demand | 17 | 13.4 |
| 1 | Household-Level Determinants | household, determinant, income, education, rural_household, status, availability, farm_household | 17 | 13.4 |
| 2 | Post-Harvest Loss and Storage | loss, postharvest_loss, fruit_vegetable, postharvest, grain, appropriate, storage, stage | 15 | 11.8 |
| 3 | Climate Change Adaptation | climate_change, adaptation_strategy, adaptation, impact, change, climate, farmer, yield | 16 | 12.6 |
| 4 | Grain Variety and Temperature | rice, yield, variety, silo, wheat, grain, crop, temperature | 7 | 5.5 |
| 5 | Technology and Storage Adoption | technology, adoption, farmer, storage, investment, improved, phl, sensor | 15 | 11.8 |
| 6 | Africa Value Chain and Investment | africa, economic, production, system, waste, smallholder, value_chain, investment | 27 | 21.3 |

Topic 6 (Africa Value Chain and Investment) is the largest single cluster with 27 dominant papers (21.3 percent of the corpus), reflecting the substantial concentration of food security research on sub-Saharan Africa and the value chain lens through which that literature approaches food availability challenges. Topics 2 (Post-Harvest Loss and Storage) and 5 (Technology and Storage Adoption) together account for 30 of 127 papers (23.6 percent), confirming that post-harvest loss and storage technology constitute a prominent thematic cluster in the availability-side literature. Combined with Topic 6's waste and value-chain orientation, PHL-adjacent themes dominate the corpus.

### 4.2.4 Corpus-Wide TF-IDF Term Importance

The 15 highest-scoring corpus-wide TF-IDF terms are: household, loss, climate_change, production, crop, farmer, impact, agriculture, technology, agricultural, rice, yield, availability, africa, system. The prominence of "loss" as the second most important term (after "household") corroborates the NMF finding that post-harvest loss is a central preoccupation of the availability-side literature. The term "availability" appearing in position 13 confirms that the screened corpus is directly oriented toward the research question's dependent variable concept. The appearance of "africa" and "rice" indicates the geographic and crop-specific concentration of the literature.

## 4.3 Regression Model Results

### 4.3.0 Descriptive Statistics and Sample Overview

Before presenting model results, Table 4.0 provides descriptive statistics for the full 160-country sample, establishing the range and distributional properties of the key variables.

**Table 4.0: Descriptive Statistics — Primary Variables (N = 160, 2021)**

| Variable | Mean | Median | Std Dev | Min | Max |
|---|---|---|---|---|---|
| cereal_availability_kg_pc (kg/year) | 161.4 | 155.3 | 67.2 | 119.3 | 714.4 |
| log(cereal_availability_kg_pc) | 5.00 | 5.05 | 0.39 | 4.78 | 6.57 |
| gdp_per_capita_usd (USD) | 14,327 | 5,498 | 20,104 | 252 | 116,935 |
| agri_employment_pct (%) | 25.4 | 20.8 | 22.1 | 0.4 | 79.5 |
| rural_electricity_access_pct (%) | 73.2 | 88.5 | 31.4 | 3.1 | 100.0 |
| cereal_loss_pct (%) | 12.8 | 12.0 | 5.3 | 2.1 | 28.4 |
| trade_pct_gdp (%) | 91.4 | 78.3 | 56.2 | 15.8 | 408.2 |
| fertiliser_kg_per_ha | 103.4 | 52.1 | 163.7 | 0.5 | 1,412.4 |

The dependent variable (cereal availability) has a moderately right-skewed distribution before log-transformation: the mean (161.4 kg/year) exceeds the median (155.3 kg/year), and the maximum value of 714.4 kg/year (an outlier reflecting countries with very high cereal-based diets or reporting anomalies) is more than four times the mean. After log-transformation, the distribution is substantially more symmetric. GDP per capita is highly right-skewed, consistent with the global income distribution, with a median of USD 5,498 but a mean of USD 14,327 pulled upward by high-income outliers. Rural electricity access shows bimodal characteristics: a large cluster of high-income countries at or near 100 percent rural electrification, and a dispersed lower cluster of lower-income countries with access rates as low as 3.1 percent. This bimodality is relevant for interpreting the electricity coefficient, as much of the identifying variation comes from the lower end of the distribution.

**Figure 4.1: Geographic Distribution of Cereal Food Availability, 2021 (N = 158 countries)**

*[Figure: outputs/figures/choropleth_cereal_availability.png — choropleth map showing quintile bands of kg cereal per person per year across 158 countries, using Robinson projection.]*

Figure 4.1 maps the geographic distribution of the dependent variable across the 158 countries for which FAO Food Balance Sheet data are available. Colour bands represent quintiles of the distribution, with each band containing approximately 32 countries. A notable pattern is that high-income Western nations — including the United States, Canada, Australia, and most of Western Europe — appear in the lower quintiles, while Central and South Asian countries (Afghanistan, Kazakhstan, Uzbekistan) and parts of North Africa appear in the upper quintiles. This counterintuitive pattern reflects the nature of the dependent variable: Element 664 of the FAO Food Balance Sheet captures cereal available for human food consumption specifically, not total agricultural output. In high-income countries, a substantially larger share of cereal production is directed towards animal feed, biofuel feedstock, and industrial processing rather than direct human consumption, which depresses the per capita food supply figure relative to countries where wheat and rice remain the primary staple. This interpretation is reinforced by the robustness check in Section 4.6, which shows that the developing-country subsample (Spec 7) yields a substantially lower R² (0.081), consistent with greater heterogeneity once high-income outliers with diversified food systems are removed.

### 4.3.1 Model A: Production Baseline

Model A regresses log-transformed cereal availability on seven production-side and structural predictors across N = 160 countries, with HC3 robust standard errors.

**Table 4.2: Model A — Production Baseline OLS Results**

| Predictor | Coefficient | p-value | Significance |
|---|---|---|---|
| Constant | 6.777 | 0.000 | *** |
| cereal_yield_kg_per_ha | −0.017 | 0.724 | n.s. |
| fertiliser_kg_per_ha | 0.047 | 0.052 | marginal |
| arable_land_pct | 0.001 | 0.513 | n.s. |
| gdp_per_capita_usd | −0.132 | 0.000 | *** |
| rural_population_pct | 0.001 | 0.581 | n.s. |
| agri_employment_pct | −0.005 | 0.027 | ** |
| livestock_production_index | 0.003 | 0.033 | ** |

**R² = 0.196, Adj R² = 0.159, F-statistic p = 4.14e−10, N = 160**

The most striking result in Model A is the large, highly significant negative coefficient on GDP per capita (−0.132, p < 0.001). This counterintuitive direction reflects the dietary transition away from cereals with rising income: high-income countries have lower per-capita cereal food supply (in kg) not because they have less food but because their diets are more diversified, with protein, dairy, and processed foods displacing direct cereal consumption. This pattern is well documented in the nutrition transition literature (Godfray et al., 2010) and confirms that the model is correctly capturing a structural demographic and dietary relationship rather than an income-depresses-food-supply mechanism.

Agricultural employment share is significantly negative (−0.005, p = 0.027), consistent with the interpretation that high agricultural employment reflects subsistence-scale agriculture with low productivity rather than commercial surplus production. Livestock production index is positively significant (0.003, p = 0.033), reflecting that countries with more developed livestock sectors tend to have more integrated agricultural systems with better storage and market infrastructure. Cereal yield per hectare is not significant (p = 0.724), which is consistent with the interpretation discussed in the literature review: at cross-country level in 2021, yield variation does not translate linearly into per-capita cereal availability because trade, storage, and distribution mediate the relationship substantially.

Model A achieves R² = 0.196 (Adj R² = 0.159), meaning the production baseline explains approximately 19.6 percent of cross-country variance in log cereal availability. The F-statistic is highly significant (p = 4.14 × 10⁻¹⁰), confirming that the predictors jointly differ from zero. The Condition Number of 1,950 (rising to 2,960 in Model F as five NLP-discovered predictors are added) indicates moderate multicollinearity among the baseline predictors, which is acknowledged but does not invalidate the results given HC3 robust standard errors.

### 4.3.2 Model B: Adding Post-Harvest Loss

Model B adds cereal_loss_pct to Model A.

**Table 4.3: Model B — Plus Post-Harvest Loss OLS Results (key change)**

| Predictor | Coefficient | p-value | Significance |
|---|---|---|---|
| cereal_loss_pct | −0.006 | 0.476 | n.s. |

**R² = 0.202, Adj R² = 0.160, Delta R² = +0.006, N = 160**

The PHL coefficient is negative as theoretically expected (higher loss → lower availability) but not statistically significant (p = 0.476). The R² gain from adding PHL is only 0.006, and crucially, the Adjusted R² barely changes (0.159 → 0.160), indicating that the PHL variable adds almost no explanatory power beyond chance. The RF cross-validated R² increases marginally from 0.104 to 0.108, and XGB CV R² from 0.094 to 0.097. These results confirm that post-harvest loss, while theoretically predicted to reduce cereal availability, does not demonstrate statistically significant predictive power at cross-country aggregation level. The direction of the null finding is discussed in Chapter 5.

### 4.3.3 Model C: Adding Logistics and Infrastructure

Model C adds trade_pct_gdp and rural_electricity_access_pct to Model B.

**Table 4.4: Model C — Plus Logistics and Infrastructure (key changes)**

| New Predictor | Coefficient | p-value | Significance |
|---|---|---|---|
| trade_pct_gdp | −0.041 | 0.340 | n.s. |
| rural_electricity_access_pct | 0.004 | 0.015 | ** |

**R² = 0.277, Adj R² = 0.229, Delta R² = +0.075, N = 160**

Model C produces the largest single-block R² gain in the sequential build-up: adding two logistics and infrastructure variables increases R² by 0.075, from 0.202 to 0.277. The Adjusted R² increases from 0.160 to 0.229, confirming that this gain exceeds the mechanical effect of adding degrees of freedom. The driver of this gain is rural_electricity_access_pct, which is significant at the 1 percent level (p = 0.015, coefficient = +0.004). Trade as a percentage of GDP remains non-significant (p = 0.340). The RF CV R² decreases slightly to 0.085, while XGB CV R² increases to 0.143. The contrast between the OLS R² gain and the RF/XGB CV R² pattern suggests that the electricity variable captures a genuine linear relationship that OLS models well, while the tree-based methods are not capturing additional non-linear signal of comparable magnitude.

### 4.3.4 Model A★: Baseline on NLP Sample

Model A★ estimates the same seven-predictor specification as Model A on the N = 160 sample used for Model F, enabling a fair nested F-test. R² = 0.194, Adj R² = 0.157, confirming that sample composition differences between Model A and Model F are negligible (the two-unit difference in R² relative to Model A reflects minor sample variation).

### 4.3.5 Model F: NLP-Discovered Themes

Model F is the full NLP-guided model, adding five NLP-discovered predictors to the Model A★ baseline.

**Table 4.5: Model F — NLP-Discovered Themes Full Results**

| Predictor | Coefficient | p-value | Significance |
|---|---|---|---|
| Constant | 6.714 | 0.000 | *** |
| cereal_yield_kg_per_ha | −0.105 | 0.265 | n.s. |
| fertiliser_kg_per_ha | 0.114 | 0.227 | n.s. |
| arable_land_pct | 0.000 | 0.855 | n.s. |
| gdp_per_capita_usd | −0.138 | 0.000 | *** |
| rural_population_pct | −0.000 | 0.970 | n.s. |
| agri_employment_pct | −0.002 | 0.550 | n.s. |
| livestock_production_index | 0.003 | 0.052 | marginal |
| cereal_loss_pct | 0.004 | 0.639 | n.s. |
| trade_pct_gdp | −0.041 | 0.340 | n.s. |
| rural_electricity_access_pct | 0.004 | 0.012 | ** |
| fertiliser_efficiency | 0.089 | 0.358 | n.s. |
| food_price_inflation_pct | −0.001 | 0.310 | n.s. |

**R² = 0.283, Adj R² = 0.224, F-statistic p = 1.90e−09, N = 160**
**Condition Number = 2,960 (multicollinearity warning)**
**Omnibus p = 0.001, Jarque-Bera p = 7.23e−05, Skew = −0.512, Kurtosis = 4.346**

Model F's overall fit (R² = 0.283) represents a substantial improvement over the production baseline (R² = 0.196), with an adjusted R² of 0.224 confirming that the additional predictors earn their degrees of freedom. The two statistically significant predictors are GDP per capita (*** as in all specifications) and rural electricity access (** stable from Model C). The NLP-specific predictors — cereal loss, trade openness, fertiliser efficiency, and food price inflation — do not individually reach significance, though together they contribute to the NLP block's joint significance as established by the nested F-test.

It is also notable that the cereal_loss_pct coefficient reverses sign between models: negative in Model B (−0.006) and positive in Model F (+0.004). This sign flip is a direct consequence of the multicollinearity flagged by the Condition Number of 2,960 — adding fertiliser efficiency, food price inflation, and the other NLP predictors changes the partial correlations among regressors sufficiently to reverse the direction of the PHL coefficient. Both values are non-significant (Model B p = 0.476; Model F p = 0.639), so neither direction can be considered reliable. The direction of the PHL effect on cereal availability cannot be identified from these cross-country specifications.

The Condition Number of 2,960 is flagged as indicative of potential multicollinearity, primarily between GDP per capita and rural electricity access (which are positively correlated across countries: wealthier countries have better electricity infrastructure). This means that individual coefficient estimates in Model F carry greater uncertainty than the standard errors alone suggest, and conclusions about the relative magnitude of specific predictors should be treated with caution. GDP per capita remains the dominant predictor across all specifications, absorbing some of what would otherwise be attributable to electricity access. In Model C (where fertiliser efficiency and food price inflation are absent), the rural electricity coefficient retains similar magnitude and significance, suggesting that the multicollinearity issue in Model F does not fully undermine this finding.

The Jarque-Bera test (p = 7.23 × 10⁻⁵) rejects normality of residuals, with slight negative skew (−0.512) and moderate excess kurtosis (4.346). Given the HC3 robust standard errors used, non-normality of residuals does not invalidate the regression coefficients or their standard errors, but it may affect inference in small-sample contexts. With N = 160 and reliance on asymptotic properties, this limitation is noted.

## 4.4 Nested F-Test: NLP Block Significance

The nested F-test comparing Model A★ (restricted) to Model F (unrestricted) on the same N = 160 sample yields:

**F(5, 147) = 3.649, p = 0.004 (***)**

This result is the central inferential finding of the dissertation. Under the null hypothesis that all five NLP predictor coefficients are jointly zero, an F statistic of 3.649 with 5 numerator and 147 denominator degrees of freedom has a probability of 0.004 of arising by chance. This is strong evidence (p < 0.01) that the NLP-discovered predictor block adds genuine explanatory power beyond the production baseline.

**Table 4.6: Nested F-Test Summary**

| Metric | Value |
|---|---|
| Models compared | Model A★ (7 predictors) vs. Model F (12 predictors) |
| Sample | N = 160 (identical) |
| Restrictions tested | 5 NLP predictors jointly = 0 |
| F(5, 147) | 3.649 |
| p-value | 0.004 (***) |
| Delta R² | +0.089 |
| Partial R² (NLP block) | 0.110 (11.0%) |
| Adj R² change | 0.157 → 0.224 |

The partial R² of 11.0 percent means that the NLP predictor block explains 11 percent of the variance in log cereal availability that remains unexplained after the production baseline. The Adjusted R² increase from 0.157 to 0.224 — a gain of 0.067 in adjusted terms — confirms that this improvement accounts for the cost of the additional five degrees of freedom and represents genuine explanatory gain. This finding validates the core methodological claim of the dissertation: NLP-guided variable discovery from the academic literature can identify predictors that demonstrably improve cross-country food availability models beyond theory-driven baselines.

## 4.5 Bootstrap Confidence Intervals

Table 4.7 presents the bootstrap 95% confidence intervals for the five NLP predictors in Model F, based on 1,000 resampling iterations.

**Table 4.7: Bootstrap 95% Confidence Intervals — Model F NLP Predictors**

| Predictor | OLS Coef | Bootstrap 95% CI | CI excludes zero? |
|---|---|---|---|
| rural_electricity_access_pct | +0.004 | [+0.001, +0.007] | Yes |
| cereal_loss_pct | +0.004 | [−0.014, +0.020] | No |
| trade_pct_gdp | −0.041 | [−0.119, +0.038] | No |
| fertiliser_efficiency | +0.089 | [−0.661, +0.257] | No |
| food_price_inflation_pct | −0.001 | [−0.003, +0.003] | No |

The bootstrap results confirm the OLS findings. Rural electricity access is the only NLP predictor for which the 95% CI consistently excludes zero, providing non-parametric confirmation of its robust positive association with cereal food availability. The remaining four predictors all have CIs that cross zero, confirming their non-significant status under the more stringent bootstrap uncertainty quantification. The wide CI on fertiliser_efficiency (spanning from −0.661 to +0.257) reflects high coefficient instability due to the multicollinearity-inflated variance in Model F's overcrowded predictor set.

## 4.6 Robustness Specifications

### 4.6.1 Baseline Robustness (Specs 1–7)

Table 4.8 summarises the seven robustness specifications estimated for the baseline production model.

**Table 4.8: Robustness Specification Summary**

| Specification | N | R² | Adj R² | Key Finding |
|---|---|---|---|---|
| Spec 1 — Baseline | 157 | 0.176 | 0.137 | GDP per capita *** |
| Spec 2 — +Precipitation | 153 | 0.255 | 0.214 | Precipitation p = 0.0006 *** |
| Spec 3 — Level DV | 157 | 0.197 | 0.159 | Functional form similar |
| Spec 4 — No Cook outliers (N−8) | 149 | 0.311 | 0.276 | GDP per capita *** |
| Spec 5 — No ISO outliers (N−16) | 141 | 0.196 | 0.154 | GDP per capita *** |
| Spec 6 — +WGI governance | 157 | 0.201 | 0.158 | WGI p = 0.032 ** |
| Spec 7 — Developing only | 108 | 0.081 | 0.017 | Model explains little within developing countries |

Key observations from the robustness specifications. First, GDP per capita is the most robust predictor: it retains significance at the 1 percent level across all specifications where it is included, including after Cook outlier removal. The R² rise from 0.176 to 0.311 in Spec 4 — a 77 percent relative improvement — is the largest single-specification gain in the robustness table and indicates that eight structurally atypical countries are substantially suppressing fit in the full-sample model; this finding is discussed at length in Section 4.6.2. Second, precipitation is a significant control (Spec 2, p = 0.0006), confirming that climate conditions affect cereal availability in ways not captured by the production-side variables. Third, governance quality (WGI political stability) is significant at 5 percent in Spec 6 (p = 0.032), suggesting that institutional quality independently predicts food availability above and beyond production and income variables. Fourth, and most critically for the dissertation's scope, the Developing-country subsample (Spec 7, N = 108) produces R² = 0.081 and Adj R² = 0.017 — a dramatic reduction in explanatory power compared to the full-sample model. This indicates that the model, as specified, explains substantially less variance in cereal food availability within the most food-insecure countries specifically, where within-group heterogeneity (conflict, governance, informal trade, subsistence agriculture) is greatest.

### 4.6.2 Cook's Distance and Isolation Forest Outlier Analysis

Cook's Distance flagged eight countries as influential observations: DRC (Democratic Republic of Congo), Cabo Verde, Myanmar, PNG (Papua New Guinea), Rwanda, Bhutan, Hong Kong SAR, and Congo Republic. Removing them raises R² from 0.176 to 0.311 (Spec 4), an increase of 13.5 percentage points and a 77 percent relative improvement — the largest specification gain in the entire robustness table. This is not a minor diagnostic detail: it reveals that a small number of structurally atypical countries are measurably distorting the full-sample model fit, and that the linear production model characterises the remaining ~150 countries substantially better than the headline R² suggests.

The eight countries fall into three distinct structural categories, each with a different reason for high Cook's Distance:

- **Conflict-affected states (DRC, Congo Republic):** These countries exhibit extreme food insecurity driven primarily by active armed conflict, displacement, and supply-chain collapse — factors entirely absent from the model's predictor set. Their residuals are large because no production-side or income variable can predict food availability in the presence of systematic conflict-induced supply disruption.

- **Import-dependent city-states and small islands (Hong Kong SAR, Cabo Verde):** These economies have near-zero domestic agricultural production and rely almost entirely on imports for cereal supply. The model, built around production-side predictors, structurally cannot fit these cases: agricultural employment, arable land, and fertiliser intensity carry no predictive meaning for economies that do not farm. High R² in these cases would require a trade-and-import focused alternative model specification.

- **Countries with policy-reformed food systems (Bhutan, Myanmar, Rwanda, PNG):** These cases combine recent large-scale agricultural policy changes — including Myanmar's transition away from rice export controls, Rwanda's post-genocide agricultural reconstruction, and Bhutan's organic farming mandate — with unusual combinations of high agricultural employment and low GDP per capita that produce cereal availability outcomes not well-predicted by the cross-sectional input-output relationships captured in the model.

The contrast with the Isolation Forest result is instructive. Dropping 16 Isolation Forest–flagged countries (Spec 5) leaves R² approximately unchanged at 0.196, despite removing structurally unusual cases including Gulf states (UAE, Kuwait, Qatar — near-zero domestic production, trade-financed cereal supply), India (massive scale, complex food policy, public distribution systems), and Bangladesh (very high population density, significant food aid). These countries are structurally unusual enough to be detected as multivariate outliers, but they sit along the model's predictor dimensions in ways that do not dramatically distort the regression plane — their leverage is not translated into disproportionate influence on the fitted coefficients. Cook's Distance specifically identifies leverage combined with residual size; Isolation Forest identifies multivariate anomaly without reference to the outcome. The divergent results confirm that the eight Cook-flagged countries are outliers specifically in the outcome-prediction sense, not merely in the predictor space.

The practical implication is that the model's R² = 0.196 is a conservative estimate of fit for the majority of countries in the sample. For conflict-free, non-island economies without recent dramatic policy regime changes — which represent approximately 150 of the 157 robustness-sample countries — the model explains approximately 31 percent of cross-country variance, a meaningfully more useful characterisation of predictive power. The full-sample R² is depressed by a handful of cases that require category-specific modelling strategies rather than a single cross-country linear model. Future research designs should consider segmenting on conflict status and island geography before applying production-side regression frameworks.

### 4.6.3 Model F Robustness

The primary Model F robustness specification (Model F Spec F1, full NLP-augmented model) was estimated on N = 131 countries (after applying additional data availability restrictions), yielding R² = 0.322, with rural_electricity_access_pct retaining significance (p < 0.05, **). This confirms that the rural electricity finding is not an artefact of the specific 160-country sample and holds in a constrained, higher-data-quality subsample.

## 4.7 Machine Learning Cross-Validation Results

**Table 4.9: Random Forest and XGBoost 5-Fold CV R² by Model**

| Model | OLS R² | RF 5-Fold CV R² | XGB 5-Fold CV R² |
|---|---|---|---|
| Model A (Baseline) | 0.196 | 0.104 | 0.094 |
| Model B (+PHL) | 0.202 | 0.108 | 0.097 |
| Model C (+Logistics) | 0.277 | 0.085 | 0.143 |
| Model F (NLP Themes) | 0.283 | 0.078 | 0.066 |

The cross-validated R² values are notably lower than the corresponding OLS in-sample R² values across all model specifications. Model F shows RF CV R² = 0.078 and XGB CV R² = 0.066, compared to OLS R² = 0.283. This gap indicates that the OLS model is capitalising on in-sample fit that does not fully generalise out-of-sample, which is a well-known phenomenon in small-N, multi-predictor cross-sectional settings. With only 160 observations and 12 predictors (a predictor-to-observation ratio of 1:13.3), the degrees of freedom available for cross-validation are limited, and tree-based ensemble methods in particular tend to require larger samples to realise their non-linearity-capturing potential. The XGB CV R² increasing from 0.094 in Model A to 0.143 in Model C but then dropping to 0.066 in Model F is consistent with diminishing returns to predictor addition in the ensemble context, possibly reflecting the multicollinearity problem identified by the Condition Number.

Notwithstanding these limitations, the general pattern of RF and XGB CV R² values confirms that the production-side variables (Model A) provide the strongest replicable out-of-sample signal, the logistics/infrastructure block (Model C) adds genuine generalisation value particularly for XGB, and the fuller NLP model (Model F) does not clearly improve out-of-sample generalisation despite improving in-sample fit. This pattern is important for situating the nested F-test finding: the NLP block is jointly significant in-sample (p = 0.004) and rural electricity is individually robust, but the full NLP model does not consistently improve predictive performance on held-out data.

## 4.8 Financial Access: Summary of Null Finding

Financial access variables — proxied in earlier iterations of the model pipeline by indicators from the Global Findex (bank account ownership, mobile money usage) and World Bank (domestic credit to private sector as % of GDP) — were tested in model specifications estimated during the analysis but not included in the primary sequential model chain reported above. In all specifications tested, financial access variables failed to demonstrate robust statistical significance for the cereal food availability (supply-side) dependent variable, with coefficients shrinking toward zero once GDP per capita was controlled. Formal reporting of these auxiliary model results is presented in Appendix B. The null finding is consistent with the theoretical position that financial access operates primarily through demand-side and access-pillar channels of food security, rather than through the supply-side availability channel captured by the FAO FBS dependent variable.

## 4.9 SHAP Value Analysis: Variable Importance Across Models

Shapley Additive Explanations (SHAP) values were computed for the Random Forest model specifications to provide a model-agnostic measure of each predictor's contribution to individual predictions, complementing the OLS coefficient analysis. Full SHAP output files are stored at `outputs/tables/shap_Model_A__Baseline_Production.csv` through `outputs/tables/shap_Model_F__NLP-Discovered_Themes.csv`. The pattern of SHAP importance confirms and extends the OLS findings in several ways.

In Model A, SHAP analysis identifies GDP per capita as the highest-importance predictor by a substantial margin, accounting for the largest average absolute SHAP value across all 160 country predictions. Agricultural employment percentage and livestock production index are the next-highest contributors, consistent with the OLS significance pattern. Cereal yield per hectare, despite being non-significant in OLS, appears in mid-ranking SHAP importance for the Random Forest, suggesting that yield captures some non-linear signal — particularly in the interaction with fertiliser intensity — that the linear OLS specification does not detect. This divergence between OLS significance and RF SHAP importance is a reminder that absence of OLS significance is not equivalent to absence of predictive relevance; non-linearity can mask relationships in linear models.

In Model F, SHAP importance confirms GDP per capita as dominant and rural electricity access as the second-highest NLP-block contributor. Cereal loss percentage has near-zero SHAP importance in the Random Forest, consistent with its near-zero OLS coefficient and the measurement error arguments discussed in Chapter 5. The SHAP analysis for Model C shows that adding rural electricity access increases its SHAP importance from near-zero (in Model A where it is absent) to second-place behind GDP per capita, which is consistent with the large R² gain associated with Model C's infrastructure block.

## 4.10 Summary of All Model Comparisons

Table 4.10 consolidates all primary model results for comparison.

**Table 4.10: Complete Model Comparison Summary**

| Model | Predictors (N) | OLS R² | Adj R² | Delta R² | RF CV R² | XGB CV R² | Key Significant Predictors |
|---|---|---|---|---|---|---|---|
| Model A | 7 | 0.196 | 0.159 | — | 0.104 | 0.094 | GDP per capita ***, agri_emp **, livestock ** |
| Model B | 8 | 0.202 | 0.160 | +0.006 | 0.108 | 0.097 | GDP per capita *** (loss n.s.) |
| Model C | 10 | 0.277 | 0.229 | +0.075 | 0.085 | 0.143 | GDP per capita ***, rural electricity ** |
| Model A★ | 7 | 0.194 | 0.157 | — | — | — | GDP per capita *** (NLP-sample baseline) |
| Model F | 12 | 0.283 | 0.224 | +0.089 | 0.078 | 0.066 | GDP per capita ***, rural electricity ** |

The progressive improvement in OLS R² from Model A to Model F is clear, but it is the adjusted R² trajectory and the nested F-test that provide the valid inferential basis for attributing this improvement to genuine predictor relevance rather than mechanical R² inflation from adding more variables. The Adj R² rises from 0.159 (Model A) to 0.224 (Model F), a gain of 0.065, and the nested F-test formally confirms that the NLP block drives this improvement beyond chance (p = 0.004). The largest single gain in Adj R² comes from the infrastructure block in Model C (+0.069 in Adj R²), driven by rural electricity access.

It is also notable that the RF and XGB CV R² values do not track the OLS R² improvements linearly. RF CV R² actually declines from Model A (0.104) to Model F (0.078), while XGB CV R² peaks at Model C (0.143) before declining in Model F (0.066). This suggests that the Random Forest in particular is penalising the addition of noisy predictors in the cross-validated context, providing a useful corrective to the in-sample OLS R² picture. The XGB peak at Model C (the infrastructure model) is consistent with the view that rural electricity access contains genuine, non-linearly-learnable signal that XGB can exploit more effectively than OLS.

---

# Chapter 5: Discussion

## 5.1 The Central Finding: NLP-Guided Discovery Adds Measurable Value

The central empirical finding of this dissertation — that the NLP predictor block adds statistically significant explanatory power to the production baseline (**F(5, 147) = 3.649, p = 0.004**, partial R² = 11.0%) — warrants careful interpretation in light of both the statistical evidence and its limitations. The nested F-test is the appropriate formal test for this comparison because it controls for sample size, degrees of freedom, and model nesting structure; a simple R² comparison between models estimated on different samples or with different degrees of freedom would not provide valid statistical inference.

The finding's methodological significance is that an NLP pipeline applied to a relatively small (127-paper) but carefully aligned corpus can identify empirically relevant predictors that improve upon theory-driven baselines in a cross-country setting. This suggests that computational literature synthesis, when combined with rigorous variable operationalisation and formal hypothesis testing, constitutes a productive complement to the standard literature review plus theory-building cycle that underlies most quantitative food security research. Yadav et al. (2023) advocate for greater use of machine learning in food security prediction but do not examine the NLP-to-prediction pipeline demonstrated here; the present study contributes evidence that such pipelines can yield testable and empirically productive insights.

The qualification is important: the partial R² of 11 percent, while statistically significant, represents modest practical explanatory gain. The five NLP predictors account for 11 percent of the residual variance from a baseline that itself explains only 19.4 percent of total variance. Cross-country food availability is therefore substantially explained by factors not captured in any of the models tested here — factors likely including conflict, informal trade, humanitarian food aid, dietary preferences, and household-level distribution mechanisms that are not measurable as consistent cross-country indicators. This reflects an inherent limitation of cross-country modelling: the aggregation masks the within-country heterogeneity that drives food insecurity at household and community level (Barrett, 2010; Misselhorn, 2005).

## 5.2 Rural Electricity Access: The Robust Infrastructure Signal

Rural electricity access emerges as the single NLP-discovered predictor with robust empirical support: significant at the 5 percent level in OLS (p = 0.012 in Model F, p = 0.015 in Model C), with a bootstrap 95% CI that excludes zero ([+0.001, +0.007]), and retaining significance in the constrained Model F robustness specification (N = 131, p < 0.05). The positive coefficient direction is consistent across all specifications: countries with higher rural electricity access have higher cereal food availability per capita, holding other factors constant.

Interpreting this finding requires attention to what rural electricity access proxies in this context. It is not a direct measure of cold chain infrastructure, grain silo capacity, or agro-processing capital, but it is a credible proxy for infrastructure readiness that enables all of these. Dinkelman (2011) demonstrates that rural electrification has broad economic multiplier effects beyond the direct use of electricity; it enables mechanisation, changes in time use, market integration, and access to information. In the agricultural context, electricity enables mechanised grain drying, refrigeration for perishable foods, pumping for irrigation, and the operation of grain mills and processing equipment — all of which reduce post-harvest losses and improve the quantity and quality of food reaching consumers.

The NMF analysis identifies technology and storage adoption (Topic 5) and Africa value chain investment (Topic 6) as two of the seven dominant themes, with electricity-related infrastructure implicit in both clusters. The empirical finding that rural electricity is the NLP-identified predictor most consistently associated with cereal availability is therefore interpretively coherent: the literature's emphasis on storage technology adoption and value-chain investment maps most cleanly onto a measurable cross-country infrastructure indicator — rural electricity access — rather than onto a direct PHL or financial variable.

This finding has policy implications. If rural electricity access is a reliable predictor of cereal food availability across countries, investments in rural electrification — including off-grid solar solutions that are increasingly cost-competitive — may be more effective at improving food availability than direct agricultural production interventions alone, particularly in the lowest-income countries where electrification gaps are largest. FAO (2015) makes a similar case for infrastructure investment as a food security intervention. However, caution is warranted: the cross-sectional association cannot establish that rural electrification causes improved food availability; the relationship may be confounded by income, governance, and other development process variables that simultaneously drive electrification and food system development.

## 5.3 Post-Harvest Loss: Literature Prominence vs. Empirical Silence

The null finding on post-harvest loss is one of the dissertation's most substantive and intellectually interesting results. NMF Topic 2 (Post-Harvest Loss and Storage, 15 papers) and Topic 5 (Technology and Storage Adoption, 15 papers), together with the significant presence of PHL themes in Topic 6 (Africa Value Chain), establish that post-harvest loss is the most prominent availability-side theme in the academic literature. Yet across every model specification estimated — Model B, Model C, Model F — the cereal_loss_pct coefficient fails to reach statistical significance (p ranges from 0.476 to 0.639), and the bootstrap CI consistently crosses zero.

Several interpretations of this divergence deserve consideration. The measurement hypothesis is the most straightforward: FAO cereal loss percentage estimates are based on heterogeneous, often sparse data that introduce substantial measurement error into the independent variable, attenuating the OLS coefficient toward zero. Sheahan and Barrett (2017) document the data quality challenges in PHL measurement, finding that loss estimates vary substantially depending on whether they are derived from farm surveys, expert consultations, or model extrapolations. Outside sub-Saharan Africa, for which dedicated PHL measurement programmes exist, loss data quality is particularly poor, which affects the full cross-country sample.

The dependent variable adjustment hypothesis provides a complementary explanation. The FAO Food Balance Sheet dependent variable already incorporates a loss adjustment in its accounting framework: it represents food supply after deducting estimated processing and storage losses. If the DV and the loss_pct predictor are derived from the same or related FAO estimation procedures, they are not truly independent, and the predictor may be capturing the residual variation beyond the FBS adjustment rather than the full loss channel. This would mechanically suppress the coefficient.

The aggregation hypothesis draws on the literature: Misselhorn (2005) and Barrett (2010) document that cross-country aggregate models systematically obscure the within-country mechanisms that are most policy-relevant. Post-harvest loss affects food availability through local, farm-level mechanisms — storage quality, transport time, market connectivity — that are highly heterogeneous within countries but are compressed into a single national average in cross-country analysis. This aggregation destroys most of the identifying variation.

The theory-practice gap hypothesis is more provocative: perhaps the academic literature has, for the past two decades, emphasised post-harvest loss as a key food security intervention precisely because it is measurable, tractable, and has a compelling narrative (reducing waste feeds more people), rather than because rigorous cross-country evidence demonstrates its primacy as an availability determinant. Lipinski et al. (2013) and HLPE (2014) make persuasive arguments for PHL reduction, but these are based primarily on production-accounting logic (if we lose X percent, reducing loss frees X percent more food) rather than on demonstrated causal pathways from loss reduction to food availability at population level. The present study's null finding does not refute the PHL narrative — the measurement issues are too severe to draw strong conclusions — but it raises the legitimate question of whether the cross-country signal is as strong as the policy narrative implies.

This gap between literature emphasis and empirical signal is important for both research and policy. Research funders and systematic review authors should recognise that literature prominence does not automatically translate into cross-country empirical predictive power: factors that are well-studied in localised contexts may not produce detectable signals at aggregate level due to heterogeneity, measurement limitations, and endogenous adjustment in the DV. Policy actors should interpret the null cross-country finding as motivation for better PHL measurement and more rigorous sub-national analysis rather than as evidence that PHL is unimportant.

## 5.4 Financial Access: A Demand-Side Factor in a Supply-Side Model

The consistent null finding on financial access variables, documented in this study's iterative model development (reported in Appendix B), is consistent with the theoretical architecture of food security. Financial access — measured through indicators such as bank account ownership, mobile money penetration, or domestic credit — primarily affects the access and stability pillars of food security: households' economic ability to obtain food and their resilience to income shocks. The present study's dependent variable, FAO FBS cereal availability, is a supply-side quantity indicator. The two concepts are theoretically orthogonal in the short run: a country can have widely available cereal food supply that is inaccessible to the poor (supply ≠ access), and a country can have high financial inclusion that does not translate into improved supply if production and logistics infrastructure are inadequate.

Demirgüç-Kunt et al. (2022) document the global expansion of financial inclusion through mobile money, digital payments, and account ownership, but note that the food security impacts of financial inclusion operate primarily through consumption smoothing, agricultural input financing, and risk management — channels that affect whether households can afford food (access pillar) rather than whether food is physically available at national level (availability pillar). Aker and Mbiti (2010) specifically identify information and market integration effects of mobile money as the mechanism most likely to affect food price discovery and market access, which again is an access rather than availability channel.

The null finding on financial access therefore represents confirmatory evidence for the theoretical framework rather than a puzzle: the model is correctly specified to detect availability-side predictors, and financial access, a predominantly demand-side variable, does not register in this specification. Future research examining the access and stability pillars of food security with appropriate dependent variables — household expenditure share on food, dietary diversity scores, food consumption scores — would provide a more appropriate test of financial access effects.

## 5.5 GDP Per Capita: The Dominant and Persistent Signal

GDP per capita is the strongest predictor across all model specifications and robustness checks, with a large, highly significant, and consistently negative coefficient. This finding requires careful interpretation to avoid the conclusion that wealth reduces food availability, which is not the mechanism.

The negative coefficient reflects the structural dietary transition associated with rising income. As countries become wealthier, diets diversify: direct cereal consumption per capita falls as protein (meat, dairy, eggs), fruits, vegetables, and processed foods take a larger share of dietary energy. High-income countries such as Japan, Germany, and Canada have lower cereal food supply per capita in the FAO FBS (measured in kg/year) than Bangladesh or Ethiopia not because their food systems are less effective but because their populations consume fewer calories from cereals and more from diverse sources. This is well documented in the nutrition transition literature (Godfray et al., 2010) and is a structural feature of the DV that the model correctly identifies.

The implication for the other model coefficients is that GDP per capita absorbs a substantial portion of the cross-country variance in food availability, leaving the other predictors to explain the residual. This is why the partial R² of the NLP block — 11 percent of the variance remaining after the baseline, which includes GDP — is the appropriate measure of NLP added value, rather than the raw delta R² in absolute terms.

## 5.5.1 Interpreting the Electricity Coefficient Magnitude

A further point of practical interpretation concerns the coefficient magnitude. The OLS coefficient on rural_electricity_access_pct in Model F is +0.004 (on the log-transformed DV). This means that a one-percentage-point increase in rural electricity access is associated with a 0.4 percent increase in cereal food availability per capita, holding all other predictors constant. The average country in the sample with below-median rural electricity access (approximately 50 percent access) has cereal availability of approximately 145 kg/person/year. A 10-percentage-point improvement in rural electricity access (roughly the difference between 50 percent and 60 percent) would be associated with a 4 percent increase in cereal availability, translating to approximately 5.8 kg/person/year. This is a non-trivial gain for food-insecure countries where caloric margins are slim, though the cross-sectional association cannot be taken as a causal prediction of the impact of specific electrification programmes.

The bootstrap CI for this coefficient, [+0.001, +0.007], translates the uncertainty into practical terms: the lower bound implies a 0.1 percent increase per percentage-point access gain and the upper bound implies a 0.7 percent increase. The range is relatively tight given the cross-country modelling context, providing reasonable confidence in the direction and approximate magnitude of the association.

## 5.6 Model Limitations and Sources of Uncertainty

This dissertation takes the position that explicit, detailed acknowledgement of limitations is part of the scientific contribution. Four categories of limitation are identified.

**Causal identification.** The cross-sectional design precludes causal inference. The finding that rural electricity access is associated with higher cereal food availability could reflect genuine infrastructure effects, reverse causation (more food availability → more investment in infrastructure), or confounding by development processes that simultaneously drive both outcomes. Panel data with country fixed effects and plausible instrumental variables would be needed to approach causal identification, which is a direction for future research.

**Measurement quality.** Three of the five NLP-discovered predictors suffer from measurement quality concerns: cereal_loss_pct (heterogeneous FAO estimation methods), trade_pct_gdp (a coarse proxy for logistics integration), and food_price_inflation_pct (available for fewer countries, with heterogeneous CPI methodologies). Measurement error in independent variables attenuates OLS coefficients toward zero, so null findings for these variables may partly reflect data inadequacy rather than true zero effects.

**Multicollinearity.** The Condition Number of 2,960 in Model F indicates meaningful multicollinearity among the predictors, particularly between GDP per capita and rural electricity access. While HC3 robust standard errors account for heteroscedasticity, they do not resolve multicollinearity-induced coefficient instability. The direction of coefficients is reliable but their magnitudes carry uncertainty, and individual p-values may be somewhat inflated relative to the joint significance test.

**Internal-to-external validity gap.** The RF and XGB cross-validated R² values (0.078 and 0.066 for Model F) are substantially below the OLS R² (0.283), indicating that in-sample OLS fit does not generalise as well as the headline R² suggests. The predictors in Model F are likely capturing a mix of genuine structural associations and in-sample noise that the CV penalty appropriately discounts. This gap is largest in Model F, consistent with the multicollinearity and degrees-of-freedom arguments above.

**Spatial autocorrelation.** A limitation not formally tested in this study is spatial autocorrelation among residuals. Countries share geographic, climatic, and institutional environments with their neighbours — African countries cluster by common rainfall patterns, governance traditions, and infrastructure deficits; European countries cluster by institutional development and market integration. If OLS residuals are spatially clustered, the effective sample size is smaller than N = 160, and standard errors may be understated even after HC3 correction. Future work should apply Moran's I tests to the residuals and consider regional fixed effects or spatially adjusted standard errors as additional robustness checks.

**Fertiliser efficiency instability.** The bootstrap CI for fertiliser_efficiency spans [−0.661, +0.257] — nearly a full unit wide around a point estimate of +0.089. This width, roughly ten times the coefficient magnitude, indicates that the variable contributes noise rather than stable signal to Model F, most likely because the ratio-based construction (cereal yield / fertiliser kg per ha) produces extreme values for countries with near-zero fertiliser application. Future specifications should consider replacing this derived ratio with its component variables separately, or excluding it from Model F entirely given its demonstrated coefficient instability.

**Developing-country subsample (Spec 7).** The collapse of explanatory power in the developing-country subsample (R² = 0.081) is a striking limitation with direct policy relevance. The countries most affected by food insecurity are precisely those where this model explains least. This likely reflects the greater role of conflict, governance failure, informal markets, and extreme weather events in food availability outcomes in developing countries — factors that are either unmeasurable or incompletely proxied by the variables available in FAOSTAT and WDI for this sample.

### 5.6.1 The Developing-Country Subsample Problem in Depth

The collapse of explanatory power in the developing-country subsample (Spec 7, N = 108, R² = 0.081, Adj R² = 0.017) deserves more extended discussion because it is the most policy-relevant finding regarding model limitations. The countries classified as developing account for the vast majority of global food insecurity: FAO, IFAD, UNICEF, WFP, and WHO (2023) document that the absolute number of food-insecure people is concentrated in South Asia and sub-Saharan Africa, both of which fall predominantly in the developing-country classification.

Within the developing-country subsample, the model's primary variables — GDP per capita, fertiliser intensity, agricultural employment — show greatly reduced coefficients and significance levels. This reduction is not necessarily because these variables are unimportant in developing countries; rather, it reflects that within the developing-country group, the variation in cereal food availability is driven by factors that are either poorly measured or absent from the variable set. Three candidate explanatory gaps stand out.

First, conflict and fragility: the Food and Agriculture Organization's Voluntary Guidelines on Food Systems and Nutrition repeatedly identify conflict as the primary driver of acute food insecurity in countries such as Yemen, Syria, the Central African Republic, and South Sudan. None of the variables in the current model — including WGI political stability in Spec 6 — fully captures the disruption to food systems caused by active armed conflict. Second, informal and humanitarian food flows: food aid and informal cross-border trade are major sources of cereal availability in some of the most food-insecure countries, and neither is captured by the FAO FBS in a way that separates them from formal commercial imports. Third, within-country distribution: the FBS measure is a national average; in countries with highly unequal domestic food distribution (many lower-income countries have strong urban-rural food availability gradients), the national average masks the food-insecure population's actual supply.

The Spec 7 result is therefore not a failure of the model per se but a signal that a different modelling strategy — likely incorporating conflict indicators, humanitarian flows, and sub-national disaggregation — would be needed to model food availability in the most food-insecure countries with useful precision. This is a direction for future research rather than a flaw in the current study's design.

### 5.6.2 Influential Observations and the Headline R² Interpretation

The Cook's Distance analysis (Section 4.6.2) carries an important positive implication that is the counterpart to the Spec 7 limitation. Removing the eight Cook-flagged countries (Spec 4) raises R² from 0.176 to 0.311. Critically, the eight flagged countries represent three narrow structural types — conflict-affected states (DRC, Congo Republic), import-dependent city-states or small islands (Hong Kong SAR, Cabo Verde), and policy-reformed agricultural systems (Bhutan, Myanmar, Rwanda, PNG) — that are individually recognisable as poor candidates for a cross-country production-side regression model. No standard single-equation model would be expected to predict food availability in Hong Kong SAR (a city with no agriculture) or DRC (a country in active conflict) using agricultural yield and fertiliser data.

The implication is that the headline R² = 0.196 understates what the model achieves for the majority of its intended scope. For the approximately 150 countries that are neither conflict-active nor structurally import-dependent nor in the midst of dramatic agricultural policy transition, the model explains approximately 31 percent of cross-country variance in cereal availability — a substantially more useful characterisation of explanatory power. Reporting the full-sample R² without this qualification gives a misleadingly pessimistic picture of the model's fit for the countries it was designed to describe.

This finding also has methodological implications for the field. Cross-country food availability models are routinely evaluated using full-sample R², which is depressed by a small number of structural anomalies. Future model evaluations in this domain would benefit from reporting Cook's Distance–trimmed R² alongside the full-sample figure as a standard diagnostic, with explicit identification of the structural categories responsible for high leverage.

### 5.6.2 The Condition Number and Multicollinearity Management

The Condition Number of approximately 2,960 in Model F is a diagnostic indicator that the predictor matrix is poorly conditioned, with near-collinear relationships among some combinations of predictors. This inflates the variance of individual OLS coefficient estimates, which is why the standard errors in Model F are larger than in Model A or Model C for the shared predictors. The primary multicollinearity source is almost certainly the GDP per capita — rural electricity access pair: in the 160-country sample, the correlation between (log) GDP per capita and rural electricity access is strongly positive (richer countries have near-universal electricity access). When both variables are included in the same regression, the model struggles to separately identify how much of the cereal availability variation is due to income effects versus infrastructure effects, producing inflated variance for both coefficients.

The practical implication is that while the joint significance of the NLP block (nested F-test p = 0.004) and the individual significance of rural electricity access (p = 0.012) are reliable findings — both survive the multicollinearity-inflated standard errors — the precise coefficient magnitudes of GDP per capita and rural electricity access in Model F should be interpreted with caution. Model C, which includes rural electricity but not the full set of NLP predictors, has a lower Condition Number (2,840) and provides a cleaner estimate of the electricity coefficient in a less multicollinear environment, where it is similarly significant (p = 0.015).

## 5.7 The NLP-to-Evidence Pipeline: Methodological Contributions and Caveats

The dissertation's methodological contribution — a reproducible pipeline from corpus construction through NLP topic modelling to empirical hypothesis testing — offers a template for future research in food security and adjacent policy domains. A limitation of the corpus as constructed is its geographic concentration: Topic 6 (Africa Value Chain and Investment) is the single largest thematic cluster at 21.3 percent of the strictly aligned papers, meaning the literature-driven variable selection reflects sub-Saharan African agricultural development priorities more heavily than the empirical global sample of 160 countries warrants. Future corpus construction should explicitly track and balance geographic coverage, for example by enforcing a maximum proportion of region-specific papers during the screening stage, to avoid biasing NLP-identified predictors toward one world region. Thomas and Harden (2008) argue for methods that can systematically synthesise large qualitative or mixed-methods literature bodies; NLP topic modelling provides such synthesis for quantitative-method corpora.

The LDA coherence shortfall (c_v ≈ 0.38 vs. 0.60 threshold) highlights a practical limitation: LDA's probabilistic inference performs poorly on small, specialised corpora. NMF's algebraic approach is more robust in this context, producing interpretable topics without requiring the corpus size that LDA needs for reliable inference. Future applications of this pipeline should pre-specify NMF as the primary topic model when corpus size is below approximately 200–300 documents, reserving LDA as a supplementary check in larger corpora.

The seven NMF topics identified in this study align closely with the major thematic clusters that a manual systematic review would likely identify: land and water resources, household determinants, post-harvest loss, climate change adaptation, grain variety, technology adoption, and Africa value chain. This convergent validity — that the NMF topics are recognisable to domain experts — provides qualitative support for the pipeline's thematic extraction quality, independent of the formal coherence score.

### 5.7.1 The Coherence Threshold and NMF Validity

The LDA coherence threshold of 0.60, pre-specified in the research proposal, was not met (c_v ≈ 0.38), which requires a discussion of what this means for the validity of the NLP results. The threshold was set in the research proposal based on the expectation that a 127-paper corpus with consistent vocabulary would be large enough for reliable LDA inference. In practice, specialised technical corpora of this size routinely produce lower coherence because: (i) the vocabulary diversity is lower than in general-purpose text collections, reducing the statistical discriminating power of the topic model; (ii) documents in a specialised corpus share many terms across topics, making topic boundaries blurry; and (iii) the short abstracts typically used (rather than full-text) provide less statistical information per document.

The NMF result does not require a coherence score in the same sense: its validity is assessed by inspecting whether the topic-word loadings produce semantically coherent clusters that are recognisable to domain experts. The seven NMF topics identified in this study — land and water, household determinants, post-harvest loss, climate change, grain variety, technology adoption, Africa value chain — are all recognisable and well-grounded in the food security literature, consistent with the themes a manual systematic review would identify. This face validity, combined with the corpus alignment statistics (94/127 papers with at least one availability driver theme), provides sufficient confidence in the NMF output to proceed with empirical operationalisation.

Future studies using NLP for literature synthesis should report both LDA and NMF results, use the coherence score to determine which is primary, and pre-specify the decision rule in the research proposal (as done here). If LDA coherence falls below threshold, NMF should be used as primary with explicit justification — the approach adopted in this dissertation.

## 5.8 Implications for Food Security Policy and Research

The findings of this dissertation have implications at three levels. At the measurement level, the null finding on post-harvest loss in cross-country models underscores the need for better PHL measurement globally, not just in Africa where dedicated measurement programmes exist. Without better data, the true cross-country significance of PHL as an availability determinant cannot be established from aggregate analysis. The policy implication is that organisations such as FAO, CGIAR, and national statistical institutes should prioritise harmonised, survey-based PHL measurement that is comparable across countries and regions, enabling future cross-country analyses to conduct clean tests of the PHL-availability relationship.

At the infrastructure investment level, the robust rural electricity finding supports the argument that infrastructure investment — particularly rural electrification as an enabling platform for storage, processing, and market connectivity — may be one of the most empirically defensible cross-country levers for improving cereal food availability. Coupling this with the Spec 2 precipitation finding (p = 0.0006) suggests that climate-resilient infrastructure investment — electrification that enables irrigation, grain drying, and cold storage — is particularly warranted given the climate vulnerability documented in Spec 2. This finding aligns with the growing international consensus on the importance of off-grid renewable energy solutions for agricultural transformation in rural sub-Saharan Africa and South Asia, where rural electrification rates remain lowest. Off-grid solar mini-grids and individual solar home systems can provide the electricity needed for grain drying, cold storage, and agro-processing at costs increasingly competitive with grid extension (World Bank, 2021).

At the research methodology level, the NLP-to-evidence pipeline demonstrated here suggests that systematic computational literature synthesis, followed by formal hypothesis testing, can identify empirically productive predictors that theory-alone approaches might not prioritise. The finding that rural electricity access — rather than post-harvest loss directly — emerges as the most robust NLP-guided predictor illustrates how the computational approach can cut through narrative emphasis in the literature (PHL) to identify the enabling infrastructure (electricity) that underlies the policy solution. This methodological contribution is transferable: the same pipeline applied to literature on water security, maternal health, or educational attainment could yield comparable benefits for those research domains.

### 5.8.1 The Climate–Availability Nexus

The significance of average annual precipitation in Spec 2 (p = 0.0006, a stronger signal than any individual NLP predictor except GDP per capita) deserves attention in the context of food security policy. NMF Topic 3 — Climate Change Adaptation — is the third-largest thematic cluster in the corpus (16 dominant papers), reflecting the academic community's well-founded concern that climate change will be among the most important determinants of food availability in coming decades. The Spec 2 result confirms that precipitation is already a significant cross-country predictor of cereal availability in the 2021 cross-section.

The interaction between climate and infrastructure is particularly important. Countries with low precipitation face dryland agricultural challenges that can be partially mitigated by irrigation infrastructure, which in turn requires electricity for pumping. Countries with high but unpredictable precipitation face post-harvest loss from moisture damage that can be mitigated by grain drying technology, again requiring electricity. The rural electricity coefficient, therefore, may partially proxy the capacity to manage climate variability through technology rather than just the static electricity access level — a mechanism that would strengthen the causal interpretation of the association if confirmed by panel or quasi-experimental studies.

The absence of a direct climate change adaptation variable in the primary models (only precipitation is tested in Spec 2) is a limitation. Future research should incorporate country-level adaptation capacity indices, climate vulnerability indices, or extreme weather event frequency as controls in food availability models, given the strong theoretical and emerging empirical support for climate effects on cereal production and availability.

### 5.8.2 The Governance–Infrastructure Nexus

The WGI political stability coefficient in Spec 6 (p = 0.032) suggests that governance quality is a relevant predictor of cereal food availability above and beyond production and income variables. The mechanism is plausible: politically stable countries with effective institutions are better able to maintain infrastructure (including rural electrification), implement agricultural policies, and maintain the supply chains that translate production into food availability. Kaufmann et al. (2010) document that governance quality is strongly associated with public infrastructure investment and service delivery quality, providing a pathway from governance to the electricity access that this study identifies as the proximate predictor of food availability.

The implication for model specification is that rural electricity access may itself be a mediator of governance effects on food availability: better governance → more infrastructure investment → higher rural electrification → better storage and logistics → higher cereal availability. This mediation hypothesis could not be tested in the cross-sectional design but is a productive direction for future structural equation modelling or path analysis using panel data.

---

# Chapter 6: Conclusion

## 6.1 Summary of the Research

This dissertation set out to answer the question of which factors emphasised by the food insecurity literature demonstrate measurable predictive value in cross-country machine learning models of cereal food availability, with particular focus on post-harvest loss and financial access. The research pursued this question through a three-stage design: systematic corpus construction and NLP topic modelling (Phase NLP), sequential cross-country regression model building using FAO and World Bank data (Phase Empirical), and formal hypothesis testing of the NLP block's explanatory contribution (Phase Inference).

The NLP stage produced seven coherent thematic clusters from 127 strictly aligned papers, using NMF as the primary topic model after LDA fell below the coherence threshold pre-specified in the research proposal. The themes — land and water resources, household determinants, post-harvest loss and storage, climate change adaptation, grain variety, technology adoption, and Africa value chain investment — represent the food security literature's dominant emphases on the availability side. Post-harvest loss and Africa value chain themes together account for the largest share of the corpus.

The empirical stage estimated four primary OLS models with HC3 robust standard errors plus seven robustness specifications and RF/XGB cross-validation. The dependent variable — FAO FBS cereal food supply per capita (kg/year), capturing net national supply after trade and stock adjustments across 160 countries — is appropriate for availability-side modelling. The sequential model build-up shows that the production baseline (R² = 0.196) is improved most by the logistics and infrastructure block (Delta R² = +0.075, Model C) and that the full NLP model achieves R² = 0.283.

The inference stage's central result is the nested F-test finding: **F(5, 147) = 3.649, p = 0.004**, with partial R² = 11.0% and Adjusted R² increasing from 0.157 to 0.224. The NLP predictor block is jointly statistically significant, confirming that NLP-guided variable discovery adds genuine explanatory power. The single predictor with robust individual support is rural electricity access (p = 0.012, bootstrap CI excludes zero), an infrastructure proxy for the storage and logistics themes dominant in the NMF analysis. Post-harvest loss (p = 0.476–0.639, CI crosses zero) and financial access variables produce null findings across all specifications.

### 6.1.1 Situating the Results Within the Broader Literature

The findings of this dissertation are in broadly consistent with, while extending, the existing cross-country food security modelling literature. Misselhorn (2005) identifies infrastructure and governance as key drivers of food insecurity in southern Africa at subnational level; the present study confirms that rural infrastructure (proxied by electricity access) is significant at cross-country global level. Headey and Ecker (2013) argue for careful dependent variable selection in food security models; the present study's use of the FAO FBS cereal availability measure is directly motivated by their methodological critique and demonstrates the value of a theory-aligned DV. Yadav et al. (2023) and Muckenhuber et al. (2020) advocate for machine learning in food security prediction; the present study shows that OLS retains interpretive advantages for the specific goal of identifying which predictors are significant, while ML provides useful generalisation checks.

The null finding on post-harvest loss at cross-country level is consistent with, though not directly comparable to, Sheahan and Barrett (2017), who note that within-country and farm-level analyses are more informative about PHL mechanisms than aggregate cross-country comparisons. Affognon et al. (2015)'s meta-analysis of PHL in sub-Saharan Africa uses within-region variation that would be aggregated away in a global cross-country study, explaining why their more positive findings on PHL's food security significance do not translate to global aggregate models.

## 6.2 Key Contributions

This dissertation makes three distinct contributions to the food security research literature. First, it provides one of the first demonstrations of a complete NLP-to-cross-country-regression pipeline, in which literature-derived themes are formally tested for empirical predictive power using nested hypothesis testing. This is a methodological template applicable to other policy-relevant research domains. Second, it documents an important and previously underanalysed divergence between literature emphasis and cross-country empirical signal for post-harvest loss, raising questions about the strength of evidence base for PHL's centrality in food availability policy. Third, it provides novel cross-country evidence for rural electricity access as a robust predictor of cereal food availability — a finding that connects the infrastructure development and food security literatures through an empirically validated cross-country analysis.

### 6.2.1 Contribution to NLP-Guided Policy Research

Beyond food security specifically, this dissertation contributes a methodological template for what might be called NLP-guided hypothesis generation and testing. The pipeline has three stages: (1) systematic corpus construction with explicit dual-alignment screening criteria; (2) topic modelling to extract thematic clusters that represent the literature's dominant explanatory frameworks; and (3) formal hypothesis testing of whether operationalised theme proxies add statistically significant predictive power. This three-stage design is transparent about where human decisions enter (topic labelling, variable operationalisation) and where computational analysis drives the output (TF-IDF weighting, NMF decomposition, nested F-test). The combination of computational objectivity and domain expert interpretation is the design's key strength, avoiding both the purely mechanical approach that ignores domain knowledge and the purely narrative approach that ignores statistical rigour.

This template is particularly valuable in research domains where: the literature is large enough to be computationally tractable but too large for comprehensive manual review; no established theory identifies a clear variable priority order; and cross-country or cross-context empirical data are available for hypothesis testing. Food security meets all three criteria, and so do many adjacent domains such as agricultural development, climate adaptation, public health, and infrastructure economics.

## 6.3 Limitations and Future Research

The limitations discussed in Chapter 5 — cross-sectional design, measurement error, multicollinearity, reduced explanatory power in developing countries, and the gap between OLS and CV R² — set a clear agenda for future research. Panel data covering multiple years would enable country fixed effects that control for unobserved heterogeneity and would provide stronger causal identification through within-country variation over time. Improved post-harvest loss measurement — ideally through harmonised, survey-based loss estimation across more countries — would allow a cleaner test of the PHL-availability relationship. Sub-national analyses, using district or province-level data where available, would avoid the aggregation problem that suppresses within-country pathways in cross-country models.

The developing-country subsample finding (R² = 0.081) points to a fundamental gap: the countries most affected by food insecurity are those the cross-country model fits worst. Future research should focus on the missing variables for developing countries — conflict exposure, informal market integration, smallholder heterogeneity, climate shock frequency — that are currently unmeasured or poorly proxied at national level. Conflict and political instability emerge as significant in the WGI governance robustness check (Spec 6) and warrant more direct operationalisation.

The Cook's Distance analysis offers the counterpart qualification: for the majority of the sample — countries that are neither conflict-active, import-dependent city-states, nor undergoing dramatic agricultural policy transition — the production-side model explains approximately 31 percent of cross-country variance (Spec 4, R² = 0.311), compared to the headline 19.6 percent. Future studies should report Cook's Distance–trimmed R² alongside full-sample R² as standard practice, since a small number of structurally anomalous countries can substantially depress fit statistics and obscure the model's genuine explanatory performance for its intended scope.

The NLP pipeline could be extended in future work through word embedding models (following Mikolov et al., 2013) that capture semantic relationships between terms beyond co-occurrence, allowing finer-grained theme extraction from larger corpora. As food security research corpora grow through platforms like OpenAlex, LDA coherence will improve with corpus size, and the LDA versus NMF comparison will become more informative.

### 6.3.1 Priority Areas for Future Research

Four priority directions for future research emerge directly from the findings and limitations of this study.

The first priority is a panel data extension. Constructing a balanced or unbalanced panel of FAO FBS cereal availability data across 10–20 years (2001–2021) would enable country fixed effects that control for all time-invariant country characteristics — geography, history, culture, political system — that confound the cross-sectional analysis. Within-country variation in rural electricity access over this period is substantial in many developing countries, providing the identifying variation needed to estimate a cleaner causal effect. Year fixed effects would control for global price shocks and climate events that affect all countries in a given year.

The second priority is improved post-harvest loss measurement. The null PHL finding cannot be definitively interpreted as evidence that PHL does not affect food availability — it could equally reflect measurement inadequacy. A coordinated international effort to conduct standardised PHL surveys in a representative sample of developing countries, using comparable methodologies across crops and storage stages, would generate the data needed to revisit the PHL-availability relationship with better statistical power and cleaner measures. The FAO Food Loss Index initiative is a step in this direction but remains incomplete.

The third priority is sub-national analysis. Moving from country-level to province or district-level data — where available in countries with subnational agricultural and food security monitoring systems — would recover the within-country heterogeneity that cross-country aggregation destroys. Countries such as India, Ethiopia, Kenya, and Brazil have subnational agricultural data that could support a panel of provinces over time, enabling analysis of how electrification, road access, and storage infrastructure interact with PHL and food availability at the local level where causal mechanisms operate.

The fourth priority is extending the NLP pipeline to full-text analysis rather than abstracts. This study used abstracts and titles for most papers, which limits the depth of thematic extraction. Full-text NLP on the same 127-paper corpus, plus an expanded corpus of several hundred papers using full OpenAlex API access, would produce richer TF-IDF features and potentially improve LDA coherence to above the 0.60 threshold. Word embedding models trained on the corpus (using the approach of Mikolov et al., 2013) would further improve semantic feature extraction, enabling clustering of semantically related terms that surface-form TF-IDF misses (for example, "hermetic storage" and "metal silo" are semantically equivalent but would be treated as separate terms in TF-IDF).

## 6.4 Final Remarks

Food insecurity is a problem that sits at the intersection of agriculture, economics, infrastructure, governance, and climate — a complexity that resists reduction to any single analytical framework. This dissertation has demonstrated that bringing together the tools of computational text analysis and cross-country econometrics can structure that complexity into testable propositions and generate findings that neither approach would produce alone. The finding that rural electricity access is a robust predictor of cereal food availability — identified through a literature that emphasises storage technology and value chain investment — is one such proposition, waiting to be deepened by richer measurement, panel designs, and sub-national analysis. The finding that post-harvest loss, the literature's most prominent availability theme, does not yet demonstrate a robust cross-country signal is another: a challenge to both data collectors and the research community to close the gap between narrative and evidence.

Scientific progress in food security research requires this kind of honest, reproducible confrontation between what the literature says and what the data show. This dissertation is a step in that direction.

---

# References

Affognon, H., Mutungi, C., Sanginga, P., & Borgemeister, C. (2015). Unpacking postharvest losses in sub-Saharan Africa: A meta-analysis. *World Development*, *66*, 49–68. https://doi.org/10.1016/j.worlddev.2014.08.002

Aker, J. C., & Mbiti, I. M. (2010). Mobile phones and economic development in Africa. *Journal of Economic Perspectives*, *24*(3), 207–232. https://doi.org/10.1257/jep.24.3.207

Barrett, C. B. (2010). Measuring food insecurity. *Science*, *327*(5967), 825–828. https://doi.org/10.1126/science.1182768

Bateman, M. (2010). *Why doesn't microfinance work? The destructive rise of local neoliberalism*. Zed Books.

Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. *Journal of Machine Learning Research*, *3*, 993–1022.

Coates, J. (2013). Build it back better: Deconstructing food security for improved measurement and action. *Global Food Security*, *2*(3), 188–194. https://doi.org/10.1016/j.gfs.2013.05.002

Demirgüç-Kunt, A., Klapper, L., Singer, D., Ansar, S., & Hess, J. (2022). *The Global Findex Database 2021: Financial inclusion, digital payments, and resilience in the age of COVID-19*. World Bank. https://doi.org/10.1596/978-1-4648-1897-4

Dinkelman, T. (2011). The effects of rural electrification on employment: New evidence from South Africa. *American Economic Review*, *101*(7), 3078–3108. https://doi.org/10.1257/aer.101.7.3078

FAO. (2015). *The State of Food and Agriculture: Social protection and agriculture — Breaking the cycle of rural poverty*. Food and Agriculture Organization of the United Nations.

FAO. (2019). *The State of Food and Agriculture: Moving forward on food loss and waste reduction*. Food and Agriculture Organization of the United Nations.

FAO, IFAD, UNICEF, WFP, & WHO. (2023). *The State of Food Security and Nutrition in the World 2023: Urbanization, agrifood systems transformation and healthy diets across the rural–urban continuum*. Food and Agriculture Organization of the United Nations. https://doi.org/10.4060/cc3017en

Godfray, H. C. J., Beddington, J. R., Crute, I. R., Haddad, L., Lawrence, D., Muir, J. F., Pretty, J., Robinson, S., Thomas, S. M., & Toulmin, C. (2010). Food security: The challenge of feeding 9 billion people. *Science*, *327*(5967), 812–818. https://doi.org/10.1126/science.1185383

Gröschl, J., & Steinwachs, T. (2017). Do natural disasters cause international trade? *Canadian Journal of Economics*, *50*(1), 286–313. https://doi.org/10.1111/caje.12252

Headey, D., & Ecker, O. (2013). Rethinking the measurement of food security: From first principles to best practice. *Food Security*, *5*(3), 327–343. https://doi.org/10.1007/s12571-013-0253-0

HLPE. (2014). *Food losses and waste in the context of sustainable food systems: A report by the High Level Panel of Experts on Food Security and Nutrition of the Committee on World Food Security*. FAO.

Kaufmann, D., Kraay, A., & Mastruzzi, M. (2010). *The Worldwide Governance Indicators: Methodology and analytical issues* (World Bank Policy Research Working Paper No. 5430). World Bank. https://doi.org/10.1596/1813-9450-5430

Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature*, *401*(6755), 788–791. https://doi.org/10.1038/44565

Lipinski, B., Hanson, C., Lomax, J., Kitinoja, L., Waite, R., & Searchinger, T. (2013). *Reducing food loss and waste* (World Resources Institute Working Paper). World Resources Institute.

Maxwell, S. (1996). Food security: A post-modern perspective. *Food Policy*, *21*(2), 155–170. https://doi.org/10.1016/0306-9192(95)00074-7

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. *arXiv:1301.3781*. https://arxiv.org/abs/1301.3781

Misselhorn, A. A. (2005). What drives food insecurity in southern Africa? A meta-analysis of household economy studies. *Global Environmental Change*, *15*(1), 33–43. https://doi.org/10.1016/j.gloenvcha.2004.11.003

Muckenhuber, J., Dorner, T. E., & Burkert, N. T. (2020). Random forest models for food security analysis. *Frontiers in Sustainable Food Systems*, *4*. https://doi.org/10.3389/fsufs.2020.00052

Mueller, N. D., Gerber, J. S., Johnston, M., Ray, D. K., Ramankutty, N., & Foley, J. A. (2012). Closing yield gaps through nutrient and water management. *Nature*, *490*(7419), 254–257. https://doi.org/10.1038/nature11420

Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. *Proceedings of the eighth ACM international conference on web search and data mining*, 399–408. https://doi.org/10.1145/2684822.2685324

Sen, A. (1981). *Poverty and famines: An essay on entitlement and deprivation*. Oxford University Press.

Sheahan, M., & Barrett, C. B. (2017). Food loss and waste in Sub-Saharan Africa. *Food Policy*, *70*, 1–12. https://doi.org/10.1016/j.foodpol.2017.03.012

Thomas, J., & Harden, A. (2008). Methods for the thematic synthesis of qualitative research in systematic reviews. *BMC Medical Research Methodology*, *8*(1), 45. https://doi.org/10.1186/1471-2288-8-45

Tilman, D., Balzer, C., Hill, J., & Befort, B. L. (2011). Global food demand and the sustainable intensification of agriculture. *Proceedings of the National Academy of Sciences*, *108*(50), 20260–20264. https://doi.org/10.1073/pnas.1116437108

Wang, L. L., Lo, K., Chandrasekhar, Y., Reas, R., Yang, J., Eide, D., Funk, K., Kinney, R., Liu, Z., Merrill, W., Mooney, P., Murdick, D., Rishi, D., Sheehan, J., Shen, Z., Stilson, B., Wade, A. D., Wang, K., Wilhelm, C., … Kohlmeier, S. (2020). CORD-19: The COVID-19 open research dataset. *arXiv:2004.10706*. https://arxiv.org/abs/2004.10706

Webb, P., Coates, J., Frongillo, E. A., Rogers, B. L., Swindale, A., & Bilinsky, P. (2006). Measuring household food insecurity: Why it's so important and yet so difficult to do. *Journal of Nutrition*, *136*(5), 1438S–1448S. https://doi.org/10.1093/jn/136.5.1438S

World Bank. (2021). *World Development Indicators 2021*. World Bank Group. https://datacatalog.worldbank.org/dataset/world-development-indicators

Yadav, K., Gupta, G. K., & Bhatt, D. (2023). Machine learning approaches for predicting food security outcomes. *Food Policy*, *114*, 102395. https://doi.org/10.1016/j.foodpol.2022.102395

Zins, A., & Weill, L. (2016). The determinants of financial inclusion in Africa. *Review of Development Finance*, *6*(1), 46–57. https://doi.org/10.1016/j.rdf.2016.05.001

---

# Appendix A: NLP Pipeline Outputs

## A.1 LDA Coherence Curve

The LDA coherence curve, produced by `src/step3_find_topics_in_papers.py`, is stored at `outputs/figures/lda_coherence_curve.png`. The curve plots c_v coherence against K (number of topics) for K = 3 to K = 12. The highest coherence achieved is approximately 0.38 at K = 8, which is below the pre-specified 0.60 threshold. The curve is relatively flat across K values, with no clear optimal elbow, confirming that the corpus size (N = 127) is insufficient for reliable LDA inference on this specialised technical vocabulary. This figure should be consulted alongside the NMF results to contextualise the decision to use NMF as the primary topic model.

## A.2 NMF Topic Sensitivity Analysis (k = 5 and k = 9)

To assess whether the k = 7 topic solution is robust to the choice of topic number, NMF was additionally estimated at k = 5 and k = 9. Table A.2 presents the top keywords for each topic at each value of k.

**Table A.2: NMF Topic Sensitivity — Top Keywords at k = 5, k = 7 (Primary), and k = 9**

| k | Topic | Top Keywords |
|---|---|---|
| 5 | 0 | climate, change, climate change, adaptation, impacts, food security, yields |
| 5 | 1 | harvest, losses, post harvest, storage, loss, grain, handling |
| 5 | 2 | waste, global, environmental, land, production, agricultural, nutrition |
| 5 | 3 | households, food security, household, food availability, insecurity, availability |
| 5 | 4 | covid, pandemic, rice, health, global |
| 7 | 0 | land, environmental, soil, crop, resource, production, water, demand *(primary result)* |
| 7 | 1 | household, determinant, income, education, rural_household, status, availability |
| 7 | 2 | loss, postharvest_loss, fruit_vegetable, postharvest, grain, storage, stage |
| 7 | 3 | climate_change, adaptation_strategy, adaptation, impact, farmer, yield |
| 7 | 4 | rice, yield, variety, silo, wheat, grain, crop, temperature |
| 7 | 5 | technology, adoption, farmer, storage, investment, improved, phl, sensor |
| 7 | 6 | africa, economic, production, system, waste, smallholder, value_chain, investment |
| 9 | 0 | climate, change, adaptation, impacts, food security, adaptation strategies |
| 9 | 1 | harvest, losses, post harvest, storage, loss, grain, handling |
| 9 | 2 | land, soil, crop, environmental, production, water, agricultural |
| 9 | 3 | households, household, food security, food availability, farm, income |
| 9 | 4 | covid, pandemic, health, initiatives, global |
| 9 | 5 | rice, rice production, asia, yield, drought, global food |
| 9 | 6 | trade, openness, trade openness, africa, countries, food security |
| 9 | 7 | systems, food systems, sustainability, nutrition, sustainable |
| 9 | 8 | waste, food waste, chain, supply, supply chain, financing, interventions |

The core themes are stable across all three solutions. Post-harvest loss (k=5 Topic 1; k=7 Topic 2; k=9 Topic 1), climate change adaptation (k=5 Topic 0; k=7 Topic 3; k=9 Topic 0), land and water resources (k=5 Topic 2 partially; k=7 Topic 0; k=9 Topic 2), and household determinants (k=5 Topic 3; k=7 Topic 1; k=9 Topic 3) appear consistently across all solutions. At k=9 the corpus produces a distinct trade/openness topic (Topic 6) and a food systems/sustainability topic (Topic 7) not present at k=5 or k=7, reflecting the finer granularity available at higher k. The Africa value chain and investment cluster (k=7 Topic 6) is absorbed into the land/environmental topics at k=5 and partially into the trade/openness topic at k=9, but its constituent keywords (africa, value_chain, smallholder, investment, waste) remain present in the solutions. This stability supports the decision to use k=7 as the primary solution.

## A.3 NMF Topic Detail (Primary k = 7)

NMF topic modelling (k = 7) was conducted using scikit-learn's NMF implementation with alternating least squares initialisation. The full topic-word matrix (topics × vocabulary), document-topic matrix (127 papers × 7 topics), and dominant topic assignments are stored at `data/processed/nmf_availability_topics.csv`. The seven topics and their top keywords are reproduced in Table 3.1 and Table 4.1 of this dissertation. The dominant paper counts (ranging from 7 for Topic 4 to 27 for Topic 6) reflect the natural variation in thematic concentration across the corpus, with the Africa value chain cluster being the most represented.

## A.3 TF-IDF Term Rankings

Corpus-wide TF-IDF term rankings, reflecting each term's average importance across all 127 documents, are stored at `data/processed/tfidf_top_keywords.csv`. The top 15 terms are: household, loss, climate_change, production, crop, farmer, impact, agriculture, technology, agricultural, rice, yield, availability, africa, system. These terms informed the operationalisation of NMF-derived themes into empirical predictors.

---

# Appendix B: Financial Access Model Results

Earlier iterations of the model-building pipeline (documented in `outputs/tables/ols_Model_C__PlusNational_Finance.txt` and `outputs/tables/ols_Model_D__PlusValue_Chain_Finance.txt`) included financial access variables alongside production and infrastructure predictors. The key findings from these specifications are:

**Model C (Plus National Finance):** Domestic credit to private sector as a percentage of GDP and bank account ownership (% of adults, from Global Findex) were both non-significant (p > 0.10) in the presence of GDP per capita. Including these variables increased the Condition Number substantially without improving Adj R², confirming that financial access adds no explanatory power to the availability-side model beyond what income controls already capture.

**Model D (Plus Value Chain Finance):** Mobile money account ownership and agricultural credit as a percentage of agricultural value added were similarly non-significant. The GDP per capita coefficient remained highly significant and stable in direction and magnitude, confirming that this variable absorbs the cross-country income variation that financial access indicators also track.

These null findings are consistent with the theoretical position that financial access operates primarily through demand-side food security channels (household food purchasing ability, income smoothing) rather than through the supply-side availability channel captured by the FAO FBS dependent variable. The formal SHAP values for these specifications are stored at `outputs/tables/shap_Model_C__PlusNational_Finance.csv` and `outputs/tables/shap_Model_D__PlusValue_Chain_Finance.csv`.

---

# Appendix C: Multicollinearity Diagnostics

## C.1 Variance Inflation Factors — Model F (12 Predictors)

Variance Inflation Factors (VIF) quantify the degree to which each predictor's variance is inflated by its correlation with other predictors. A VIF above 5 is commonly flagged as indicating meaningful multicollinearity; above 10 is considered severe. The VIF values for all twelve Model F predictors are presented in Table C.1.

**Table C.1: VIF Values — Model F (N = 160, log-transformed predictors)**

| Variable | VIF |
|---|---|
| agri_employment_pct | 5.29 |
| gdp_per_capita_usd | 4.34 |
| fertiliser_kg_per_ha | 3.45 |
| rural_electricity_access_pct | 3.13 |
| rural_population_pct | 2.62 |
| cereal_yield_kg_per_ha | 2.37 |
| cereal_loss_pct | 1.97 |
| fertiliser_efficiency | 1.47 |
| trade_pct_gdp | 1.28 |
| livestock_production_index | 1.16 |
| arable_land_pct | 1.10 |
| food_price_inflation_pct | 1.06 |

The highest VIF is agri_employment_pct (5.29), marginally above the conventional warning threshold of 5, followed by gdp_per_capita_usd (4.34). No predictor exceeds VIF = 10, which would indicate severe multicollinearity. However, the Condition Number of 2,960 (reported in Section 4.3.5) reflects multicollinearity in the full predictor matrix that pairwise VIFs do not fully capture, particularly involving the GDP per capita — rural electricity access correlation (r = 0.684, Table C.2).

## C.2 Pairwise Correlation Matrix — NLP Predictors and GDP Per Capita

**Table C.2: Pairwise Pearson Correlations (Model F NLP Variables + GDP, N = 160)**

| | GDP/pc | Loss% | Trade% | Elec% | Fert.Eff | FoodCPI |
|---|---|---|---|---|---|---|
| GDP per capita | 1.000 | −0.581 | 0.384 | **0.684** | −0.340 | −0.150 |
| cereal_loss_pct | −0.581 | 1.000 | −0.227 | −0.646 | 0.194 | 0.106 |
| trade_pct_gdp | 0.384 | −0.227 | 1.000 | 0.202 | −0.139 | −0.169 |
| rural_electricity_access_pct | **0.684** | −0.646 | 0.202 | 1.000 | −0.367 | −0.103 |
| fertiliser_efficiency | −0.340 | 0.194 | −0.139 | −0.367 | 1.000 | −0.022 |
| food_price_inflation_pct | −0.150 | 0.106 | −0.169 | −0.103 | −0.022 | 1.000 |

The strongest correlation is between gdp_per_capita_usd and rural_electricity_access_pct (r = 0.684), which is the primary source of the inflated Condition Number. cereal_loss_pct is also strongly negatively correlated with both GDP (r = −0.581) and rural electricity (r = −0.646), meaning these three variables share a common development gradient and their individual coefficients carry more uncertainty than the joint F-test result.

---

# Appendix C2: Robustness Specification Details

Full robustness specification results are stored at `outputs/tables/robustness_specifications.csv` and `outputs/tables/robustness_model_f.csv`. The seven primary robustness specifications (Specs 1–7) are described and summarised in Section 4.6 of this dissertation. Key additional information on each specification:

**Spec 2 (+ Precipitation):** Average annual precipitation data sourced from the World Bank's climate knowledge portal and matched to FAOSTAT country codes. The significant positive coefficient on precipitation (p = 0.0006) reflects that higher rainfall enables greater agricultural production, consistent with agronomic literature. The addition of precipitation as a control reduces the coefficient magnitudes on other predictors but does not change the sign or significance pattern for GDP per capita, which remains the dominant control.

**Spec 4 (No Cook Outliers):** The eight countries dropped — DRC, Cabo Verde, Myanmar, PNG, Rwanda, Bhutan, Hong Kong SAR, Congo Republic — span different outlier types: import-dependent city-states (Hong Kong SAR), small island states (Cabo Verde), conflict-affected states (DRC, Congo Republic), and countries with recently reformed food systems (Rwanda, Myanmar). Dropping them increases R² by approximately 13 percentage points (from 0.176 to 0.311), suggesting these countries collectively exert substantial leverage on the model fit.

**Spec 7 (Developing Only, N = 108):** The sharp decline in explanatory power (R² = 0.081, Adj R² = 0.017) in the developing-country subsample is consistent with the hypothesis that cross-country variation in cereal availability among developing countries is driven by factors incompletely captured by the available predictor set — in particular, conflict, governance failure, climate shocks, and informal market dynamics.

---

# Appendix D: Data and Code Availability

All code used in this analysis is stored in the `src/` directory of the research repository:

- `src/step3_find_topics_in_papers.py`: Corpus loading, LDA sweep, coherence scoring, NMF topic extraction, TF-IDF keyword ranking
- `src/step4_score_and_filter_papers.py`: Paper alignment scoring, corpus validation, strict inclusion filtering
- `src/step7_run_prediction_models.py`: Variable assembly, regression models A through F, bootstrap CIs, RF/XGB cross-validation
- `src/step8_check_results_are_reliable.py`: Nested F-test, robustness specifications, VIF checks
- `src/step9_write_the_findings_report.py`: Plain-text synthesis narrative

Processed datasets are stored in `data/processed/`:
- `step3_theme_variable_mapping.csv`: NMF topic-to-variable mapping
- `scored_literature_alignment.csv`: Per-paper alignment scores
- `strictly_aligned_papers.csv`: Final 127-paper corpus
- `tfidf_top_keywords.csv`: Corpus-wide TF-IDF rankings
- `nmf_availability_topics.csv`: NMF topic assignments

Model outputs are stored in `outputs/tables/`:
- `ols_Model_A__Baseline_Production.txt` through `ols_Model_F__NLP-Discovered_Themes.txt`: Full OLS regression output for all primary models
- `bootstrap_confidence_intervals.csv`: 1,000-iteration bootstrap CIs for Model F NLP predictors
- `robustness_specifications.csv`: Summary table of seven robustness specifications
- `literature_alignment_summary.csv`: Corpus source and alignment statistics
- `nlp_empirical_synthesis.csv`: NLP theme to empirical predictor to model result mapping table

All scripts are written in Python 3.10+ and depend on standard scientific libraries (pandas, numpy, scikit-learn, statsmodels, xgboost). The full requirements list is contained in the repository's environment specification. Code is written in a beginner-friendly style with inline commentary to support reproducibility by researchers not specialised in Python programming.

---

*End of Dissertation*

*Odekunle Jibola Johnson | Sheffield Hallam University | May 2026*
