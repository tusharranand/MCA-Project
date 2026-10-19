# Project Workbook Solutions

## AI-Powered Customer Churn Prediction and Intelligent Decision Support Platform

These responses are based on the supplied customer workbook, the implemented local prototype, and its saved model report. The workbook is an academic case study for IntelliServe India. The customer records are observational, and the dataset does not document the business definition or time horizon of churn. Results below describe this dataset and this random holdout; they are not evidence of production performance or causal retention impact.

### Project evidence used

- Customer dataset: 2,200 rows and 18 columns; `Churned` contains 1,917 `No` and 283 `Yes` labels (12.86% positive). The audit found no fully duplicated rows and no missing values in the 18 listed columns.
- Model inputs: 15 columns after excluding `Customer_ID`, `Previous_Churn_Risk_Score`, and `Churned`. The current pipeline therefore does not use the previous risk score, although the planning assumption says it may be valid at scoring time.
- Evaluation: stratified 80/20 holdout with random seed 42; 1,760 training and 440 test records. The saved report compares Logistic Regression, Decision Tree, and Random Forest.
- Selected provisional model: Logistic Regression, with holdout precision 0.2667, recall 0.6316, F1 0.3750, ROC-AUC 0.7663, and accuracy 0.7273. Its confusion matrix is `[[284, 99], [21, 36]]` with rows actual `No/Yes` and columns predicted `No/Yes`.
- The dashboard labels scores below 40% Low, 40%-below 70% Medium, and 70% or above High. These are provisional bands, not validated business thresholds. The current app does not provide feature-level explanations or execute retention actions.

## 1. AI/ML Business Problem Orientation

### 1.1 What business decision will the churn model support, and what should the target represent?

The model supports prioritizing customers for retention review by estimating which customer records resemble those labeled as churned. The available supervised-learning target is the workbook's binary `Churned` column (`Yes`/`No`), and the unit of analysis is one customer record. The model cannot discover the real-world meaning or future prediction horizon from data alone: those must be defined by the data owner. Since the workbook supplies no observation date or label-window definition, this prototype predicts the provided label, not a confirmed future churn event over a known period. Predictions are advisory in nature.

### 1.2 Which three measurable outcomes should define project success?

1. Predictive performance: improve positive-class F1 and ROC-AUC over a majority-class baseline while reporting precision, recall, and the confusion matrix. On the current holdout Logistic Regression has F1 0.3750, recall 0.6316, and ROC-AUC 0.7663; it identifies 36 of 57 churn-positive test records.
2. Data and model reliability: reproducibly load and validate the 2,200-row dataset, prevent identifier/target leakage, and produce repeatable evaluation results. The current audit reports zero missing values and zero fully duplicated rows; unique-ID checks and temporal validity still need explicit verification.
3. Decision-support usefulness: present risk estimates and bands that can help an analyst prioritize review. The current local app demonstrates scoring, but retained customers, campaign efficiency, avoided churn value, and intervention impact were not measured because outcome and action data are not available.

### 1.3 What modeling requirements and constraints should be established before training?

Use `Churned` as the label and exclude `Customer_ID` from predictors. Use only features that exist at scoring time; the current primary model also excludes `Previous_Churn_Risk_Score`, so its validity should be tested separately before adding it. Preserve a stratified, reproducible split and fit all preprocessing only on training data. Report class imbalance and positive-class metrics rather than accuracy alone. Provide human-readable, non-causal interpretation, provisional configurable risk bands, and advisory suggestions only. The current deliverable is a local prototype using the supplied workbook; no production latency, scale, security, or privacy guarantee is claimed. The label horizon, operating threshold, and business cost of errors remain dataset limitations.

## 2. Problem Framing and Success Metrics

### 2.1 How would you define a suitable baseline and acceptance criteria?

Use the majority-class predictor as the minimum baseline: predicting `No` for every record gives 87.14% accuracy on the full dataset but zero recall for churners, so accuracy alone is unsuitable. Compare candidate models on the same stratified split, then require the chosen model to improve useful positive-class metrics over this baseline and report the errors explicitly. Logistic Regression currently achieves F1 0.3750 and recall 0.6316 on the holdout. No business-approved minimum precision, recall, calibration, or utility target exists, so numerical acceptance thresholds must not be invented. The reported comparison is an initial holdout comparison, not a completed cross-validated acceptance study.

### 2.2 What data-quality issues could distort churn modeling?

Potential issues include ambiguous label semantics and horizon, class imbalance, duplicate customer IDs, repeated snapshots, inconsistent category spelling, invalid numeric ranges, measurement errors, partial observation windows, and post-churn information leaking into predictors. In the audit, there are 2,200 rows, 18 columns, zero missing values, zero fully duplicated rows, and 283 positive labels. A full duplicate-row check does not prove customer IDs are unique. There is no date field to check partial time windows or create a temporal split. The data dictionary and field provenance should be checked, especially for the previous score and satisfaction/support variables.

### 2.3 Which fields appear most relevant to customer churn risk?

The observed univariate patterns suggest tenure, recent activity, support-ticket volume, satisfaction, contract type, payment failures, and AutoPay status are useful candidates. For example, churn is 22.7% in the highest `Days_Since_Last_Activity` quartile versus 8.3% in each of the two lowest quartiles; it is 23.0% for customers with 4-9 support tickets in 90 days versus 9.5% with 0-2; and it is 18.3% in the lowest satisfaction quartile versus 8.7% in the highest. Monthly contracts have a 18.4% churn rate versus 9.1% for annual contracts. These are descriptive associations, not causal effects or model feature-importance rankings. Spend and plan type show comparatively small differences in simple group rates.

## 3. Dataset Understanding and Data Quality

### 3.1 What does exploratory analysis reveal about churn patterns?

Overall churn prevalence is 12.86% (283/2,200), so the classes are imbalanced. Churn rates are higher among customers with longer inactivity, more support tickets, lower satisfaction, shorter tenure, monthly contracts, payment failures, or no AutoPay. The rate is 17.0% for tenure up to 19 months and 6.9% above 56 months; it is 17.1% among customers with 2-3 payment failures compared with 12.4% among those with 0-1. Regional rates range from 10.6% in West to 14.8% in South. Plan rates are close, from 12.3% to 13.5%, and age rates are not monotonic. Group comparisons do not adjust for confounding and must not be interpreted as causes. No temporal pattern can be established from this snapshot dataset.

### 3.2 Which model metrics should be interpreted together?

Interpret the confusion matrix with precision, recall, F1, and ROC-AUC, alongside class prevalence and the operating threshold. Precision answers how often flagged customers are labeled churners; recall shows what share of labeled churners are found; F1 summarizes their balance; ROC-AUC measures ranking across thresholds. The current Logistic Regression holdout produced 36 true positives, 21 false negatives, 99 false positives, and 284 true negatives. Its precision is 26.67%, recall 63.16%, and F1 37.50%. This means it finds more than half of positive records but many flagged customers are not positive in the held-out labels. Business costs and intervention capacity are unknown, so no single metric establishes business value.

### 3.3 Which candidate ML algorithms would you compare and why?

Compare a majority baseline, Logistic Regression, Decision Tree, and Random Forest. The first is a transparent performance floor; Logistic Regression offers a relatively interpretable linear baseline; a Decision Tree can capture nonlinear splits but may be unstable; Random Forest can capture interactions and nonlinear patterns. The saved holdout comparison reports Logistic Regression F1 0.3750/ROC-AUC 0.7663, Decision Tree F1 0.2020/ROC-AUC 0.5459, and Random Forest F1 0/ROC-AUC 0.7111. Random Forest predicted every test record as `No` at its default class decision despite a nontrivial ROC-AUC. Logistic Regression is therefore the provisional choice based on reported F1, not proof it is optimal. Gradient boosting could be a later comparator if justified and validated.

## 4. Data Preparation and Exploratory Analysis

### 4.1 How will you prevent target leakage and overfitting?

Drop `Churned` from predictors and exclude `Customer_ID`. The implemented pipeline also excludes `Previous_Churn_Risk_Score`; keep that choice for the primary reported results unless its generation time and point-in-time validity are independently confirmed, and if tested, report a separate ablation. Split before fitting imputation, scaling, or encoding; keep transformations and estimator in one pipeline; stratify by target; tune only within training data; and reserve an untouched test set for final evaluation. The current code uses a stratified 80/20 holdout with seed 42 and training-fitted pipeline transforms. It does not report cross-validation or temporal validation, so stability and future-period generalization remain unverified.

### 4.2 Which engineered features could improve churn prediction?

Start with the supplied fields and add only transparent features that are available at scoring time. Candidates include inactivity bands from `Days_Since_Last_Activity`, engagement levels from `App_Sessions_30D` and `Transactions_90D`, service-friction indicators combining `Support_Tickets_90D` and `Avg_Resolution_Hours`, and a payment-friction count from `Payment_Failures_90D`. A tenure or contract-duration segment may also help summarize lifecycle patterns. Define all windows from data available before the prediction point, then compare each feature set on identical splits. Do not derive features from `Churned`, the outcome period, or post-outcome records; do not claim an engineered feature helps until the evaluation demonstrates it.

### 4.3 How will missing values, categorical variables, and scale differences be handled?

The audited workbook currently has no missing values, but the pipeline should handle future missing data. For numeric fields, use a median imputer and standardization; for categorical fields, use a most-frequent imputer and one-hot encoding with unknown categories ignored. Fit these steps inside the model pipeline on the training partition only and reuse the fitted pipeline for test and inference. The implemented pipeline follows this pattern. Validate input columns and types at loading/scoring time and report invalid values clearly rather than silently changing the source workbook.

## 5. Feature Engineering and Model Design

### 5.1 How should training experiments be reproducible?

Record the dataset path/version and checksum where practical, schema and label assumptions, excluded features, preprocessing steps, split method and random seed, model parameters, Python and package versions, metrics, threshold/risk-band definitions, and model artifact version. Save the report and fitted pipeline together, and provide a command that can regenerate them from the same source data. The project stores a JSON model report and a versioned local model artifact; seed 42 and the 80/20 split are specified in code. A stronger handover should also record data checksums and environment lock/version details so a later run can be compared reliably.

### 5.2 Which validation strategy best estimates generalization performance?

Given one row per customer and no observation timestamp in the supplied fields, use a stratified random holdout plus stratified cross-validation within the training portion. Keep the test set untouched until model and threshold choices are made. If duplicate customers or multiple time snapshots are discovered, use grouped or time-based splitting to prevent the same customer or future information crossing partitions. The current evaluation uses a stratified 80/20 holdout only (440 test cases, 57 positive); it does not include cross-validation. Therefore, the reported metrics are a first estimate and can vary with the split.

### 5.3 How should the classification threshold be selected?

Select a threshold using validation data only, based on the relative impact of missed churners, false alerts, and the number of customers that can be reviewed or contacted. Compare precision-recall and expected workload across candidate thresholds; agree on a business preference before choosing. Then freeze that threshold before evaluating the test set. The prototype currently has a 0.4 prediction threshold and 0.7 high-risk boundary for display, but these are provisional configuration values and were not selected through a documented business-cost optimization. The saved model comparison uses estimator default class predictions, so its metrics should not be presented as performance at a validated 0.4 operating threshold.

## 6. Model Training and Comparison

### 6.1 How should class imbalance be handled?

The positive rate is 12.86%, so stratify the split and use metrics sensitive to the positive class rather than accuracy alone. The implemented Logistic Regression, Decision Tree, and Random Forest use class weighting, and the Logistic Regression holdout recall is 63.16%. Resampling is not required by the current evidence; if tried, it must be applied only inside training folds and compared fairly against class weighting. The Random Forest result is a warning: it achieved 87.05% accuracy but zero recall because it predicted no positive test records. Any imbalance treatment must be judged on precision, recall, PR-AUC, calibration, and practical threshold behavior.

### 6.2 Which metrics are essential for evaluating the churn classifier?

Report the confusion matrix, precision, recall, F1, ROC-AUC, and PR-AUC, along with the 12.86% prevalence and a majority baseline. If probabilities drive prioritization, assess calibration using a calibration curve or Brier score. For this project, also report the count and fraction of customers flagged at each threshold and, if business value data becomes available, lift/gain and expected cost or benefit. The saved comparison currently reports accuracy, precision, recall, F1, ROC-AUC, and confusion matrix; PR-AUC, calibration, lift, and intervention/business value are not yet reported.

### 6.3 What evidence would indicate that the model is overfitting?

Overfitting would be indicated by much stronger training scores than validation/test scores, wide variation across cross-validation folds, unstable feature importance, or sharp degradation on a later time period or a new customer sample. The current report gives holdout scores but no train scores or cross-validation spread, so it cannot establish whether a candidate overfits. The Decision Tree's weak holdout F1 and ROC-AUC suggest poor generalization on this split, but without train-versus-validation evidence they do not alone prove overfitting. Compare these measures before adding complexity.

## 7. Model Evaluation and Validation

### 7.1 What are the highest responsible-AI risks in this project?

Key risks are biased labels or unrepresentative sampling, unequal errors across customer groups, sensitive attributes or proxies influencing prioritization, treating correlation as a causal explanation, presenting an uncalibrated score as certainty, and customers receiving inappropriate treatment from a prototype. `Age` and `Region` are included in the current feature set; subgroup error rates should be checked before any real use. Privacy risk is lower in the declared local academic scope but still requires avoiding unnecessary disclosure of customer records. The UI must make clear that scores are estimates, suggestions are advisory, and the model has not proven that an intervention prevents churn.

### 7.2 How should explainability be incorporated into the solution?

Provide global explanations that summarize which features influence predictions across the test data and local explanations that show which inputs contributed to an individual score. Use coefficients for the Logistic Regression where appropriate, with careful handling of one-hot encoded categories; a model-agnostic method can supplement this if validated. Check explanation stability and avoid causal language. The current prediction response only returns generic advisory text; it does not yet calculate feature-level explanations. Until implemented, do not claim that the dashboard identifies each customer's strongest drivers.

### 7.3 Where is the largest model-performance improvement opportunity?

First improve evaluation and decision selection: the 12.86% positive class and 99 false positives on the Logistic Regression holdout make threshold and precision-recall trade-offs important. Add cross-validation and, if available, temporal data to establish stable performance. Next, validate label meaning and investigate the previous risk score's provenance; it is excluded from the current 15-feature model despite the working assumption that it is available at prediction time. Check data quality, subgroup performance, calibration, and feature usefulness before trying more complex models. Do not optimize on the test set or assume higher accuracy is an improvement.

## 8. Explainable and Responsible AI

### 8.1 How would you prove that a model improvement is genuinely beneficial?

Compare the candidate and baseline using the same train/validation/test partitions, feature availability, preprocessing, positive-class definition, and operating threshold selection method. Tune only on training/validation data and evaluate the final selected model once on the untouched test set. Report uncertainty or repeated-fold variation, confusion-matrix changes, calibration, subgroup metrics, and workload. A business improvement requires outcome evidence such as retention or cost data; higher ROC-AUC or F1 alone does not prove more customers were retained. The current Logistic Regression has the best reported candidate F1, but no intervention outcomes are available.

### 8.2 Which validation checks are mandatory before recommending deployment?

Check schema and target semantics; ID uniqueness and data quality; feature point-in-time availability and leakage; reproducible split and baseline comparison; threshold-specific confusion matrix, PR-AUC and calibration; subgroup sample sizes and error rates; explanation accuracy and limitations; inference behavior for missing/unknown values; artifact/version consistency; input drift; and clear failure behavior. Confirm that the dashboard calls the saved pipeline and labels estimates correctly. The local prototype supports the basic scoring flow, but temporal validation, cross-validation, calibrated probabilities, feature-level explanations, and post-deployment outcome monitoring are not established, so deployment recommendation must remain academic/demo only.

### 8.3 What phased AI/ML implementation roadmap would you recommend?

1. Confirm label semantics and field provenance; verify unique customer IDs and any time structure.
2. Complete data profiling and EDA; document class balance, quality issues, and observed patterns.
3. Run leakage-safe baselines and candidate models with cross-validation, threshold analysis, and an untouched final test set.
4. Add validated global/local explanations, subgroup checks, calibration, and advisory risk/action rules.
5. Complete the local dashboard/API workflow, schema validation, model version display, reproducible setup, and regression tests.
6. Prepare the workbook/report and demo with explicit limitations. Defer production hosting, automated interventions, persistence, and retraining until real operational requirements and outcomes exist.

## 9. Prediction and Decision Support Design

### 9.1 Which three AI/ML findings should management act on first?

1. Inactivity is a practical review signal: customers in the highest inactivity quartile have 22.7% churn versus 8.3% in each of the two lowest quartiles. Review recent engagement and consider a non-coercive re-engagement suggestion.
2. Service friction merits attention: churn is 23.0% among customers with 4-9 support tickets versus 9.5% with 0-2 tickets. Consider service follow-up and resolution review rather than automatically offering a discount.
3. Contract and tenure patterns may help target review: monthly-contract churn is 18.4% versus 9.1% annual, and tenure above 56 months has 6.9% churn versus 17.0% at 1-19 months. Investigate renewal timing and onboarding support. These are descriptive signals only; they should be validated with model explanations and operational context before being used as rules.

### 9.2 How does the final solution balance performance, interpretability, fairness, and usability?

The prototype chooses Logistic Regression as a simple, relatively interpretable baseline with the highest reported holdout F1 (0.3750) and ROC-AUC (0.7663) among the three candidates. Its recall of 63.16% comes with low precision of 26.67%, so analysts should see the score as a prioritization aid, not a definitive churn judgment. The dashboard exposes risk probability, provisional Low/Medium/High bands, and model version. Fairness has not been demonstrated: subgroup error checks remain necessary, especially because Age and Region are predictors. Feature-level explanations and validated threshold choice are not yet implemented; usability should not be mistaken for full transparency.

### 9.3 What evidence supports the final AI/ML recommendation?

The recommendation is to retain Logistic Regression as the provisional academic model and use the local prototype only for advisory scoring. Evidence includes the audited dataset (2,200 records; 12.86% positive labels; no missing values or fully duplicate rows), stratified 80/20 split, and candidate holdout comparison. Logistic Regression achieved F1 0.3750, recall 0.6316, precision 0.2667, ROC-AUC 0.7663, and 36 true positives out of 57 positive test cases. It outperformed the Decision Tree and Random Forest on reported F1, while the Random Forest missed every positive at its default prediction threshold. Limitations include the undocumented churn horizon, a single holdout, no PR-AUC/calibration/fairness results, and no measured retention or financial impact.

## 10. Model Robustness and Optimization

### 10.1 What limitations remain in the analysis?

The target's business meaning and prediction horizon are unknown; a random snapshot split cannot establish future-period performance. The dataset may be synthetic or otherwise unrepresentative, and no data-generating provenance or intervention outcomes are provided. There is class imbalance, one holdout rather than cross-validation, no reported calibration or PR-AUC, and no subgroup fairness evaluation. The previous risk score is excluded from the model even though the planning assumption says it is valid at scoring time; its provenance and contribution should be tested before any decision to include it. The dashboard's bands are provisional and local suggestions are not proven to reduce churn.

### 10.2 How could material customer behavior change affect the model?

Changes in customer mix, product features, service operations, pricing, or usage could cause feature drift (input distributions change) or concept drift (the relationship between features and churn changes). The estimated probabilities and risk ranking could become unreliable. Monitor schema, missingness, category frequencies, feature distributions, and score distributions. When later labels become available, track precision, recall, F1, PR-AUC, calibration, and subgroup outcomes over time. Trigger investigation and evaluation when drift or performance changes materially; retrain only after validation and approval, with a versioned rollback path. The current prototype has no live outcome feedback, so no numeric drift or retraining trigger is validated.

### 10.3 What AI governance controls should remain after implementation?

Maintain a named model/data owner, documented purpose and feature definitions, data access and retention controls, model and dataset versions, reproducible evaluation records, change review before replacing an artifact, monitoring and incident procedures, subgroup fairness checks, and a clear route to withdraw or roll back the model. Keep predictions advisory and avoid unsupported causal claims or automatic service changes. For this local academic prototype, document these controls as recommended operating practices; authentication, approval workflows, persistent action logs, and production-grade governance are outside the current scope.

## 11. ML Deployment and Monitoring Readiness

### 11.1 How should AI/ML outcomes be communicated to management?

Explain that the model estimates similarity to historical `Churned=Yes` records, not certainty or causality. Present churn prevalence, baseline, precision, recall, F1, ROC-AUC, confusion matrix, and the threshold used in plain language. For the current Logistic Regression holdout, it found 36 of 57 positive cases but generated 99 false alerts, so a high-recall review queue also requires analyst capacity. Show the number and share of records in each risk band and state that the bands are provisional. Do not quote retained-customer value or expected savings because no intervention outcome or financial data is available.

### 11.2 Which project artifacts demonstrate reproducibility?

The key artifacts are the approved input workbook and schema/data dictionary, data audit report, preprocessing/model training code, fixed split seed and split strategy, model comparison report, dependency list, saved model pipeline with version, application code, automated tests, run instructions, and this solutions/report documentation. The project contains a JSON model report and a local model artifact, and the README documents startup and test commands. For stronger reproducibility, record a dataset checksum, exact environment/package versions, threshold-selection evidence, and any feature/exclusion decisions with each model version.

### 11.3 What would change if interpretability were prioritized over marginal predictive gain?

Keep Logistic Regression as the primary candidate, provide clear coefficient-based global and local explanations, and compare it with a shallow, constrained Decision Tree or transparent scorecard. Prefer stable, understandable features and explicit decision rules over a small metric gain from a less interpretable model. Quantify any performance trade-off on the same held-out data and retain subgroup and calibration checks. The current candidate results already favor Logistic Regression by F1, but feature-level explanations still need implementation and validation.

## 12. Final AI/ML Recommendation and Roadmap

### 12.1 What would change if the dataset contained highly sensitive personal data?

Minimize collected fields and retain only those necessary for the approved purpose; remove direct identifiers from modeling; restrict access and storage; avoid logging raw records; document retention/deletion; and conduct a privacy and legal review before use. Consider whether sensitive attributes should be excluded from prediction while still being handled securely for fairness auditing, subject to applicable rules. Use synthetic or de-identified demo records. The current local academic prototype should not be treated as permission to use real sensitive data without these controls.

### 12.2 What is the final recommended AI/ML operating state?

For the completed academic scope, use the local Streamlit prototype with the saved Logistic Regression pipeline to score manually supplied customer fields. Show the model version, estimated churn probability, and provisional risk band, and present all results as advisory. Do not automate outreach or treatment; retention suggestions require separate, explicitly defined rules and are not yet generated by the current UI. Re-run the training/audit workflow when the approved dataset changes, review the report before replacing the model, and keep the source data and artifacts local. This is a demonstration workflow, not a production operating state.

### 12.3 How will you verify model success after go-live?

For an academic demonstration, verify that the app starts from the documented launcher, loads the expected model version, validates inputs, produces repeatable outputs, and passes automated tests. For a real pilot, capture authorized outcome labels after a defined horizon and compare precision, recall, F1, PR-AUC, calibration, subgroup performance, and drift against pre-agreed baselines. Evaluate retention impact only with a suitable controlled design and intervention outcomes. Define alert thresholds, review ownership, retraining criteria, and rollback before deployment. The project has no live customer outcomes, so post-go-live success cannot yet be claimed.

## Final findings and recommendations

1. The dataset has useful churn signals, particularly recent inactivity, support-ticket volume, satisfaction, contract type, and tenure, but these are observational associations.
2. Logistic Regression is the provisional model based on the current holdout comparison. Its recall is useful for surfacing many labeled churn cases, but its precision is low and the default Random Forest classifier failed to detect positive cases.
3. Keep the platform local and advisory. Before any real operational use, define the label horizon, validate performance across folds/time and subgroups, assess calibration, add genuine explanations, validate risk thresholds, and test suggestions with outcome data.

## Evidence and interpretation notes

- Dataset counts, data-quality findings, model metrics, and confusion matrices come from `churn_platform/reports/model_report.json` generated by the project training workflow.
- Group churn rates in Sections 2.3, 3.1, and 9.1 were calculated directly from the `Customer Intelligence Data` sheet using quartile bins for numeric fields. They are descriptive and can vary under other grouping choices.
- The project workbook itself is a brief and question template; it does not contain the answers or actual outcome history needed to claim intervention effectiveness.
