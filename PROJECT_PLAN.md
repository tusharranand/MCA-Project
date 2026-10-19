# AI-Powered Customer Churn Prediction and Intelligent Decision Support Platform

## 1. Purpose

Build an academic, end-to-end customer churn prediction and decision-support platform. The system will estimate churn risk from the supplied customer data, explain the main factors behind a prediction, and present a human-reviewed next-best-action suggestion. It is an MCA-level applied machine-learning project: it should demonstrate sound data preparation, model comparison, evaluation, explainability, application design, testing, and responsible use without claiming to be a production-ready retention system.

### Confirmed brief details from the workbook

- Project title: AI-Powered Customer Churn Prediction and Intelligent Decision Support Platform.
- Client context: anonymized client is IntelliServe India, operating a digital service platform with customer profile, engagement, service, transaction, and satisfaction data.
- Business goal: use historical data to predict churn risk, identify leading drivers, support retention prioritization, and define a practical roadmap for responsible deployment.
- Assessment context: the project is mandatory and is followed by an MCQ assessment; the workbook itself is required but is not graded directly.
- Assessment structure: 15 project MCQs and 35 domain MCQs, total 100 marks, with a recommended pass threshold of 50/100.
- Working model: this is a self-driven project; no mentor support is provided, and the student must independently research, analyze, and design the solution.
- Core learning scope: business framing, data quality, EDA, feature engineering, model comparison, evaluation, responsible AI, explainability, decision-support design, robustness, deployment-readiness, and final recommendation.

### Intended users

- **Retention analyst:** reviews a ranked customer list, risk drivers, and suggested follow-up.
- **Project administrator/developer:** imports data, trains or selects a versioned model, and monitors system quality.

The client industry, real operating users, and business workflow are not confirmed. The developer must verify them against the project brief before fixing user-facing terminology or intervention rules.

## 1.1 Implementation status update

The project has now started in a dedicated local prototype folder and the initial scaffold is in place for the academic application workflow.

- [x] Created the implementation workspace and project structure for the churn platform.
- [x] Added the core configuration and CQRS-style service contracts required by the planned architecture.
- [x] Created the initial API shell with health and prediction routes plus an analyst dashboard shell.
- [x] Validated the scaffold contract layer with a direct Python assertion script.
- [x] Confirmed the real workbook schema and target semantics from the approved customer dataset: 2,200 rows, 18 columns, and a binary Churned label with 283 positive cases and 1,917 negative cases.
- [x] Completed the initial data audit and baseline model comparison on the actual customer dataset, producing a reproducible report in the project reports folder.
- [x] Added a threshold-aware risk classification contract to the prediction service to classify low/medium/high churn risk consistently.
- [x] Verified the regression suite passes on the implemented project structure and the churn training workflow remains reproducible.
- [x] Persisted a provisional logistic-regression pipeline and connected the dashboard to it with optional customer fields, session risk-band counts, a pie chart, and a submitted-customer risk table.
- [x] Added regression coverage for model-backed scoring when optional features are missing.
- [x] Fixed the Streamlit dashboard import path so the dashboard script cannot shadow the sibling `app` package.
- [x] Made the dashboard prediction table horizontally scrollable while keeping Customer_ID sticky on the left and Risk percent/Risk band sticky on the right.
- [x] Fixed prediction table contrast and explicit widths so submitted customer values remain visible while the center columns scroll.
- [x] Matched the pinned prediction table's opaque row and header surfaces to the dashboard's dark theme.
- [x] Added a single-click Windows launcher that creates the venv if needed, reinstalls dependencies, retrains the model, and opens the API and dashboard.
- [x] Finalized the project usage guide so the local prototype can be started and tested from a single project entry point.
- [ ] Reconcile the remaining brief-level assumptions around the exact churn definition, client intervention policy, and any business metric preference before finalizing the decision-support rules.

## 1.2 Pre-implementation readiness gate

The following items are the minimum required before coding starts. If any item is still unknown, treat it as a risk and document the assumption.

- [ ] Read and reconcile the PDF brief with the workbook for the exact churn definition, target horizon, and expected deliverables.
- [ ] Confirm the approved data source, workbook names, and whether the dataset is synthetic, anonymized, or restricted.
- [ ] Confirm how sensitive or personal data must be handled and whether any fields are excluded by policy.
- [ ] Verify whether `Previous_Churn_Risk_Score` is available at prediction time and whether it can be used safely as a feature.
- [ ] Confirm the preferred evaluation metric and error trade-off: e.g., recall vs precision, false negatives vs false positives.
- [ ] Confirm whether the project must be a full app/API/dashboard or a lighter local academic prototype.
- [ ] Confirm whether intervening actions are allowed and who can approve them.
- [ ] Confirm whether the platform should be local-only or hosted in any environment.

### Assumptions approved for implementation

The following assumptions are now approved to proceed with the local academic prototype:

1. The exact churn definition and prediction horizon will be inferred from the supplied dataset and documented from the observed target semantics. The model will be trained to predict the available binary `Churned` outcome.
2. `Previous_Churn_Risk_Score` may be used as a feature if it is present in the data and is considered valid at prediction time; otherwise it will be excluded from the primary model and documented as a risk check.
3. The current scope is a local prototype only, with a lightweight dashboard/API workflow. A fuller production-style dashboard can be added later if time and resources permit.
4. Retention actions are advisory suggestions only; human approval is not part of the current scope.
5. Data and privacy restrictions are not expected to be highly strict for this academic prototype, but the plan will still document any sensitive fields and their handling as a limitation.

## 1.2 Step-by-step implementation checklist

1. Requirements confirmation
   - Read the PDF brief and confirm the client use case, churn definition, target variable, and project deliverables.
   - Confirm the staff/users, intervention rules, and final platform scope.
   - Document assumptions in the project plan if the brief is still unclear.

2. Data audit and schema validation
   - Load the approved workbook and inspect rows, columns, dtypes, missing values, duplicate IDs, and target distributions.
   - Validate schema, check class balance, and confirm the meaning of each field.
   - Identify risky columns such as identifiers, leakage-prone features, and the previous churn-risk score.

3. Data quality and EDA
   - Analyse churn patterns by customer profile, tenure, contract, plan, spending, support, and satisfaction.
   - Produce descriptive charts and summary tables for churn behaviour and key field relationships.
   - Record missingness, outliers, bad values, and any data-quality issues.

4. Feature preparation and leakage checks
   - Define a reusable preprocessing pipeline for missing values, categorical encoding, and scaling.
   - Keep identifier fields out of model inputs.
   - Exclude unverified leaky features from the primary model and justify any exceptions.

5. Train/test split and baseline setup
   - Create a reproducible train/test split with a fixed seed and proper stratification.
   - Set a majority-class baseline and other simple comparators.
   - Ensure no preprocessing fit happens on the full dataset.

6. Model development and tuning
   - Train at least a simple baseline model and a small set of standard tabular classifiers.
   - Compare Logistic Regression, Decision Tree, and Random Forest (or equivalent approved alternatives).
   - Tune only on the training data and record model configuration and seeds.

7. Evaluation and threshold selection
   - Report confusion matrix, precision, recall, F1, ROC-AUC, and PR-AUC as relevant.
   - Choose a primary metric based on the business objective and churn cost trade-off.
   - Select threshold using validation data only, then freeze it before final testing.

8. Explainability and responsible AI checks
   - Produce global feature importance and individual explanation outputs for the selected model.
   - Audit subgroup performance and document reliability limitations where sample sizes are small.
   - Record fairness, privacy, and risk concerns as part of the report.

9. Decision support design
   - Define risk bands and decision-support rules based on validated evidence.
   - Avoid automated customer intervention unless approved by the brief.
   - Ensure all recommendations remain advisory and human-reviewed.

10. Application implementation
   - [x] Build the local dashboard or app scaffold using the approved architecture.
   - [x] Add an initial API shell and a reusable service layer to support prediction and analyst review flows.
   - [x] Connect the model pipeline to the project workflow through a reproducible training script that audits the workbook and produces a baseline comparison report.
   - [x] Add a threshold-aware prediction classification layer so the risk service returns consistent low/medium/high outputs and a human-review status.
   - [x] Connect the dashboard to the saved model artifact and display multi-customer session results with risk percentages and risk-band counts.
   - [x] Load customer form fields from the audited dataset schema while excluding the customer identifier, target, and unverified previous-risk score from model inputs.
  - [x] Add the single startup launcher that bootstraps the environment, retrains the model, and opens the API plus dashboard from one entry point.
  - [x] Document the project startup and test commands in the project README for a simple local evaluation flow.
11. API and persistence (if required)
   - Implement query endpoints for customer and prediction review.
   - If action logging is required, use a small persistence layer and enforce CQRS separation between read and write operations.
   - Keep model files, training data, and raw customer records protected.

12. Testing and verification
   - Validate input schema, edge cases, missing values, invalid categories, and prediction response structure.
   - Run regression checks on model loading, input validation, and application behavior.
   - Confirm the app can run and be explained by another developer.

13. Documentation and handover
   - Complete the data-quality report, EDA summary, model comparison results, and responsible-AI notes.
   - Document setup instructions, known limitations, and final recommendations.
   - Prepare the final project report in the format required by the workbook template.

## 2. Source Material and Evidence

- The [client workbook](Client%20Data%20Set%20for%20Student%20Project%20Workbook.xlsx) contains a `Customer Intelligence Data` sheet with 2,200 customer records and 18 columns. The observed target field is `Churned`; the fields include customer profile, tenure, plan, spend, activity, support, payment, satisfaction, discount, contract, and `Previous_Churn_Risk_Score` information.
- The [competitor workbook](Competitor%20Alpha%20Data%20Set%20for%20Student.xlsx) contains a `Data Dictionary` sheet (18 field definitions). It appears to be a schema/reference, not a second labeled customer dataset; verify the workbook before using it for modeling or competitor claims.
- The [student analysis template](Sample%20Student%20Data%20Analysis%20Template.xlsx) provides expected analysis sections: Data Quality; EDA and Churn Patterns; Feature Engineering; Model Comparison; Explainability and Fairness; Deployment and Monitoring; Final Recommendations.
- The [project brief PDF](MCA%20alml%20Project%20workbook.pdf) is present but its text could not be extracted during planning. Its client context, required deliverables, constraints, and grading requirements remain unverified. Read it before implementation and update this plan if it conflicts with these provisional decisions.

## 3. Scope and Boundaries

### In scope

- Profile and validate the supplied labeled customer dataset.
- Build a reproducible tabular classification pipeline and compare a simple baseline with a small set of interpretable/standard models.
- Explain model behavior at global and individual-customer level.
- Present ranked risk, configurable risk bands, and cautious, rule-based retention suggestions for human review.
- Provide a usable local web application, documented API, model/data version information, and a project report aligned with the supplied analysis template.
- Evaluate model quality, data leakage, subgroup performance, basic privacy controls, and application behavior.

### Out of scope unless the brief explicitly requires it

- Automatically contacting customers, changing prices/plans, or triggering retention offers.
- Claims of causal impact, uplift, or proven revenue improvement. The supplied data is observational and has no confirmed intervention outcomes.
- Deep learning, real-time event streaming, cloud-scale infrastructure, or production service-level guarantees.
- Using the competitor dictionary as competitor performance data.

## 4. Assumptions and Decisions to Verify

### Provisional assumptions

- `Churned` is the intended binary label. Confirm its values, meaning, and prediction horizon with the client/brief; do not assume what time window “churn” represents.
- Each row represents one customer at one observation point. No event date or prediction horizon was observed in the listed fields, so a temporal train/test evaluation may not be possible.
- The dataset is authorized for this academic use and contains no unmasked direct personal identifiers beyond the customer ID. This is not yet confirmed.
- The first release is a local academic prototype. Hosting, concurrent-user expectations, and production security requirements are unknown.

### Decisions requiring confirmation before implementation

1. What industry, customer service, churn definition, and prediction horizon does the PDF/client define?
2. Is `Previous_Churn_Risk_Score` generated before the churn outcome and available at prediction time? What model/version produced it? Exclude it from the primary model unless these points are verified; evaluate it only as a separately documented ablation if safe.
3. Are records real, synthetic, anonymized, or otherwise restricted? What data retention and access rules apply?
4. Which intervention actions are actually available, and what are their costs, eligibility rules, and approval owners?
5. Who will use the platform, where will it run, and what does the project brief require for deliverables and evaluation?
6. Which business error is more costly: missed churners or unnecessary retention actions? Agree on the evaluation metric and decision threshold after seeing validation results; do not invent a target threshold.

Record answers and any plan changes in this section before building the final workflow.

## 5. Proposed System Design

### Provisional stack

- Python for data and model work; pandas and scikit-learn for preprocessing, training, and evaluation.
- FastAPI for prediction and customer-query endpoints; SQLite for a small local prototype if prediction/intervention history must persist.
- Streamlit for the analyst-facing dashboard, calling the API rather than duplicating model logic.
- pytest for automated checks; joblib for a versioned local model artifact. Pin dependency versions and document the supported Python version.

Confirm these choices against the PDF, developer environment, and available project time. If the brief expects a simpler single-process application, a Streamlit-only implementation is an acceptable scope reduction, provided model logic remains in reusable modules and the reduced architecture is documented.

### Component flow

1. **Data intake and validation:** load the approved workbook/CSV; check required columns, types, allowed categories, missingness, duplicates, and target values.
2. **Training pipeline:** split data before fitting imputers/encoders/scalers; train candidate models; persist the selected pipeline with a model version and training metadata.
3. **Prediction service:** validate requests, apply the saved preprocessing/model pipeline, return churn probability, risk band, top contributing factors where supported, model version, and a clear human-review status.
4. **Decision support UI:** show aggregate churn patterns, a sortable customer risk queue, customer-level explanation, and eligible rule-based suggestions. Clearly distinguish model estimates from observed facts and business rules.
5. **Persistence (only if required):** keep prediction queries separate from state-changing commands. For the API/database boundary, use CQRS: query handlers read customer/prediction records; command handlers record an approved intervention or analyst feedback. Do not let a prediction request silently change a customer record.

### Suggested endpoint contract

- `GET /health`: service readiness and application version.
- `POST /predict`: validate one or more feature records and return probability, risk band, explanation, model version, and validation errors where applicable.
- `GET /customers` and `GET /customers/{customer_id}`: retrieve permitted customer/prediction views.
- `POST /interventions`: record a human-approved action only if intervention tracking is confirmed in scope; include actor, timestamp, action, and related model version.

Do not expose training data, model files, or unrestricted customer records through public endpoints. Add authentication only to the level required by the confirmed deployment; a local demo must still avoid committing data or secrets.

## 6. Data and Modeling Plan

### Data audit

- Confirm workbook dimensions, headers, types, target values, unique customer IDs, duplicate rows/IDs, missing values, invalid ranges, category spelling, and class balance.
- Produce a data-quality table with finding, impact, and chosen action. Do not silently delete records or impute without documenting the decision.
- Check whether all predictors would be available at the point a real prediction is made. Exclude identifiers from model features. Explicitly investigate `Previous_Churn_Risk_Score` for target leakage and exclude it from the primary model until proven safe.
- Review the data dictionary and reconcile each field's meaning and permitted range with the observed workbook.
- Keep a reproducible, non-destructive cleaning pipeline. Do not overwrite the source workbook.

### Exploration and feature preparation

- Report churn prevalence and counts; inspect churn by tenure, plan, contract, region, activity, support, payment, and satisfaction, using appropriate summaries and plots.
- Use grouped/stratified summaries as descriptive evidence, not proof that a field causes churn.
- Treat `Customer_ID` as an identifier, not a predictor. Encode categorical values and handle missing numeric/categorical values inside a scikit-learn pipeline fitted only on training data.
- Start with the supplied fields. Engineer additional features only when their definition is transparent, reproducible, available at prediction time, and supported by the data (for example, a carefully documented activity or service-friction indicator).
- Check for leakage after every feature change. Do not use `Churned`, post-outcome information, or transformations fit on the full dataset as predictors.

### Split and candidate models

- With no confirmed time field, begin with a stratified train/test holdout and stratified cross-validation on the training portion. Fix and record random seeds. If the brief reveals time-ordered observations or repeat-customer snapshots, redesign the split to respect time/customer grouping.
- Establish a majority-class baseline, then compare Logistic Regression (interpretable baseline), Decision Tree (transparent nonlinear comparator), and Random Forest (robust tabular comparator). Add another boosting model only if time and the brief justify it.
- Fit preprocessing and model as a single pipeline. Tune only on training folds; use the untouched test set once for final evaluation.
- Select and report one primary model using validation evidence and the client-approved error trade-off. If no trade-off is provided, present the metric trade-offs and leave the operating threshold configurable rather than claiming an optimal business decision.

### Evaluation and explainability

- Report confusion matrix, precision, recall, F1, ROC-AUC, and PR-AUC where meaningful; include churn prevalence because it affects interpretation. Compare against the majority baseline.
- Choose the primary selection metric with the client/brief. Discuss false negatives and false positives in plain language. Select any probability threshold using validation data only, then freeze it before final test reporting.
- Check calibration if probabilities are presented as actionable risk estimates. Use calibration plots or a suitable calibration metric and label probabilities clearly if they are not calibrated.
- Provide global feature importance and individual explanations using a defensible method compatible with the selected model. Label explanations as associations used by the model, not causal reasons.
- Compare relevant subgroup performance (at minimum age bands and region if lawful and sufficiently represented). Do not treat small subgroup estimates as reliable; disclose sample sizes and uncertainty. Do not use age or other sensitive attributes to determine intervention eligibility without explicit approval and legal review.

### Decision support rules

- Define risk bands from validation evidence and an agreed operational threshold; never hard-code arbitrary "high/medium/low" cutoffs without documentation.
- Map a small set of observable model drivers to approved, non-coercive review suggestions (for example, service follow-up where support friction is a driver). Confirm action eligibility with the client.
- Each suggestion must state its trigger, reason, and any eligibility guardrail. The analyst must approve and record actions; the model must not automate customer contact or treatment.
- Do not present recommendations as proven to prevent churn. The platform has no confirmed treatment-effect data.

## 7. Application and Report Deliverables

- Working local application with data validation, model-backed prediction, ranked risk review, customer-level explanation, and human-approved decision support.
- Reproducible training/evaluation code and a versioned model artifact created from the supplied/approved data.
- OpenAPI documentation for implemented API routes, if the FastAPI architecture is retained.
- Data dictionary reconciliation and data-quality/EDA evidence.
- Model comparison, selected-model rationale, held-out test results, threshold rationale, leakage checks, and explainability/fairness limitations.
- Developer setup/run instructions, architecture description, test results, known limitations, and a short demo scenario that does not expose restricted customer data.
- Final report organized to match the tabs in the supplied student analysis template.

## 8. Phased Developer Roadmap

| Phase | Work and deliverable | Exit check |
|---|---|---|
| 0. Requirements and feasibility | Read the PDF; reconcile client goals, target meaning, users, action rules, data permissions, required deliverables, stack, and hosting. Update assumptions/open questions. | Stakeholders/brief confirm the churn label, prediction use, required deliverables, and allowed data use; unresolved high-impact items are explicitly blocked or scoped out. |
| 1. Data audit and EDA | Implement a read-only loader, schema/quality checks, descriptive analysis, and a concise data report. | Counts reconcile to the workbook; label distribution, missingness, duplicates, invalid values, and possible leakage are documented; source data remains unchanged. |
| 2. Baseline and model evaluation | Build leakage-safe preprocessing/training pipelines, baseline and candidate models, validation/tuning, and test report. | Reproducible split and seed are recorded; no test leakage; model comparison uses agreed metrics; selected model is compared with baseline on untouched test data. If improvement is absent, report it rather than hiding it. |
| 3. Decision support design | Confirm threshold/risk bands, explanation method, action rules, eligibility guardrails, and analyst review flow. | Each displayed risk/action has a documented definition; actions are advisory and human-approved; unsupported causal claims are absent. |
| 4. Platform implementation | Implement the API and dashboard (or approved reduced architecture), model loading/versioning, request validation, and optional CQRS-based history. | Main user workflow works with valid data; invalid schema/input returns useful errors; query operations do not mutate records; only approved commands record actions. |
| 5. Verification and handover | Run model, API, UI, privacy, fairness, and regression checks; complete report, setup guide, and demo. | Acceptance criteria in Section 9 pass or have explicit known limitations; a new developer can set up and run the project from the guide. |

## 9. Acceptance Criteria

### Data and model

- The approved input schema and `Churned` label semantics are documented; invalid or incompatible files fail with actionable messages.
- Data quality findings and preprocessing decisions are reproducible. Source data is not modified by training or app runs.
- The primary pipeline excludes identifiers and unverified/leaky fields, including the previous risk score until its point-in-time validity is established.
- Train/validation/test boundaries prevent leakage; preprocessing is learned only from training data; random seeds and data/model versions are recorded.
- The report includes the agreed metrics, class balance, majority baseline, confusion matrix, threshold rationale, and honest limitations. Numerical pass targets are set only after the brief/client agrees on the business trade-off and baseline evidence.
- Explanations are consistent with the selected model and explicitly non-causal. Subgroup checks include sample sizes and limitations.

### Platform and responsible use

- A permitted user can submit/inspect a customer record and see the risk estimate, risk band, model version, explanation, and advisory status.
- Bulk or single-record inputs are validated; missing, malformed, and unknown-category behavior is tested and documented.
- The dashboard distinguishes model output, raw data, and business rules. No customer action occurs without human approval.
- If persistence is implemented, commands and queries are separated; intervention records include the approving user/time and model version, with no unnecessary personal data.
- Setup instructions reproduce the environment and launch the application. Health checks and automated tests pass in the documented environment.

## 10. Test and Quality Strategy

- **Data tests:** required columns, target domain, duplicates, numeric ranges, category handling, missing values, and immutability of source files.
- **ML tests:** preprocessing fit boundary, deterministic rerun, no identifier/target leakage, pipeline serialization/reload parity, baseline comparison, and threshold calculation on validation data only.
- **API tests:** valid/invalid payloads, single and batch prediction, model unavailable, health route, response schema, and persistence/query-command behavior if present.
- **UI checks:** upload/validation, risk sorting/filtering, explanation clarity, approved action flow, empty/error/loading states, and no layout or data-label ambiguity.
- **Responsible-use checks:** subgroup results and sample sizes, sensitive feature use, probability calibration language, human-review controls, and logs that avoid customer data where not required.
- **Reproducibility:** record dependency versions, data source/version, split seed, selected model/configuration, and evaluation artifact.

## 11. Deployment and Monitoring

- Target local development/demo unless the brief confirms another environment. Keep source data and model artifacts out of public repositories; configure secrets outside source control.
- Load a known model version at startup and return that version in prediction responses. Fail clearly if the artifact is missing or incompatible.
- Log service health, validation failures, latency, model version, and aggregate prediction counts without logging raw customer features by default.
- Monitor input schema/missingness/category drift and prediction distributions. Once verified outcome labels become available, monitor precision/recall, calibration, and subgroup performance against agreed triggers.
- A shift or quality alert prompts analyst review and possible retraining; it must not silently replace the active model. Record model approval, evaluation, and rollback procedure.
- Since no post-deployment labeled outcomes or retraining cadence is confirmed, monitoring thresholds and retraining frequency remain open decisions.

## 12. Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Churn label or prediction horizon is ambiguous | Resolve from the PDF/client before training; document exact label and use case. |
| Prior risk score leaks target information or embeds an unknown model | Exclude by default; investigate generation time/provenance; use only in a separately reported safe ablation if approved. |
| Small or unrepresentative dataset yields unstable metrics | Show class counts and uncertainty; use stratified validation; avoid production/generalization claims. |
| Random split overstates future performance | Confirm observation-time structure; use temporal or grouped splits if records support them. |
| Sensitive attributes or proxy features lead to unfair decisions | Audit subgroup metrics, restrict intervention use, keep human review, and document legal/business approval. |
| Analysts interpret scores/explanations as certainty or causality | Show probability limitations, model version, uncertainty/threshold context, and non-causal language. |
| Recommendations have no demonstrated retention effect | Present as suggestions only; evaluate impact later with an approved controlled design and outcomes. |
| Scope exceeds an MCA project timeline | Prioritize reproducible audit, baseline comparison, useful dashboard, and transparent report; defer optional advanced modeling/cloud features. |

## 13. Completion Definition

The project is complete when the brief has been reconciled, all agreed acceptance criteria are checked, the application and evaluation are reproducible by another developer, and the final report makes data limitations, model trade-offs, fairness/privacy checks, and human decision responsibilities clear. Any criterion blocked by missing client information must be listed as an explicit limitation, not marked as passed.