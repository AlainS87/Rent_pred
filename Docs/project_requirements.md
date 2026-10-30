# CS5228 Data Mining Project — Requirements README

> Source: CS5228 Project specification, last updated **4 Aug 2026**.  
> This file reorganizes the official project requirements into an implementation-oriented checklist for the team.

---

## 1. Project Goal

The project is a practical **data mining regression task** focused on predicting **monthly rental rates of HDB flats in Singapore**.

Given information about an HDB flat — such as its size, number of rooms, location, age, and related contextual information — the task is to predict its expected monthly rent using historical data.

The competition component is implemented as a **Kaggle InClass Competition**.

### Important emphasis

The project is **not primarily about maximizing Kaggle rank**.

The main focus is on:

- understanding the dataset and problem;
- selecting appropriate data mining methods;
- motivating and justifying design decisions;
- implementing the chosen approach correctly;
- evaluating and interpreting results;
- comparing alternatives;
- identifying failure cases and limitations;
- maintaining reproducible and readable code.

---

## 2. Problem Formulation

### Task type

- **Supervised learning**
- **Regression**
- **Target:** `MONTHLY_RENT` — monthly HDB rental rate in SGD
- **Competition evaluation metric:** Root Mean Square Error (**RMSE**)

The Kaggle test set does **not** include `MONTHLY_RENT`; predictions must be generated for each test row and submitted in the required `Id,Predicted` format.

### Inputs

Potential predictors include:

- temporal information;
- HDB flat characteristics;
- block characteristics;
- geographic/location information;
- nearby landmarks and amenities;
- broader economic indicators;
- additional external data collected by the team.

There is no requirement to use every provided feature.

---

## 3. Dataset

## 3.1 Core Dataset

The competition provides the following main files:

- `train.csv` — training set
- `test.csv` — test set
- `example-submission.csv` — example submission in the required format

### Main data fields

| Field | Meaning | Example / Notes |
|---|---|---|
| `RENT_APPROVAL_DATE` | Year and month when the rent was approved | `"2021-01"` |
| `TOWN` | HDB town | `"Bishan"`, `"Queenstown"`, `"Clementi"` |
| `BLOCK` | Block number | `"300"`, `"306A"`, `"254"` |
| `STREET` | Street name | `"ubi avenue 1"`, `"woodlands drive 14"` |
| `FLAT_TYPE` | Flat type | `"2 room"`, `"3 room"`, `"4-room"` |
| `FLAT_MODEL` | Flat model | `"model a"`, `"dbss"`, `"executive"` |
| `FLOOR_AREA_SQM` | Floor area in square metres | `104.0`, `74.0`, `136.0` |
| `FURNISHED` | Whether the flat is furnished | `"yes"` / `"no"` |
| `LEASE_COMMENCE_DATE` | Year when the lease commenced | `1984`, `1985`, `1986` |
| `FEE` | Additional upfront rental-agreement fee | `200`, `300`, `400` |
| `MONTHLY_RENT` | Monthly rent in SGD | **Training target only**; e.g. `2350`, `4000`, `3200` |

### Train / test difference

- `train.csv` contains the predictor columns plus `MONTHLY_RENT`.
- `test.csv` contains the predictor columns but **does not contain `MONTHLY_RENT`**.
- The Kaggle preview indicates that `test.csv` has **10 columns** and about **50,000 rows**.
- The visible test period spans roughly **March 2025 to July 2026**.
- In the displayed Kaggle preview, `FURNISHED` appears to be constant (`yes`) and `FEE` appears to be `0.0` throughout the shown test data. This should be verified programmatically on the downloaded dataset before making modeling decisions.

### Historical coverage

The course project description states that the consolidated rental data covers approved applications from **January 2021 to June 2026**.

### Important implementation note

Several string fields show inconsistent presentation/casing in the examples, e.g.:

- `Pasir Ris Street 71` vs lowercase street names;
- flat-type formatting such as `4-room` vs natural-language variants;
- block identifiers containing letters, e.g. `869a`, `103a`.

Therefore:

- treat `BLOCK` as a string / categorical identifier rather than a pure integer;
- normalize string casing and whitespace before categorical or text-based processing;
- inspect category normalization carefully before encoding.

---

## 3.2 Auxiliary Data

The Kaggle competition includes an `auxiliary-data` folder containing additional files that may be used for feature engineering.

### Location / point-of-interest data

Most auxiliary files contain locations of places of interest, including:

- existing MRT stations;
- planned MRT stations;
- shopping malls;
- schools / primary schools.

The intended use is to derive spatial features such as:

- distance to nearest MRT;
- distance to nearest planned MRT;
- distance to nearest mall;
- distance to nearest school;
- counts of amenities within a radius;
- accessibility or local-density features.

The course explicitly leaves the feature design up to the team. For example, the team may decide to use only a subset of malls or other landmarks if that choice is justified.

### HDB / block-level information

The broader project description also states that auxiliary information may include HDB block attributes such as:

- number of floors;
- year of completion;
- geolocation / latitude-longitude;
- other block-level details.

### Economic indicators

Auxiliary data also includes:

- Certificate of Entitlement (**COE**) prices;
- stock prices for large Singapore-based companies by market capitalization.

Both are provided over the same general date range as the main rental data and may be joined using temporal features.

### External data collection

The project permits teams to collect additional data.

The Kaggle page gives traffic / travel-time data from LTA as an example, but notes that real-time collection can be substantially more difficult than downloading historical data.

Potential external features may include:

- travel time to CBD / major employment hubs;
- public-transport accessibility;
- traffic conditions;
- additional amenity data;
- macroeconomic or policy indicators.

External data is optional and should only be used when its expected benefit justifies the collection and integration effort.

---

## 3.3 Data Usage Rules / Flexibility

The project intentionally allows substantial flexibility.

### Allowed

- use only a subset of supplied attributes;
- engineer new features;
- choose your own preprocessing strategy;
- choose your own models;
- tune hyperparameters;
- collect additional external data if useful;
- compare multiple modeling approaches.

### Not required

- using every supplied feature;
- trying every possible preprocessing strategy;
- trying every possible model;
- achieving the best Kaggle score.

### Expected

The team should justify:

- why particular features were selected;
- why particular features were excluded;
- how EDA influenced preprocessing;
- why particular models were chosen;
- how hyperparameters were selected;
- why the final approach behaves as observed.

---

## 4. Expected Project Workflow

A strong project should approximately cover the following workflow.

### 4.1 Problem Understanding

- [ ] Clearly define the prediction problem.
- [ ] Identify important stakeholders.
- [ ] Explain why HDB rental prediction is useful.
- [ ] Define meaningful research / analysis questions.
- [ ] Avoid presenting Kaggle ranking as the primary motivation.

### 4.2 Exploratory Data Analysis

- [ ] Inspect dataset shape, schema, and data types.
- [ ] Understand the target distribution.
- [ ] Inspect missing values.
- [ ] Inspect categorical variables.
- [ ] Inspect numeric variables.
- [ ] Inspect temporal trends.
- [ ] Inspect geographic patterns.
- [ ] Identify suspicious records and potential outliers.
- [ ] Examine relationships between candidate features and rent.
- [ ] Use appropriate plots, figures, and tables.

### 4.3 Data Cleaning & Preprocessing

Potential steps include:

- [ ] missing-value handling;
- [ ] duplicate handling;
- [ ] categorical encoding;
- [ ] numeric transformations;
- [ ] date/time feature extraction;
- [ ] outlier treatment;
- [ ] feature scaling where appropriate;
- [ ] geographic feature processing;
- [ ] feature generation;
- [ ] feature reduction / selection;
- [ ] leakage checks.

Every major preprocessing decision should be motivated by the data or experimental results.

### 4.4 Feature Engineering

Possible feature families include:

#### Property features

- flat type;
- floor area;
- flat / block age;
- lease-related features;
- block characteristics.

#### Temporal features

- year;
- month;
- elapsed time;
- market trend variables.

#### Geographic features

- latitude / longitude;
- town;
- distance to MRT;
- distance to shopping malls;
- distance to schools;
- density / count of nearby amenities.

#### Economic features

- stock-market indicators;
- COE prices;
- other externally collected macroeconomic indicators.

> External data is optional, not mandatory.

### 4.5 Modeling

- [ ] Establish at least one sensible baseline.
- [ ] Train appropriate regression methods.
- [ ] Compare alternative models.
- [ ] Motivate each model choice.
- [ ] Use a valid model-selection / validation strategy.
- [ ] Tune important hyperparameters.
- [ ] Record experiments systematically.

For advanced methods, include relevant references in the final report.

### 4.6 Evaluation

- [ ] Use the competition's required metric where applicable.
- [ ] Use a correct experimental setup.
- [ ] Compare models under consistent conditions.
- [ ] Analyze generalization performance.
- [ ] Investigate cases where models perform poorly.
- [ ] Discuss possible causes of errors.
- [ ] Discuss limitations.
- [ ] Suggest future improvements.

---

## 4.7 Kaggle Evaluation & Submission Format

### Metric

Kaggle evaluates submissions using **Root Mean Square Error (RMSE)** between the predicted and true rental rates:

\[
\mathrm{RMSE}
=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)^2
}
\]

where:

- \(y_i\) is the true monthly rent;
- \(\hat{y}_i\) is the predicted monthly rent;
- \(n\) is the number of test samples.

Because RMSE squares the error term, relatively large prediction errors receive disproportionately high penalties. Error analysis should therefore pay attention to high-error / outlier cases.

### Submission format

For every test-set row, predict the `Predicted` value associated with its `Id`.

Required format:

```csv
Id,Predicted
0,2100.0
1,2300.0
2,3600.0
```

### Submission checklist

- [ ] Number of predictions equals number of test rows.
- [ ] Row order matches the test-set order expected by Kaggle.
- [ ] Header is exactly `Id,Predicted`.
- [ ] Predictions are numeric.
- [ ] No missing predictions.
- [ ] Submission CSV can be loaded cleanly before upload.

---

## 5. Progress Report

### Format

- **Slide deck submitted as PDF**
- Approximately **10–15 slides**

### Purpose

The progress report is intended to:

1. allow the teaching team to check whether the project is moving in a sensible direction;
2. encourage teams to begin the project early.

### Suggested content

There is no mandatory structure, but the report should communicate the current state of the project, including topics such as:

- project goals and questions;
- understanding of the task;
- understanding of the dataset;
- initial EDA;
- initial preprocessing decisions;
- initial modeling decisions;
- preliminary results;
- issues / challenges / obstacles;
- planned next steps.

### Final 1–2 slides

The final **1–2 slides should describe the next steps** until project completion.

### Deadline

**Sunday, 4 October 2026**

The source explicitly states that this is **not a hard deadline**, and earlier submissions are encouraged.

### Grading implication

The progress report is **not explicitly graded**, but **not submitting it may negatively affect the final grade**.

### Progress Report Checklist

- [ ] PDF format
- [ ] Approximately 10–15 slides
- [ ] Problem motivation / goals
- [ ] Dataset understanding
- [ ] Initial EDA
- [ ] Initial design decisions
- [ ] Initial results if available
- [ ] Challenges / blockers
- [ ] Last 1–2 slides contain next steps
- [ ] Submit around / before 4 Oct 2026

---

## 6. Final Report

### Format

- **PDF**
- Scientific-paper style
- Maximum **8 pages**

### Page limit

The 8-page limit **includes**:

- main text;
- tables;
- plots;
- figures.

The 8-page limit **excludes**:

- references;
- appendix.

### Appendix

The appendix may contain supplementary material such as:

- additional EDA plots;
- additional evaluation plots;
- supplementary tables;
- workload breakdown.

However:

> The main report should remain readable and fully understandable without relying on the appendix.

### Templates

Official Word and LaTeX templates are provided by the course.

---

## 7. Required Final Report Content

The exact organization is flexible, but the report should cover the following areas.

### 7.1 Team Information

Include:

- [ ] team name;
- [ ] name of every team member;
- [ ] student ID of every team member.

### 7.2 Workload Breakdown

Include an overview of which team member was mainly responsible for each project component.

Possible formats:

- table;
- Gantt chart;
- other clear workload summary.

This may be included in the appendix.

---

### 7.3 Motivation

Explain:

- what problem is being solved;
- why the problem matters;
- who may benefit;
- what questions the project investigates;
- what makes the problem challenging.

Aiming only for a high Kaggle rank is **not sufficient motivation**.

---

### 7.4 Exploratory Data Analysis & Preprocessing

Explain and justify:

- how the team explored and understood the data;
- important observations found during EDA;
- how those observations informed preprocessing.

Potential topics include:

- data reduction;
- transformations;
- missing-value handling;
- outlier handling;
- feature generation;
- categorical data handling;
- geographic preprocessing.

---

### 7.5 Data Mining Methods

Describe:

- which methods were used;
- why those methods were selected;
- how the methods were applied;
- how hyperparameters were chosen.

The report does **not** need to reproduce textbook explanations of standard techniques.

For more advanced models or methods:

- [ ] provide appropriate references.

---

### 7.6 Evaluation & Interpretation

The report should:

- compare the performance of different methods;
- identify which methods performed better in the experiments;
- discuss plausible reasons for performance differences;
- investigate failure cases;
- discuss limitations;
- propose potential future improvements.

Interpretation matters — do not report only metric values.

---

## 8. Final Submission

The final submission must contain:

- [ ] final report as PDF;
- [ ] source code.

Submission is via **Canvas** in a zipped folder.

Instead of directly including source code, the team may provide:

- [ ] a link to a GitHub repository.

### Reproducibility matters

Code quality is explicitly part of grading.

The codebase should be:

- complete;
- well organized;
- documented;
- readable;
- easy for another person to run;
- sufficient to retrace the workflow and reproduce results.

### Final deadline

The provided project specification currently contains a placeholder:

> `Nov XX, 11.59 pm`

Therefore, the exact final-submission date is **not specified in the supplied source** and should be updated once the course announces it.

---

## 9. Grading

| Component | Weight | Main Expectations |
|---|---:|---|
| Methodological Quality | **60%** | Correct and well-motivated preprocessing, visualization, methods, experimental design, evaluation, and discussion |
| Quality of Report | **30%** | Clear, concise, comprehensible scientific presentation; appropriate references; within 8-page limit |
| Reproducibility | **10%** | Complete, readable, documented, organized, reproducible source code |

---

## 10. Methodological Quality — Detailed Requirements

Methodological quality is the most important component of the grade.

### 10.1 Data Preparation

- [ ] preprocessing choices are appropriate;
- [ ] preprocessing follows from EDA where possible;
- [ ] missing values are handled correctly;
- [ ] categorical attributes are handled correctly;
- [ ] transformations are implemented correctly;
- [ ] feature engineering is justified.

### 10.2 Visualization

Use suitable:

- plots;
- figures;
- tables;
- architecture diagrams;
- workflow diagrams.

Visualizations should help communicate results rather than merely decorate the report.

### 10.3 Methods

- [ ] methods are appropriate for the task;
- [ ] methods are motivated;
- [ ] methods are implemented correctly;
- [ ] alternative approaches are discussed;
- [ ] important design decisions are justified.

### 10.4 Evaluation

- [ ] experiment setup is valid;
- [ ] suitable metrics are used;
- [ ] methods are compared fairly;
- [ ] common errors are analyzed;
- [ ] principal limitations are discussed.

---

## 11. Report Quality Requirements

The report should be:

- clear;
- concise;
- logically organized;
- comprehensible;
- technically accurate.

Additional requirements:

- [ ] cite relevant related work;
- [ ] cite advanced methods where appropriate;
- [ ] stay within the 8-page main-report limit;
- [ ] ensure the main report is understandable without the appendix.

---

## 12. Reproducibility Requirements

A reviewer should be able to understand and reproduce the workflow from the submitted repository.

Recommended repository qualities:

- clear directory structure;
- documented environment / dependencies;
- reproducible preprocessing;
- reproducible training;
- reproducible evaluation;
- fixed or documented random seeds where relevant;
- saved configuration where appropriate;
- clear instructions for running the pipeline;
- meaningful filenames;
- readable code;
- limited notebook-only hidden state;
- outputs / models separated from source code where practical.

A repository-level README should ideally explain:

1. environment setup;
2. dataset placement;
3. preprocessing;
4. feature generation;
5. training;
6. evaluation;
7. Kaggle prediction generation.

---

## 13. Team Assessment

By default, all team members receive the same mark.

However, marks may differ if individual contribution levels warrant it.

The course uses **TEAMMATES peer review** to help assess contributions.

### Recommended collaboration practices

- [ ] start early;
- [ ] meet regularly;
- [ ] aim for 1–2 full-team meetings per week where possible;
- [ ] track responsibility for tasks;
- [ ] maintain buffer time for exams and unexpected delays;
- [ ] inform the teaching team early if serious internal conflicts arise.

---

## 14. Suggested Repository Structure

The following structure is **not an official course requirement**, but is recommended to support the reproducibility requirement.

```text
CS5228_data_mining_project/
│
├── README.md
├── project_requirements_readme.md
├── requirements.txt              # or environment.yml / pyproject.toml
│
├── data/
│   ├── raw/
│   ├── external/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline.ipynb
│   └── 04_experiments.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── utils/
│
├── configs/
│
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── outputs/
│   ├── figures/
│   ├── models/
│   ├── predictions/
│   └── experiment_logs/
│
├── reports/
│   ├── progress/
│   └── final/
│
└── docs/
```

---

## 15. Recommended Minimum Experimental Plan

The following is a practical project plan derived from the official requirements rather than an additional course requirement.

### Stage A — Baseline

- [ ] load and validate data;
- [ ] perform initial EDA;
- [ ] create a simple preprocessing pipeline;
- [ ] train a simple regression baseline;
- [ ] establish a validation protocol;
- [ ] generate an initial Kaggle submission.

### Stage B — Strong Tabular Models

- [ ] improve categorical handling;
- [ ] improve temporal features;
- [ ] add property-age / lease features;
- [ ] add block features;
- [ ] compare several suitable regression models;
- [ ] tune major hyperparameters.

### Stage C — Spatial Features

- [ ] incorporate latitude / longitude;
- [ ] compute distances to useful landmarks;
- [ ] create nearby-amenity features;
- [ ] test whether location engineering improves validation performance.

### Stage D — Contextual / Economic Features

- [ ] align economic data by time;
- [ ] test stock / COE indicators;
- [ ] determine whether contextual features add predictive value.

### Stage E — Interpretation

- [ ] feature importance / contribution analysis;
- [ ] segment-wise error analysis;
- [ ] temporal error analysis;
- [ ] geographical error analysis;
- [ ] identify systematic failure cases;
- [ ] document limitations.

### Stage F — Finalization

- [ ] select final model / ensemble based on experiments;
- [ ] retrain using the chosen setup;
- [ ] generate final Kaggle predictions;
- [ ] clean repository;
- [ ] verify reproducibility;
- [ ] produce report figures and tables;
- [ ] complete final report;
- [ ] include workload breakdown;
- [ ] verify page limit.

---

## 16. High-Priority Checklist

Before final submission, verify all of the following.

### Methodology

- [ ] EDA directly informs preprocessing decisions.
- [ ] There is a clear baseline.
- [ ] Multiple sensible methods are compared.
- [ ] Model-selection methodology is defensible.
- [ ] Hyperparameter choices are documented.
- [ ] Evaluation setup avoids obvious leakage.
- [ ] Important design choices are justified.
- [ ] Failure cases are analyzed.
- [ ] Limitations are discussed.

### Report

- [ ] Team name included.
- [ ] All names included.
- [ ] All student IDs included.
- [ ] Motivation goes beyond Kaggle ranking.
- [ ] EDA and preprocessing are connected logically.
- [ ] Methods and hyperparameter selection are described.
- [ ] Different methods are compared.
- [ ] Results are interpreted rather than only listed.
- [ ] Limitations and future work are included.
- [ ] Advanced methods have references.
- [ ] Main paper is at most 8 pages.
- [ ] Main paper is understandable without appendix.
- [ ] Workload breakdown is included.

### Code / Reproducibility

- [ ] Repository structure is understandable.
- [ ] Dependencies are documented.
- [ ] Data paths are configurable / documented.
- [ ] Training can be reproduced.
- [ ] Evaluation can be reproduced.
- [ ] Prediction generation can be reproduced.
- [ ] Code is readable and organized.
- [ ] Important parameters are documented.
- [ ] Randomness is controlled where relevant.

### Submission

- [ ] Final PDF included.
- [ ] Source code included, **or** GitHub repository linked.
- [ ] Submission package opens correctly.
- [ ] Exact final deadline checked on Canvas once announced.



## 16.1 Competition Facts From Kaggle

At the time of the supplied Kaggle page snapshot:

- competition type: private community prediction competition;
- task: HDB rental-rate prediction;
- evaluation metric: RMSE;
- dataset package: 8 CSV files, about 23.6 MB in total;
- Kaggle preview reports 74 columns across all listed files;
- `test.csv`: about 50,000 rows and 10 predictor columns;
- competition does not award Kaggle points or medals.

These values describe the Kaggle page snapshot and should not be treated as methodological requirements.

---

## 17. Key Takeaway

The project should optimize for **methodological quality, interpretation, and reproducibility**, not only leaderboard performance.

A strong submission should make it easy for a reader to answer:

1. **What problem are you solving and why does it matter?**
2. **What did you learn from the data?**
3. **How did those observations shape preprocessing and features?**
4. **Why did you choose these models?**
5. **How were experiments designed and compared?**
6. **Why did some approaches work better than others?**
7. **Where does the final approach fail?**
8. **What are its limitations?**
9. **Can someone else reproduce the result from the submitted code?**
