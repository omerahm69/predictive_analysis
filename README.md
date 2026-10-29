Heart Failure Clinical Records — Predictive Analysis

Project Overview

This project explores the application of machine learning to clinical records of patients with heart failure.

The primary objective is to investigate relationships between clinical measurements and recorded death events during follow-up, develop a binary classification model, and identify patient subgroups through unsupervised clustering.

The project includes exploratory data analysis (EDA), feature engineering, supervised classification, cluster analysis, model evaluation, and an interactive Streamlit dashboard.

Important: This is an educational and exploratory machine-learning project. Its predictions are not intended for clinical diagnosis, treatment decisions, or validated patient-level risk assessment.

How to Use This Repository

The project is designed to run locally on Windows using Python and a virtual environment.

1. Clone the repository

git clone <repository-url>
cd Predictive_analysis

Replace <repository-url> with the actual GitHub repository URL.

2. Create a virtual environment

py -3.12 -m venv venv

3. Activate the environment

In PowerShell:

.\venv\Scripts\Activate.ps1

Alternatively, use the virtual environment's Python executable directly without activation.

4. Install dependencies

python -m pip install -r requirements.txt

5. Run Jupyter Notebook

Open the notebooks in VS Code and select the project's virtual environment as the Python kernel.

6. Run the Streamlit application

python -m streamlit run app.py

The application will open in your web browser.

Dataset

Source: Heart Failure Clinical Records Dataset — Kaggle

The dataset contains 299 patient records, 12 recorded variables, and one binary outcome variable, DEATH_EVENT.

The variables include:

Variable

Description

age

Patient age

anaemia

Presence of anaemia

creatinine_phosphokinase

Creatine phosphokinase enzyme level

diabetes

Presence of diabetes

ejection_fraction

Percentage of blood ejected during cardiac contraction

high_blood_pressure

Presence of hypertension

platelets

Platelet count

serum_creatinine

Serum creatinine concentration

serum_sodium

Serum sodium concentration

sex

Recorded sex

smoking

Smoking status

time

Follow-up duration

DEATH_EVENT

Death recorded during follow-up (0 = No, 1 = Yes)

For the current classification workflow, time is excluded from the predictor variables because it represents follow-up information that would not be available at baseline.

Project Hypotheses

Hypothesis 1: Serum Creatinine

Hypothesis: Higher serum creatinine levels are associated with an increased frequency of recorded death events among patients with heart failure.

Validation approach:

Conduct Pearson and Spearman correlation analyses.

Compare serum creatinine distributions across outcome groups.

Evaluate its contribution to classification models.

Conclusion: To be documented based on the observed exploratory results and model evaluation. Correlation alone does not establish causation.

Hypothesis 2: Ejection Fraction

Hypothesis: Lower ejection fraction is associated with an increased frequency of recorded death events among patients with heart failure.

Validation approach:

Analyze the relationship between ejection fraction and DEATH_EVENT.

Compare distributions between outcome groups.

Evaluate feature importance and model performance.

Conclusion: To be documented after reviewing the results.

Machine Learning Workflow

Data collection: Import the Heart Failure Clinical Records dataset.

Exploratory data analysis: Investigate distributions, missing values, correlations, outliers, and outcome class balance.

Data splitting: Separate training and test datasets, using stratification where appropriate.

Feature engineering: Apply numerical transformations and scaling using training data only.

Classification: Train and evaluate Logistic Regression and other selected classifiers.

Cluster analysis: Identify groups of patients with similar clinical characteristics.

Model evaluation: Assess classification and clustering performance.

Streamlit application: Present findings, model results, and cluster profiles.

Requirements and Validation

Requirement 1: Data Visualization and Correlation Analysis

Investigate how clinical measurements relate to the recorded death-event outcome.

The analysis includes:

Descriptive statistics and missing-value assessment.

Pearson and Spearman correlations.

Feature distributions by outcome group.

Identification of variables associated with DEATH_EVENT.

Requirement 2: Binary Classification

Objective: Develop a supervised model to classify the recorded DEATH_EVENT outcome.

The target has two classes:

0: No death event recorded during follow-up.

1: Death event recorded during follow-up.

Models: Logistic Regression as a baseline, with additional models considered for comparison.

Evaluation metrics:

ROC-AUC.

Recall (sensitivity) for death events.

Precision.

F1-score.

Confusion matrix.

Probability calibration, where appropriate.

Provisional project success criteria:

Recall for DEATH_EVENT = 1 of at least 80% on the held-out test set.

Precision for DEATH_EVENT = 0 of at least 80% on the held-out test set.

These thresholds are educational project objectives, not clinically established acceptance criteria. Performance should be reported with the limitations associated with the small dataset.

Because follow-up duration varies, this binary classification model does not estimate mortality at a fixed future time horizon.

Requirement 3: Cluster Analysis

Objective: Identify patient subgroups with similar clinical characteristics using unsupervised machine learning.

Method: StandardScaler, Principal Component Analysis (PCA), and K-Means clustering.

Evaluation:

Silhouette score.

Cluster sizes and distributions.

Clinical feature profiles.

Descriptive death-event proportions within clusters.

Provisional success criteria:

Average silhouette score of at least 0.45.

No more than 15 clusters for interpretability.

These are proposed thresholds, not confirmed results. Cluster profiles should be interpreted as exploratory patterns rather than validated clinical risk groups.

Streamlit Dashboard Design

Page 1: Quick Project Summary

Project objectives and dataset overview.

Description of the available clinical variables.

Summary of the machine-learning tasks.

Project limitations.

Page 2: Project Hypothesis and Validation

Serum creatinine hypothesis.

Ejection fraction hypothesis.

Correlation findings.

Supporting visualizations.

Conclusions and limitations.

Page 3: ML — Predict Death Event

Classification model description.

Feature engineering and preprocessing.

Model pipeline.

Feature importance or model coefficients.

Classification report and confusion matrix.

Training and test performance.

Page 4: ML — Cluster Analysis

Clustering methodology.

PCA and K-Means pipeline.

Silhouette analysis.

Cluster distributions.

Distribution of recorded death events across clusters.

Features characterizing each cluster.

Cluster profiles.

Repository Structure

Predictive_analysis/
├── app.py
├── app_pages/
│   ├── multipage.py
│   ├── page_summary.py
│   ├── page_project_hypothesis.py
│   ├── page_predict_Death_event.py
│   └── page_cluster.py
├── src/
│   ├── data_management.py
│   └── machine_learning/
│       └── evaluate_clf.py
├── jupyter_notebooks/
├── outputs/
│   ├── datasets/
│   │   ├── collection/
│   │   └── split/
│   ├── reports/
│   └── ml_pipeline/
│       ├── predict_death_event/
│       │   └── v1/
│       └── cluster_analysis/
│           └── v1/
├── requirements.txt
└── README.md

Limitations

The dataset contains only 299 patient records, which limits the precision and generalizability of model evaluation.

Additional limitations include varying follow-up durations, possible overfitting, limited external validation, and the inability to infer causal relationships from correlations.

A future clinically oriented extension could use survival-analysis methods to account explicitly for follow-up duration and censoring.

Future Improvements

Compare Logistic Regression with Random Forest and XGBoost.

Perform stratified cross-validation and hyperparameter tuning.

Assess calibration and confidence intervals.

Improve interpretability through model coefficients and feature attribution.

Investigate survival-analysis approaches.

Evaluate performance on an independent external dataset.

