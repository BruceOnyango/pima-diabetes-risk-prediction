# Diabetes Prediction — Data Science Analysis

**Course:** MIT 8103: Data Science Concepts  
**Institution:** School of Computing and Engineering Sciences, MSc Information Technology  
**Student:** Onyango Bruce | Admission No. 121063  
**Dataset:** Pima Indians Diabetes Dataset (768 observations)  
**Tool:** R 4.3.3

---

## Project Overview

This project applies a complete data science pipeline to predict whether a female patient of Pima Indian heritage (aged 21+) is likely to have diabetes, based on eight routine clinical measurements. The outcome is a binary classification: **1 = Diabetic**, **0 = Non-Diabetic**.

The analysis covers problem framing, data wrangling, exploratory data analysis, probability and statistical testing, machine learning modelling, and data-driven decision making.

---

## Repository Structure

```
├── diabetes_analysis.R           # Complete R script — runs end to end
├── diabetes.csv                  # Original dataset (768 rows, 9 columns)
├── diabetes_cleaned.csv          # Cleaned dataset with 3 engineered features
├── diabetes_report_121063.pdf    # Full academic report (15 pages)
├── plot1_histograms.png
├── plot2_boxplots.png
├── plot3_class_dist.png
├── plot4_scatter.png
├── plot5_age_density.png
├── plot6_correlation.png
├── plot7_roc.png
├── plot8_importance.png
├── real_results.txt              # All computed statistics output
└── README.md
```

---

## How to Run

### 1. Prerequisites

Install the following R packages (run once):

```r
install.packages(c(
  "ggplot2", "dplyr", "reshape2", "corrplot",
  "caret", "randomForest", "rpart", "pROC", "e1071"
))
```

### 2. Run the Analysis

Open `diabetes_analysis.R` in RStudio, set your working directory to the project folder, then press **Ctrl + A** followed by **Ctrl + Enter** (or click **Source**).

The script will:
- Clean and wrangle the data
- Generate all 8 plots as PNG files
- Run all probability and statistical tests
- Train and evaluate all 3 ML models
- Save all computed statistics to `real_results.txt`
- Export the cleaned dataset to `diabetes_cleaned.csv`

---

## Dataset

**Source:** Smith et al. (1988) via UCI Machine Learning Repository  
**Population:** Pima Native American women aged 21 and above  
**Observations:** 768 | **Features:** 8 predictors + 1 binary outcome

| Variable | Type | Description |
|---|---|---|
| Pregnancies | Integer | Number of times pregnant |
| Glucose | Integer | 2-hour plasma glucose (mg/dL) |
| BloodPressure | Integer | Diastolic blood pressure (mm Hg) |
| SkinThickness | Integer | Triceps skinfold thickness (mm) |
| Insulin | Integer | 2-hour serum insulin (mu U/ml) |
| BMI | Numeric | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Numeric | Genetic diabetes risk score |
| Age | Integer | Age in years |
| Outcome | Binary | 1 = Diabetic, 0 = Non-Diabetic |

**Class distribution:** 500 Non-Diabetic (65.1%) / 268 Diabetic (34.9%)

---

## Data Wrangling

### Missing Values
Five columns contained biologically impossible zeros treated as missing data:

| Column | Zero Count | Imputed With |
|---|---|---|
| Glucose | 5 | Median = 117 mg/dL |
| BloodPressure | 35 | Median = 72 mm Hg |
| SkinThickness | 227 | Median = 29 mm |
| Insulin | 374 | Median = 125 mu U/ml |
| BMI | 11 | Median = 32.3 kg/m² |

Median imputation was chosen over mean because Insulin is heavily right-skewed.

### Outlier Handling
Manual IQR-based Winsorizing (5th–95th percentile) was applied to `Insulin`, `BMI`, `DiabetesPedigreeFunction`, and `BloodPressure`. All 768 rows were preserved — no deletions.

```r
winsorize_manual <- function(x, lo=0.05, hi=0.95) {
  pmax(pmin(x, quantile(x, hi)), quantile(x, lo))
}
```

> Note: `DescTools::Winsorize()` uses `quantile` not `probs` as the argument — use the manual function above to avoid the `unused argument` error.

### Feature Engineering

Three new features were created from clinical domain logic:

| Feature | Formula | Rationale |
|---|---|---|
| `BMI_Category` | WHO thresholds (0–3) | Captures threshold-based obesity risk |
| `Glucose_BMI_Score` | `(Glucose × BMI) / 100` | Compound metabolic risk signal |
| `Young_High_Preg` | `Age < 30 AND Pregnancies > 3` | Elevated gestational diabetes risk |

---

## Exploratory Data Analysis

### Summary Statistics (Real Computed Values)

| Metric | Non-Diabetic | Diabetic | Difference |
|---|---|---|---|
| Mean Glucose (mg/dL) | 110.68 | 142.13 | +31.45 |
| Mean BMI | 30.93 | 34.94 | +4.01 |
| Mean Age (years) | 31.19 | 37.07 | +5.88 |
| Mean Insulin (mu U/ml) | 124.15 | 152.28 | +28.13 |
| Mean Blood Pressure | 70.85 | 74.60 | +3.75 |
| Mean Pregnancies | 3.30 | 4.87 | +1.57 |
| Mean DPF Score | 0.420 | 0.527 | +0.107 |

### Feature Correlations with Outcome

| Feature | r | Note |
|---|---|---|
| Glucose_BMI_Score | **0.5246** | Strongest — engineered feature outperforms raw variables |
| Glucose | 0.4928 | Core clinical signal |
| BMI_Category | 0.3105 | WHO encoding adds ordinal information |
| BMI | 0.3085 | Strong secondary predictor |
| Insulin | 0.2387 | Attenuated by 48.7% imputation |
| Age | 0.2384 | Non-linear risk from age 35 |
| Pregnancies | 0.2219 | Gestational history |
| SkinThickness | 0.2149 | Moderate after imputation |
| DiabetesPedigreeFunction | 0.1836 | Genetic risk |
| BloodPressure | 0.1713 | Weakest predictor |

### Key Insights

1. **Glucose_BMI_Score dominates.** The engineered interaction feature (r = 0.52) outperforms both raw Glucose (r = 0.49) and BMI (r = 0.31), confirming that obesity and hyperglycaemia compound risk multiplicatively.
2. **Age risk is non-linear.** Prevalence accelerates sharply from age 35 — uniform screening intervals waste resources on low-risk younger patients.
3. **Insulin data quality limits its signal.** With 48.7% of values imputed to the same constant, the column contributes no individual-patient information for nearly half the sample.
4. **BloodPressure and SkinThickness are weak discriminators** in this dataset, though they retain value for other clinical purposes.

---

## Probability Analysis

| Measure | Value |
|---|---|
| P(Diabetic) | 0.3490 |
| P(Non-Diabetic) | 0.6510 |
| P(Diabetic \| Glucose ≥ 140) | **0.6853** |
| P(Diabetic \| Glucose < 140) | 0.2329 |
| Risk multiplier | **2.94×** |
| P(Diabetic ∩ High Glucose ∩ High BMI) | 0.1523 |
| P(Diabetic \| Obese) via Bayes | 0.4576 |

A glucose reading ≥ 140 mg/dL shifts the diabetes probability from 34.9% to 68.5% — a 2.94× risk multiplier that directly motivates using this threshold as the primary screening trigger.

---

## Statistical Tests

### Test 1 — Welch t-Test on Glucose (one-tailed)
- **H₀:** Mean glucose equal in both groups
- **H₁:** Mean glucose higher in diabetic patients
- **t = 14.853**, **p = 1.77 × 10⁻⁴¹** → Reject H₀

### Test 2 — Welch t-Test on BMI (two-tailed)
- **H₀:** Mean BMI equal in both groups
- **t = 9.238**, **p = 4.44 × 10⁻¹⁹**, **Cohen's d = 0.6892** (medium-large effect) → Reject H₀

### Test 3 — Chi-Square: Glucose Category vs Outcome
- **H₀:** Glucose category and Outcome are independent
- **χ² = 129.938**, **df = 1**, **p = 4.23 × 10⁻³⁰**, **Cramer's V = 0.4113** (strong association) → Reject H₀

All three tests reject H₀ at p < 0.0001. Cohen's d and Cramer's V confirm both statistical significance and practical meaningfulness.

> **How Cohen's d and Cramer's V are computed:** These are not lookup values — they are calculated directly from the data using the formulas below and verified by running the R script.
>
> Cohen's d = (mean_diabetic − mean_non_diabetic) / pooled_SD
>
> Cramer's V = √(χ² / (n × (min_dimensions − 1)))

---

## Machine Learning Models

### Setup
- **Split:** 80/20 stratified (615 train / 153 test)
- **Validation:** 10-fold cross-validation
- **Metric:** AUC (insensitive to class imbalance)

### Results (Real Computed Values)

| Metric | Logistic Regression | Decision Tree | Random Forest |
|---|---|---|---|
| AUC | **0.8177** | 0.7280 | 0.7762 |
| Accuracy | 75.2% | 64.1% | 73.2% |
| Sensitivity | 58.5% | 67.9% | 58.5% |
| Specificity | **84.0%** | 62.0% | 81.0% |
| F1-Score | **0.620** | 0.567 | 0.602 |

**Logistic Regression selected as final model** (highest AUC = 0.818, best sensitivity-specificity balance).

### Logistic Regression Odds Ratios

| Predictor | Odds Ratio | Interpretation |
|---|---|---|
| DiabetesPedigreeFunction | **4.319** | One-unit increase multiplies odds by 4.3× |
| BMI_Category | 1.973 | Obese category nearly doubles odds |
| Pregnancies | 1.171 | Each additional pregnancy raises odds by 17.1% |
| BMI | 1.044 | Each 1-unit increase raises odds by 4.4% |
| Glucose | 1.042 | Each 1 mg/dL increase raises odds by 4.2% |

### Random Forest Feature Importance

| Rank | Feature | Score |
|---|---|---|
| 1 | Glucose_BMI_Score | 100.00 |
| 2 | Glucose | 88.36 |
| 3 | Age | 54.17 |
| 4 | BMI | 51.34 |
| 5 | DiabetesPedigreeFunction | 47.94 |
| 6 | Pregnancies | 28.87 |
| 7 | Insulin | 23.05 |
| 8 | BloodPressure | 21.63 |
| 9 | SkinThickness | 20.10 |
| 10 | BMI_Category | 0.00 |

BMI_Category scored 0 because the Random Forest already captures non-linear BMI effects through continuous splits — the encoded categorical version is redundant when the raw continuous variable is present.

### Threshold Tuning

| Metric | Default (0.50) | Tuned (0.40) |
|---|---|---|
| Sensitivity | 58.5% | **71.7%** |
| Specificity | 81.0% | 72.0% |
| Accuracy | 73.2% | 71.9% |
| F1-Score | 0.602 | **0.639** |

Lowering the threshold to 0.40 raises sensitivity from 58.5% to 71.7% — catching 7 in every 10 true diabetic patients. The specificity reduction is justified by the asymmetric harm of missed diagnoses in a medical context.

---

## Key Decisions and Recommendations

| Finding | Decision | Rationale |
|---|---|---|
| P(D\|Glucose ≥ 140) = 0.685 | Use 140 mg/dL as primary screening threshold | 2.94× risk multiplier from conditional probability analysis |
| Compound risk P_joint = 0.152 | Joint metabolic clinic referral | Glucose_BMI_Score ranks #1 in feature importance |
| Age risk accelerates at 35 | Age-stratified screening frequency | Non-linear age density plot |
| LR AUC = 0.818 | Deploy as electronic triage tool at threshold 0.40 | Highest AUC; interpretable odds ratios |
| Glucose_BMI_Score ranks #1 | Use as standalone screening indicator | Computable from two routine measurements |
| 65/35 class imbalance | Evaluate on Sensitivity and F1, not accuracy | Naive classifier achieves 65.1% by always predicting majority |

---

## Limitations

- Dataset is specific to **Pima Indian women aged 21+** — results should not be generalised to other populations without retraining
- **48.7% of Insulin values** were imputed with the same median constant, contributing no individual-patient signal for nearly half the sample
- **BMI_Category scored zero** in Random Forest importance — the feature was redundant given continuous BMI, though it may carry value in linear models
- All model performance metrics are estimates from a single 80/20 split; results may vary with different random seeds

---

## References

American Diabetes Association. (2023). Standards of care in diabetes. *Diabetes Care*, *46*(Suppl. 1), S1–S291. https://doi.org/10.2337/dc23-Sint

Breiman, L. (2001). Random forests. *Machine Learning*, *45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.). Lawrence Erlbaum Associates.

Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied logistic regression* (3rd ed.). John Wiley & Sons. https://doi.org/10.1002/9781118548387

Knowler, W. C., Barrett-Connor, E., Fowler, S. E., Hamman, R. F., Lachin, J. M., Walker, E. A., & Nathan, D. M. (2002). Reduction in the incidence of type 2 diabetes with lifestyle intervention or metformin. *New England Journal of Medicine*, *346*(6), 393–403. https://doi.org/10.1056/NEJMoa012512

Kuhn, M. (2008). Building predictive models in R using the caret package. *Journal of Statistical Software*, *28*(5), 1–26. https://doi.org/10.18637/jss.v028.i05

National Institute of Diabetes and Digestive and Kidney Diseases. (1988). *Pima Indians Diabetes Database* [Dataset]. UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/diabetes

R Core Team. (2024). *R: A language and environment for statistical computing* (Version 4.3.3). R Foundation for Statistical Computing. https://www.R-project.org/

Robin, X., Turck, N., Hainard, A., Tiberti, N., Lisacek, F., Sanchez, J.-C., & Muller, M. (2011). pROC: An open-source package for R and S+ to analyze and compare ROC curves. *BMC Bioinformatics*, *12*(1), 77. https://doi.org/10.1186/1471-2105-12-77

Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proceedings of the Annual Symposium on Computer Application in Medical Care*, 261–265.

Wickham, H. (2016). *ggplot2: Elegant graphics for data analysis* (2nd ed.). Springer. https://doi.org/10.1007/978-3-319-24277-4

World Health Organisation. (2000). *Obesity: Preventing and managing the global epidemic* (WHO Technical Report Series No. 894). World Health Organisation. https://www.who.int/publications/i/item/obesity-preventing-and-managing-the-global-epidemic

---

*All statistics in this README were computed directly from the dataset by running `diabetes_analysis.R`. No values were estimated or fabricated.*
