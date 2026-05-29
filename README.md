# Bank Marketing Campaign Optimization: Predictive Analytics for Term Deposit Conversions

**By Thierry Laguerre** *Data Science Portfolio | Spring 2026*

---

## 📈 Executive Summary & Business Value

In retail banking, direct marketing campaigns (such as telemarketing) are essential for capital acquisition via term deposits. However, untargeted phone outreach incurs heavy operational overhead, squanders salesperson hours, and hazards customer fatigue.

Using data from a **Portuguese banking institution containing 45,211 engagement instances**, this project transforms an inefficient brute-force marketing strategy into an algorithmic, data-driven system. By analyzing 16 distinct behavioral, demographic, and historical campaign features, this predictive framework maps the exact profile of high-conversion clients.

### Key Business Metrics Addressed:

*   **Customer Acquisition Cost (CAC) Reduction:** By weeding out cold prospects, the sales team focuses strictly on high-probability conversions.
*   **Conversion Rate (ROI) Optimization:** Identifying optimal operational variables (such as ideal contact months, minimal call durations, and cross-sell thresholds) to capture the maximum volume of term deposits.
*   **Resource Allocation:** Eliminating wasted outbound telemarketing bandwidth to significantly optimize operational expenditure (OpEx).

---

## 🛠️ Data Science Stack & Tooling

*   **Runtime Environment:** Jupyter Notebooks / Google Colab
*   **Data Acquisition:** `ucimlrepo` (UCI Machine Learning Repository API Interface)
*   **Data Wrangling & Pipeline Architecture:** `pandas` (Vectorized manipulation, complex data indexing)
*   **Scientific Computing:** `numpy` (High-performance array manipulation)
*   **Data Source Citation:** Moro, S., Rita, P., & Cortez, P. (2014). *A data-driven approach to predict the success of bank telemarketing.* Decision Support Systems.

---

## 📊 Dataset Profile & Data Architecture

The project processes a multivariate architecture consisting of customer demographic data, financial indicators, and historical interaction telemetry.

*   **Volume:** 45,211 records
*   **Dimensionality:** 16 independent variables (Features) + 1 target variable (`y`)
*   **Target Variable (`y`):** Binary classification indicating whether the client subscribed to a term deposit (`yes` / `no`).

### Feature Classification:

| Category                      | Variable Elements                                                                                             |
| :---------------------------- | :------------------------------------------------------------------------------------------------------------ |
| **Demographics / Financials** | `age`, `job`, `marital`, `education`, `balance` (yearly average in Euros), `default` status                   |
| **Liability Profiles**        | `housing` (has housing loan?), `loan` (has personal loan?)                                                     |
| **Current Campaign Metrics**  | `contact` (device type), `day_of_week`, `month`, `duration` (call length in seconds), `campaign` (number of contacts) |
| **Historical Campaign Metrics**| `pdays` (days since previous contact), `previous` (prior contact count), `poutcome` (previous campaign outcome) |

---

## ⚙️ Robust Data Engineering & Preprocessing Pipeline

Real-world datasets contain structural imperfections. This project implements an explicit, reproducible data cleaning framework to protect model downstream performance against missing values and noise.

```python
# 1. Pipeline Consolidation
df = pd.concat([X, y], axis=1)

# 2. Strategic Missing Value Treatment
# Eliminating incomplete demographic profiles where 'job' or 'education' are omitted
df_cleaned = df.dropna(subset=['job', 'education']).copy()

# 3. Categorical Missingness Insulation
# Rather than dropping rich transactional profiles due to sparse structural columns,
# 'poutcome' and 'contact' are preserved through strategic label categorization.
df_cleaned['poutcome'] = df_cleaned['poutcome'].fillna('unknown_outcome')
df_cleaned['contact'] = df_cleaned['contact'].fillna('unknown_contact')
```

**Data Cleansing Rationale:**

*   **Missing Value Analysis:** Identifying structural gaps (e.g., `poutcome` was missing 36,959 records because those customers had never been part of a prior campaign).
*   **Domain Preservation:** Rather than losing over 80% of the dataset by aggressively applying `dropna()` across columns, uncontacted customers are strategically re-labeled to `unknown_outcome`. This allows the machine learning models to differentiate cleanly between brand-new customer acquisitions and legacy customer interactions.

---

## 🔍 Analytical Insights Showcase (EDA)

Through structural distribution analysis (`df_cleaned.describe()`), several critical operational behaviors emerge:

*   **The Call Duration Dilemma:** Call duration (`duration`) is highly correlated with subscription success. However, because call duration is only known *after* a phone call concludes, predictive models must leverage pre-call characteristics (`balance`, `housing`, `job`, `poutcome`) to guide triage teams *before* dialing.
*   **The Law of Diminishing Returns (`campaign`):** Tracking customer fatigue by observing drop-offs in conversion after a threshold number of sequential dials within a single marketing cycle.
*   **Financial Qualification Metrics:** Analyzing how baseline financial stability parameters (average yearly balance in Euros) combined with existing liability exposure (housing/personal loans) influence asset subscription rates.
