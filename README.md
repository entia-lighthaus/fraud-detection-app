# Fraud Detection System

This project is a production-grade machine learning pipeline for detecting fraudulent financial transactions in real time. Built with LightGBM, SHAP explainability, and a hybrid supervised + unsupervised architecture. Deployed as an interactive web application on Streamlit Cloud.

**Live App:** [entia-fraud-detection-app.streamlit.app](https://entia-fraud-detection-app.streamlit.app)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Modelling Pipeline](#modelling-pipeline)
- [Model Performance](#model-performance)
- [Explainability](#explainability)
- [Web Application](#web-application)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Deployment](#deployment)
- [Key Learnings](#key-learnings)
- [Next Steps](#next-steps)

---

## Project Overview

Fraud detection is one of the most challenging machine learning problems in production. Two properties make it uniquely difficult:

**Class imbalance** — fraud accounts for only 1–2% of all transactions. A naive model that predicts "legitimate" for every transaction achieves 98%+ accuracy while catching zero fraud. Accuracy is a useless metric here.

**Adversarial adaptation** — fraudsters actively adapt their behaviour to evade detection. A model that only learns from historical fraud patterns will miss novel attack vectors.

This project addresses both challenges through a hybrid architecture: a supervised LightGBM classifier trained on engineered behavioural features, augmented with an unsupervised Isolation Forest that scores transactions for statistical anomaly — independent of any fraud label.

---

## Architecture

```
Raw Transaction Data
        │
        ▼
┌───────────────────┐
│  Feature          │
│  Engineering      │  ← 36 features across 9 groups
│  (9 categories)   │
└───────────────────┘
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
┌───────────────────┐            ┌──────────────────────┐
│  SMOTE            │            │  Isolation Forest    │
│  Oversampling     │            │  (unsupervised)      │
│  (training only)  │            │  anomaly_score →     │
└───────────────────┘            │  feature #36         │
        │                        └──────────────────────┘
        └──────────────┬───────────────────┘
                       ▼
             ┌──────────────────┐
             │   LightGBM       │
             │   Classifier     │
             └──────────────────┘
                       │
                       ▼
             ┌──────────────────┐
             │  Threshold       │
             │  Tuning          │  ← optimised for business cost
             └──────────────────┘
                       │
                       ▼
             ┌──────────────────┐
             │  SHAP            │
             │  Explanation     │  ← per-transaction reasoning
             └──────────────────┘
```

---

## Dataset

The model was trained on 100,000 synthetically generated transactions designed to replicate real card-not-present fraud patterns observed in African fintech markets.

| Property | Value |
|---|---|
| Total transactions | 100,000 |
| Fraud rate | 1.20% (1,200 transactions) |
| Date range | January 2024 – June 2024 |
| Unique users | 5,000 |
| Unique devices | ~8,500 |

**Train / Validation / Test Split (time-based)**

| Split | Period | Rows | Fraud % |
|---|---|---|---|
| Train | Jan – Apr 2024 | ~67,000 | ~1.2% |
| Validation | May 2024 | ~16,500 | ~1.2% |
| Test | Jun 2024 | ~16,500 | ~1.2% |

>  A time-based split is critical. Random splits allow the model to train on future transactions and predict past ones — this is data leakage and produces inflated, misleading metrics in production.

**Fraud was simulated with realistic patterns:**
- Concentrated in `online` and `atm` merchant categories
- Always card-not-present
- Originates from foreign merchants (US, GB, CN, RU)
- IP addresses from high-risk geographies
- Bimodal amounts: small probing transactions followed by large cashouts

---

## Feature Engineering

Feature engineering is the most impactful step in the entire pipeline. Raw transaction data alone is insufficient — the model needs behavioural signals derived from patterns across time, identity, and network relationships.

36 features were engineered across 9 groups.

---

### Group 1 — Amount Features (2 features)

| Feature | Description |
|---|---|
| `amount` | Raw transaction amount in ₦ |
| `amount_log` | Log-transformed amount: `log(1 + amount)` |

**Insight:** Transaction amounts follow a log-normal distribution — most are small, a few are very large. Log transformation compresses this range and makes the distribution more symmetric, which helps tree-based models find better split points. Fraud shows a bimodal amount pattern (very small probing transactions and very large cashouts) that raw amount partially captures.

---

### Group 2 — Time Features (4 features)

| Feature | Description |
|---|---|
| `hour` | Hour of day (0–23) |
| `dayofweek` | Day of week (0=Monday, 6=Sunday) |
| `is_weekend` | 1 if Saturday or Sunday |
| `is_night` | 1 if hour is between 22:00 and 05:00 |

**Insight:** Fraud does not occur uniformly across time. In our EDA, the peak fraud hour was 16:00 — counterintuitively during business hours, when fraudulent transactions blend into normal traffic volume. Time features give the model temporal context that amount and identity signals cannot provide alone. In production datasets, night-time and weekend transactions often show elevated fraud rates as fraud operations centres operate outside business hours.

---

### Group 3 — Identity Features (3 features)

| Feature | Description |
|---|---|
| `cross_border` | 1 if merchant country ≠ user's home country |
| `ip_country_match` | 1 if IP country matches user's home country |
| `card_not_present` | 1 if physical card was not used |

**Insight:** These three features consistently rank among the strongest fraud signals in card-not-present fraud detection.

- `cross_border`: In our data, domestic fraud rate was 0% while cross-border fraud rate was 2.95%. Every fraud transaction in our dataset originated from a foreign merchant — a near-perfect discriminator.
- `ip_country_match`: When a user's IP is routing through a foreign country despite being a domestic cardholder, this strongly suggests VPN usage, proxy routing, or device compromise.
- `card_not_present`: Fraud overwhelmingly occurs in online or ATM transactions where no physical card verification is possible. This is the highest-volume fraud vector globally.

---

### Group 4 — Device Fingerprinting Features (5 features)

| Feature | Description |
|---|---|
| `is_emulator` | 1 if device ID falls in the emulator range (9000–9500) |
| `is_unknown_browser` | 1 if browser is "Unknown" — indicator of automation |
| `is_new_device` | 1 if this is the first time this user used this device |
| `device_shared_users` | Number of distinct users who have transacted on this device |
| `user_device_count` | Number of distinct devices this user has ever used |

**Insight:** Device fingerprinting is one of the most powerful fraud signals available to fintechs. The reasoning behind each feature:

- `is_emulator`: Emulated devices (Android emulators, virtual machines) are commonly used by fraud automation scripts to bypass physical device requirements. Real users almost never transact from emulators.
- `is_unknown_browser`: Legitimate browsers identify themselves with a user-agent string. Scripts and automated fraud tools often return "Unknown" or an empty user-agent.
- `is_new_device`: A new device appearing on an existing account is a high-risk signal — it may indicate account takeover. First seen in account takeover detection systems at scale.
- `device_shared_users`: A device used by many different users is a fraud ring signal. Legitimate shared devices (family tablets) might be used by 2–3 people. A device shared by 50+ accounts is almost certainly a fraud operation device.
- `user_device_count`: An account suddenly operating from 5+ devices in a short window suggests credential compromise.

---

### Group 5 — Velocity Features (4 features)

| Feature | Description |
|---|---|
| `txn_count_1h` | Number of transactions by this user in the last 1 hour |
| `txn_count_24h` | Number of transactions by this user in the last 24 hours |
| `sum_24h` | Total spend by this user in the last 24 hours |
| `avg_amount_7d` | Average transaction amount by this user over the last 7 days |

**Insight:** Velocity features are the most powerful single group in the feature set. They answer the question: *"Is this transaction unusual relative to this user's recent history?"*

Fraud often manifests as a sudden burst of activity — multiple transactions in a short window as a fraudster attempts to extract value before the account is locked. A user making 15 transactions in one hour when they typically make 1–2 per day is a strong signal.

>  **Critical implementation detail:** Velocity features must be computed using only past transactions (look-back windows). Using any future data — even accidentally — constitutes data leakage and will produce metrics that cannot be reproduced in production.

`avg_amount_7d` serves as the normalisation baseline for ratio features (Group 8).

---

### Group 6 — Sequence & Behavioural Features (5 features)

| Feature | Description |
|---|---|
| `time_since_last_txn` | Seconds since this user's previous transaction |
| `amount_change_pct` | Percentage change in amount vs the previous transaction |
| `is_rapid_succession` | 1 if less than 5 minutes since the last transaction |
| `is_first_txn` | 1 if this is the user's first ever transaction |
| `sudden_large_spend` | 1 if amount is 5x+ the previous transaction amount |

**Insight:** These features approximate what sequence models (LSTMs, Transformers) capture — temporal discontinuities in behaviour — but as engineered signals rather than learned representations.

The core intuition is that **fraud emerges as a change in pattern**, not just as an anomalous single transaction. A ₦500 transaction is not suspicious. A ₦500 transaction coming 30 seconds after a ₦0.50 probing transaction on a new device cross-border is highly suspicious. Sequence features capture this transition signal.

- `sudden_large_spend`: The classic fraud pattern is a small test charge (to verify the card works) followed immediately by a large cashout. This feature directly encodes that pattern.
- `is_first_txn`: First transactions have no history — velocity and sequence features are undefined. This flag signals to the model that it is operating without behavioural context, which itself carries risk.

---

### Group 7 — Graph-Based Features (4 features)

| Feature | Description |
|---|---|
| `ip_user_count` | Number of distinct users associated with this IP country |
| `merchant_txn_density_24h` | Transactions at this merchant category in the last 24 hours |
| `user_merchant_diversity` | Number of distinct merchant categories this user has transacted in |
| `device_fraud_neighbourhood` | Historical fraud rate of transactions on this device |

**Insight:** Graph features treat the transaction network as a connected graph where accounts, devices, IPs, and merchants are nodes. High connectivity — one IP shared by thousands of users, one device touching many accounts — is a hallmark of organised fraud rings.

- `ip_user_count`: A single IP address (or IP country in our implementation) being used by thousands of users suggests shared infrastructure — VPNs, proxies, or botnet exit nodes commonly used in fraud operations.
- `merchant_txn_density_24h`: A sudden spike in transactions at a specific merchant category can indicate a coordinated attack — a fraud ring simultaneously targeting a single merchant or payment processor.
- `device_fraud_neighbourhood`: If a device has been associated with fraud in the past, future transactions on that device carry elevated prior risk. This is a simplified graph centrality measure.
- `user_merchant_diversity`: Low merchant diversity can signal a fraudster using a compromised account for a single specific type of transaction (e.g., gift card purchases only).

> In production systems, full graph features are computed using graph databases (Neo4j) and Graph Neural Networks. Our simplified implementation captures first-order graph signals that provide significant lift without the infrastructure overhead.

---

### Group 8 — Ratio Features (1 feature)

| Feature | Description |
|---|---|
| `amount_vs_avg7d` | Current amount divided by user's 7-day average amount |

**Insight:** A ₦50,000 transaction is not inherently suspicious. For a user whose average transaction is ₦5,000, it is a 10x anomaly. For a corporate account averaging ₦500,000, it is unremarkable. This ratio normalises the amount signal to the individual user's baseline, making it far more discriminative than raw amount alone.

---

### Group 9 — Merchant Category Features (7 features)

| Feature | Description |
|---|---|
| `cat_atm` | 1 if merchant category is ATM |
| `cat_online` | 1 if merchant category is online |
| `cat_dining` | 1 if merchant category is dining |
| `cat_gas` | 1 if merchant category is gas station |
| `cat_grocery` | 1 if merchant category is grocery |
| `cat_retail` | 1 if merchant category is retail |
| `cat_travel` | 1 if merchant category is travel |

**Insight:** EDA revealed a stark contrast in fraud rates by merchant category:

| Category | Fraud Rate |
|---|---|
| ATM | 8.53% |
| Online | 3.56% |
| All others | 0.00% |

ATM and online transactions are overwhelmingly overrepresented in fraud. One-hot encoding preserves this signal and allows the model to learn category-specific fraud rates independently. The absence of fraud in grocery, dining, gas, retail, and travel is itself informative — the model learns that a grocery transaction is a strong negative indicator.

---

### Feature #36 — Isolation Forest Anomaly Score

| Feature | Description |
|---|---|
| `anomaly_score` | Unsupervised anomaly score from Isolation Forest (higher = more suspicious) |

**Insight:** This is the hybrid modelling component. Isolation Forest is trained on transaction features with **no access to fraud labels**. It learns what "normal" looks like by randomly partitioning the feature space — anomalous transactions (including novel fraud patterns not seen in training labels) are isolated in fewer partitions and receive higher anomaly scores.

This score is then injected as feature #36 into LightGBM. The supervised model decides how much weight to assign it alongside the 35 engineered features.

**Why this matters:** LightGBM can only learn fraud patterns that exist in the training labels. If a new type of fraud emerges after training, it may not match any known pattern — but it will likely still look statistically unusual. Isolation Forest catches these novel patterns and surfaces them to LightGBM as a "something looks wrong here" signal.

In our trained model, `anomaly_score` ranked as the **#1 most important feature** by LightGBM's feature importance metric — validating the hybrid approach.

---

## Modelling Pipeline

### Class Imbalance Handling

```
Original training set:
  Legitimate : 78,800  (98.8%)
  Fraud      :    950  ( 1.2%)

After SMOTE (sampling_strategy=0.1):
  Legitimate : 78,800  (91%)
  Fraud      :  7,880  ( 9%)  ← 950 real + 6,930 synthetic
```

SMOTE (Synthetic Minority Oversampling Technique) creates new synthetic fraud examples by interpolating between existing ones in feature space — not by duplicating rows. This gives the model more fraud decision boundary to learn from.

`scale_pos_weight` in LightGBM handles the remaining imbalance after SMOTE.

### Model Configuration

```python
lgb.LGBMClassifier(
    n_estimators      = 1000,     # max trees (early stopping cuts this)
    learning_rate     = 0.05,     # small steps for stability
    num_leaves        = 63,       # controls tree complexity
    max_depth         = 7,
    min_child_samples = 30,       # prevents overfitting on rare fraud
    feature_fraction  = 0.8,      # 80% of features per tree
    bagging_fraction  = 0.8,      # 80% of rows per tree
    bagging_freq      = 5,
    reg_alpha         = 0.1,      # L1 regularisation
    reg_lambda        = 0.1,      # L2 regularisation
    scale_pos_weight  = 9.0,      # remaining imbalance penalty
)
```

Early stopping monitors validation `average_precision` and halts training if no improvement for 50 consecutive rounds — preventing overfitting.

---

## Model Performance

All metrics reported on the held-out test set (June 2024 — never seen during training or validation).

| Metric | Value | Interpretation |
|---|---|---|
| PR-AUC | 0.9856 | Primary metric. Random baseline = 0.012 |
| ROC-AUC | 0.9998 | Near-perfect class separation |
| Lift over random | 83x | 83x better than random guessing |
| Fraud Precision | 93.2% | 93 in 100 flagged transactions are real fraud |
| Fraud Recall | 94.1% | Catches 94 in every 100 fraud cases |
| Fraud F1 | 0.937 | Harmonic mean of precision and recall |

### Threshold Tuning

The default decision threshold of 0.5 is inappropriate for imbalanced data. Three thresholds were evaluated:

| Strategy | Threshold | Precision | Recall | Use Case |
|---|---|---|---|---|
| Best F1 | 0.8023 | 0.932 | 0.941 | General purpose — chosen |
| 85% Recall | 0.9443 | 0.976 | 0.851 | Low false alarm tolerance |
| Min Business Cost | 0.2482 | 0.737 | 1.000 | Maximum fraud capture |

**Business cost model** (FN = ₦50,000 fraud loss, FP = ₦500 investigation cost):
Minimum total cost achieved at threshold 0.2482 = ₦33,500 on the test set.

---

## Explainability

Every prediction is explained using SHAP (SHapley Additive exPlanations). SHAP assigns each feature a contribution value showing how much it pushed the fraud score up or down for that specific transaction.

**Example — Fraud Transaction Explanation:**

```
card_not_present   = 1    → +2.79  ↑ toward fraud
cross_border       = 1    → +2.12  ↑ toward fraud
amount_log         = 0.84 → +1.69  ↑ toward fraud
cat_online         = 1    → +1.58  ↑ toward fraud
ip_country_match   = 0    → +1.16  ↑ toward fraud
device_fraud_neigh = 0.13 → +0.79  ↑ toward fraud
is_unknown_browser = 0    → -0.67  ↓ toward legit
ip_user_count      = 4347 → +0.62  ↑ toward fraud

Final score: 0.9323 → BLOCK
```

In plain English: *"A card-not-present online transaction, cross-border, with a foreign IP routing through shared infrastructure used by 4,347 other accounts, on a device with a 13% fraud history."*

---

## Web Application

The Streamlit app has three pages:

### Page 1 — Transaction Checker
Input transaction details manually and receive:
- Fraud/Legitimate classification banner
- Fraud probability gauge (0–100%)
- Fraud score, threshold, anomaly score, and decision
- Top 10 SHAP drivers with direction and magnitude
- Interactive SHAP bar chart

### Page 2 — Model Performance
- Key metric cards (PR-AUC, ROC-AUC, Recall, Precision)
- Top 20 feature importances (interactive bar chart)
- Model configuration reference table

### Page 3 — Batch Scanner
- Upload a CSV of transactions
- Scores every row and appends `fraud_score`, `fraud_flag`, and `decision`
- Summary metrics (total scanned, flagged fraud, fraud rate)
- Score distribution histogram with threshold line
- Download results as CSV

---

## Project Structure

```
fraud-detection-app/
│
├── app.py                          ← Streamlit application (3 pages)
│
├── fraud_detection_model.ipynb     ← Full training pipeline (Google Colab)
│
├── artifacts/                      ← Serialised model components
│   ├── lgbm_model.txt              ← LightGBM native format (portable)
│   ├── iso_forest.pkl              ← Isolation Forest (protocol=2)
│   ├── feature_cols.json           ← Feature list (version-independent)
│   ├── threshold.json              ← Decision threshold
│   └── imputer_medians.json        ← Median values for NaN imputation
│
├── utils/
│   ├── __init__.py
│   └── feature_engineering.py     ← Feature builder for inference
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup & Installation

**Prerequisites:** Python 3.11+, Git

```bash
# Clone the repository
git clone https://github.com/entia-lighthaus/fraud-detection-app.git
cd fraud-detection-app

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

App will open at `http://localhost:8501`

---

## Deployment

The app is deployed on **Streamlit Community Cloud** — free hosting with automatic redeployment on every GitHub push.

**To redeploy after changes:**
```bash
git add .
git commit -m "describe your change"
git push
```

Streamlit Cloud detects the push and redeploys automatically within ~60 seconds.

---

## Key Learnings

**On class imbalance:** Accuracy is a misleading metric when fraud rate is 1.2%. PR-AUC is the correct primary metric — it measures ranking quality across all thresholds. The model achieved 83x lift over random baseline.

**On threshold tuning:** The default threshold of 0.5 caught zero fraud in one training session because the model correctly learned to output conservative probability scores. Moving to the best-F1 threshold unlocked the full performance the PR-AUC was already showing. Threshold tuning is not optional.

**On version compatibility:** scikit-learn, joblib, and pickle files are Python-version dependent. Saving all artifacts as version-independent formats (LightGBM native `.txt`, plain JSON for metadata) eliminates deployment friction across environments.

**On reproducibility:** A single `np.random.seed()` at the top of a notebook is not sufficient. Each stochastic component (SMOTE, LightGBM, Isolation Forest, SHAP sampling) must receive an explicit `random_state` parameter to ensure results are reproducible across kernel restarts.

**On hybrid modelling:** The Isolation Forest anomaly score ranked as the single most important feature in the trained LightGBM model — demonstrating that unsupervised anomaly detection adds genuine signal beyond what supervised learning alone captures, particularly for novel fraud patterns.

---

## Next Steps

- **Real data:** Train on the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection) on Kaggle — the industry benchmark for this problem
- **Sequence modelling:** Replace rolling velocity features with an LSTM or Transformer that learns temporal patterns end-to-end
- **Graph Neural Networks:** Replace simplified graph features with a full GNN using PyTorch Geometric or DGL
- **Model monitoring:** Track fraud rate, score distributions, and feature drift over time using Evidently AI
- **API deployment:** Wrap the model in a FastAPI endpoint for integration with external payment systems
- **Retraining pipeline:** Schedule monthly model retraining using GitHub Actions as new transaction data accumulates

---

## Built With

- [LightGBM](https://lightgbm.readthedocs.io/) — gradient boosting classifier
- [SHAP](https://shap.readthedocs.io/) — model explainability
- [scikit-learn](https://scikit-learn.org/) — Isolation Forest, SimpleImputer, metrics
- [imbalanced-learn](https://imbalanced-learn.org/) — SMOTE oversampling
- [Streamlit](https://streamlit.io/) — web application framework
- [Plotly](https://plotly.com/) — interactive visualisations
- [Google Colab](https://colab.research.google.com/) — model training environment

---

*Built as a complete end-to-end ML engineering project — from synthetic data generation through feature engineering, model training, explainability, and production deployment.*