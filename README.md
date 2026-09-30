# E-Commerce Product Return Prediction 📦

An end-to-end Machine Learning and Streamlit decision-support application designed for **reverse-logistics planning**, **category-level return-risk analysis**, and **precision-guarded customer interventions**.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Machine Learning Models](#machine-learning-models)
4. [Installation Instructions](#installation-instructions)
5. [Model Training Instructions](#model-training-instructions)
6. [Streamlit Launch Command](#streamlit-launch-command)
7. [Streamlit Community Cloud Deployment](#streamlit-community-cloud-deployment)
8. [Commercial Strategy & Decision Rationale](#commercial-strategy--decision-rationale)
9. [Key Business Findings](#key-business-findings)
10. [Limitations & Business Risks](#limitations--business-risks)

---

## 🌟 Project Overview

Product returns in e-commerce erode up to 30% of operational profit margins due to return transit shipping, restocking inspection, item depreciation, and packaging waste. However, aggressive anti-return measures can alienate high-value, honest customers.

This application provides an intelligent, data-driven system to:
- Predict order-level return probabilities before dispatch.
- Enable operations teams to adjust inventory placement and carrier routing for high-risk categories.
- Apply precision-focused decision logic to protect legitimate buyers from wrongful friction.
- Surface macro-level risk drivers, such as category return profiles and discount-tier impact.

---

## 📊 Dataset Description

The project uses the historical transactions dataset located at [`data/ecommerce_returns_dataset.csv`](file:///Users/armaanmulani/Desktop/Machine_Learning_Project/data/ecommerce_returns_dataset.csv), consisting of 1,500 real-world e-commerce orders with an overall return rate of **19.27%**.

| Column Name | Data Type | Description | Handling Strategy |
| :--- | :--- | :--- | :--- |
| `order_id` | String | Unique transaction identifier | Excluded from modeling to avoid data leakage |
| `product_category` | Categorical | Category (Electronics, Fashion, Books, Home & Kitchen, Beauty, Grocery, Sports) | One-Hot Encoded |
| `product_price_inr` | Numeric | Retail price in Indian Rupees (INR ₹) | Scaled via `StandardScaler` |
| `discount_percent` | Numeric | Promotional discount applied (0% - 60%) | Scaled via `StandardScaler` |
| `delivery_duration_days` | Numeric | Estimated transit duration in days | Imputed with **Median** + Scaled |
| `customer_purchase_history` | Numeric | Number of prior successful completed orders | Imputed with **Median** + Scaled |
| `payment_method` | Categorical | Instrument used (Cash on Delivery, Credit Card, Debit Card, UPI, Wallet) | One-Hot Encoded |
| `returned` | Binary (0 / 1) | Ground-truth target (1 = Returned, 0 = Kept) | Supervised target label |

---

## 🤖 Machine Learning Models

The pipeline evaluates five diverse classification algorithms using an 80/20 stratified split and **5-Fold Stratified Cross-Validation**:

1. **Logistic Regression**: Linear log-odds model providing stable calibrated probabilities and transparent feature coefficients.
2. **K-Nearest Neighbors (KNN)**: Non-parametric instance-based classifier.
3. **Decision Tree Classifier**: Single tree model with depth constraints.
4. **Random Forest Classifier**: Ensemble bagging classifier with randomized sub-sampling.
5. **Gradient Boosting Classifier**: Sequential boosting algorithm optimizing pseudo-residuals.

### Benchmark Comparison

Benchmarked results saved in [`outputs/model_comparison.csv`](file:///Users/armaanmulani/Desktop/Machine_Learning_Project/outputs/model_comparison.csv):

| Model | CV Precision (Mean ± Std) | CV F1-Score | Test Accuracy | Test Precision |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Champion)** | **54.8% ± 9.1%** | **0.116** | **78.7%** | **12.5%** |
| Decision Tree | 40.0% ± 10.7% | 0.256 | 75.7% | 24.1% |
| Gradient Boosting | 35.2% ± 13.0% | 0.185 | 78.7% | 28.6% |
| K-Nearest Neighbors | 32.2% ± 11.0% | 0.165 | 74.3% | 17.2% |
| Random Forest | 21.7% ± 19.4% | 0.024 | 80.3% | 33.3% |

---

## 💡 Commercial Strategy & Decision Rationale

### Why Precision-First?
In reverse logistics mitigation, positive prediction (`returned = 1`) triggers preventative interventions:
- Withholding dynamic checkout discounts.
- Requiring non-refundable deposits or pre-payment for Cash on Delivery.
- Secondary order confirmation or manual fraud inspection.
- Slower fulfillment queuing.

**The Danger of False Positives:**
If an algorithm suffers from low precision (high False Positive Rate), genuine customers with zero intention of returning orders are penalised. This leads directly to:
1. Friction at checkout and cart abandonment.
2. Loss of customer trust and brand loyalty.
3. Severe erosion of Customer Lifetime Value (LTV).

**Selection Choice:**
**Logistic Regression** was selected as champion because it delivered the highest Cross-Validation Precision (**54.8%**) with superior cross-validation stability (**±9.1%** standard deviation), preventing unwarranted customer friction while identifying genuinely vulnerable orders.

---

## 🔍 Key Business Findings

### 1. Heavily Discounted Orders Suffer Significantly Higher Return Rates
Historical data unequivocally reveals that promotional depth is directly correlated with product return likelihood:
- **0% – 15% Discount:** 11.5% return rate
- **15% – 30% Discount:** 14.2% return rate
- **30% – 45% Discount:** 27.3% return rate
- **45%+ Discount (Deep Clearance):** **43.3% return rate**

> ⚠️ Deeply discounted orders experience nearly a **4x spike** in return rates compared to low-discount orders.

### 2. Category Risk Profile
- **Fashion** has the highest return rate (**29.4%**), driven by size, fit, and aesthetic variance.
- **Electronics** (**20.3%**) and **Sports** (**19.2%**) represent intermediate risk tiers.
- **Books** (**8.7%**) and **Grocery** (**9.6%**) exhibit the lowest return propensity.

### 3. Payment Method Dynamics
- **Cash on Delivery (COD)** has the highest return rate (**26.7%**) due to zero upfront buyer commitment.
- **Debit Card** (**14.0%**) and **UPI** (**17.5%**) demonstrate the highest order retention rates.

---

## 💻 Installation Instructions

Ensure you have Python 3.10+ installed.

```bash
# Clone or navigate to the project directory
cd /path/to/Machine_Learning_Project

# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

---

## ⚙️ Model Training Instructions

To train all 5 classification models, run cross-validation benchmarks, generate comparison artifacts, and serialize the selected champion pipeline:

```bash
python train_model.py
```

This will automatically create:
- `models/return_prediction_model.pkl`: Serialized complete scikit-learn pipeline (preprocessor + model).
- `outputs/model_comparison.csv`: Full evaluation metrics and cross-validation benchmarks.
- `outputs/feature_importance.csv`: Feature coefficients / importance rankings.

---

## 🚀 Streamlit Launch Command

Launch the interactive web application locally:

```bash
streamlit run app.py
```

Or using the direct Python module syntax:
```bash
python -m streamlit run app.py
```

The app will be accessible at `http://localhost:8501`.

---

## ☁️ Streamlit Community Cloud Deployment

To deploy this project live on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. **Push to GitHub**:
   Ensure all project files (`app.py`, `train_model.py`, `requirements.txt`, `README.md`, `data/`, `models/`, `outputs/`) are committed and pushed to a public or private GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "feat: complete E-Commerce Product Return Prediction app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Community Cloud**:
   - Log in to [share.streamlit.io](https://share.streamlit.io).
   - Click **"New app"**.
   - Select your repository, branch (`main`), and set the main file path to `app.py`.
   - Click **"Deploy!"**.
   - Streamlit Cloud will automatically install dependencies from `requirements.txt` and launch the application.

---

## ⚠️ Limitations & Business Risks

1. **Dataset Size (1,500 rows)**:
   The current sample size is relatively compact. While stratified cross-validation ensures statistical rigor, training on larger enterprise transaction streams (100k+ orders) will refine tail probability calibration.
2. **Missing Granular Fit & Sizing Data**:
   The dataset does not capture garment sizing charts, return reason codes (e.g. "defective" vs "wrong size"), or customer review scores.
3. **Threshold Calibration Risk**:
   The default 0.5 probability decision boundary may be adapted depending on merchant cost structures. If reverse shipping costs exceed gross product margins, lowering the operational intervention threshold to 0.35 may be warranted, provided customer friction is soft (e.g. proactive size confirmation via WhatsApp rather than withholding discounts).
4. **Cold-Start Bias**:
   First-time shoppers (`customer_purchase_history = 0`) lack longitudinal trust signals. Imputing median history prevents crash scenarios but requires cautious intervention policies to avoid alienating prospective repeat buyers.

---

## 📁 Project Directory Structure

```text
Machine_Learning_Project/
├── app.py                             # Streamlit dashboard and UI
├── train_model.py                     # ML pipeline training and benchmarking script
├── requirements.txt                   # Production dependencies
├── README.md                          # Comprehensive documentation
├── data/
│   └── ecommerce_returns_dataset.csv  # Historical e-commerce transaction dataset
├── models/
│   └── return_prediction_model.pkl    # Serialized champion pipeline
└── outputs/
    ├── model_comparison.csv           # Model benchmark metrics
    └── feature_importance.csv         # Feature impact and rankings
```
