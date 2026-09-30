"""
E-Commerce Product Return Prediction - Streamlit Application
============================================================
An enterprise-grade reverse logistics and risk-scoring dashboard designed to:
- Predict order-level return probabilities
- Mitigate reverse logistics expenses
- Provide category-level and discount-tier return intelligence
- Enforce precision-first decision making to protect genuine customer loyalty
"""

import os
import warnings
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress harmless sklearn unpickling warnings when running across different Python/sklearn versions
warnings.filterwarnings("ignore", module="sklearn")

# ---------------------------------------------------------------------------
# Page Configuration & Modern Theme Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Product Return Prediction",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics, glassmorphism cards, and badge accents
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: #FFFFFF;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #F8FAFC;
    }
    .main-header p {
        margin-top: 0.5rem;
        font-size: 1.05rem;
        color: #94A3B8;
        line-height: 1.5;
    }
    
    .card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 1.5rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.5rem;
    }
    
    /* Prediction High Risk Card */
    .risk-high-card {
        background: linear-gradient(135deg, #FEF2F2 0%, #FFF1F2 100%);
        border: 2px solid #EF4444;
        border-radius: 16px;
        padding: 1.75rem;
        color: #991B1B;
        box-shadow: 0 8px 20px rgba(239, 68, 68, 0.12);
        margin-top: 1.5rem;
    }
    
    /* Prediction Low Risk Card */
    .risk-low-card {
        background: linear-gradient(135deg, #F0FDF4 0%, #ECFDF5 100%);
        border: 2px solid #10B981;
        border-radius: 16px;
        padding: 1.75rem;
        color: #065F46;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.12);
        margin-top: 1.5rem;
    }

    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.85rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .badge-high {
        background-color: #EF4444;
        color: #FFFFFF;
    }
    .badge-low {
        background-color: #10B981;
        color: #FFFFFF;
    }
    
    .metric-bubble {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(0, 0, 0, 0.05);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    .sidebar-warning {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 6px;
        color: #92400E;
        font-size: 0.88rem;
        line-height: 1.45;
        margin-top: 1rem;
    }

    .sidebar-metric-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }

    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.65rem 1.75rem;
        border-radius: 10px;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3);
        transition: all 0.2s ease-in-out;
        width: 100%;
        font-size: 1.05rem;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.45);
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = "models/return_prediction_model.pkl"
DATA_PATH = "data/ecommerce_returns_dataset.csv"
COMPARISON_PATH = "outputs/model_comparison.csv"
IMPORTANCE_PATH = "outputs/feature_importance.csv"


# ---------------------------------------------------------------------------
# Data & Model Loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing model pipeline...")
def load_model():
    """Load pre-trained machine learning pipeline and metadata.
    Auto-trains and compiles directly in the current runtime if the pickle
    is missing or incompatible with the host's scikit-learn version."""
    if os.path.exists(MODEL_PATH):
        try:
            pipeline = joblib.load(MODEL_PATH)
            # Verify inference to ensure complete class & schema compatibility
            sample_check = pd.DataFrame(
                [
                    {
                        "product_category": "Books",
                        "product_price_inr": 500,
                        "discount_percent": 10.0,
                        "delivery_duration_days": 3.0,
                        "customer_purchase_history": 2.0,
                        "payment_method": "UPI",
                    }
                ]
            )
            _ = pipeline.predict(sample_check)
            return pipeline
        except Exception:
            # Caught scikit-learn version mismatch or incompatible pickle
            pass

    # Self-healing fallback: Fast train champion pipeline directly inside current environment (~0.05s)
    if os.path.exists(DATA_PATH):
        try:
            from train_model import prepare_data, train_champion_pipeline
            X, y = prepare_data(DATA_PATH)
            pipeline = train_champion_pipeline(X, y)
            return pipeline
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def load_dataset():
    """Load benchmark dataset for exploratory insights."""
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_comparison_metrics():
    """Load candidate model benchmarking results."""
    if not os.path.exists(COMPARISON_PATH):
        return None
    return pd.read_csv(COMPARISON_PATH)


@st.cache_data(show_spinner=False)
def load_feature_importance():
    """Load feature importance dataset."""
    if not os.path.exists(IMPORTANCE_PATH):
        return None
    return pd.read_csv(IMPORTANCE_PATH)


pipeline = load_model()
df_data = load_dataset()
comparison_df = load_comparison_metrics()
importance_df = load_feature_importance()

# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>📦 E-Commerce Product Return Prediction</h1>
        <p>
            Enterprise Decision Support System designed for <strong>Reverse Logistics Optimization</strong>,
            <strong>Category-Level Risk Profiling</strong>, and <strong>Precision-Guarded Customer Interventions</strong>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Missing model check
if pipeline is None:
    st.error(
        """
        ⚠️ **Trained Model Artifact Missing!**
        
        The model file `models/return_prediction_model.pkl` could not be located,
        and `data/ecommerce_returns_dataset.csv` was not found to auto-train.
        
        Please run the training pipeline first:
        ```bash
        python train_model.py
        ```
        Once completed, refresh this page to begin predicting.
        """
    )
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar: Model Metrics & Precision Governance
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🎯 Decision Governance")

    # Extract model metadata
    model_name = getattr(pipeline, "model_name_", "Logistic Regression")
    metrics = getattr(pipeline, "metrics_", {})

    st.markdown(f"**Champion Model:** `{model_name}`")

    if metrics:
        st.markdown("### Key Evaluation Metrics")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(
                "CV Precision",
                f"{metrics.get('CV_Precision_Mean', 0.5476) * 100:.1f}%",
                help="Mean cross-validation precision across 5 stratified folds",
            )
            st.metric(
                "Test Accuracy",
                f"{metrics.get('Accuracy', 0.7867) * 100:.1f}%",
            )
        with col_m2:
            st.metric(
                "CV F1-Score",
                f"{metrics.get('CV_F1_Mean', 0.1156):.3f}",
            )
            st.metric(
                "CV Stability (σ)",
                f"{metrics.get('CV_Precision_Std', 0.0912):.3f}",
            )

    st.markdown("---")
    st.subheader("💡 Why Precision-First?")
    st.markdown(
        """
        In reverse logistics and anti-return operations, flagging an order as a likely return initiates
        preventive friction (e.g. restrictive return policies, manual verification, or withheld promotional incentives).
        
        **The Business Rationale:**
        Prioritizing **Precision** ensures that when the system alerts on a high-risk return, that prediction is accurate.
        """
    )

    st.markdown(
        """
        <div class="sidebar-warning">
            <strong>⚠️ Commercial False Positive Warning:</strong><br>
            A False Positive wrongly penalizes a genuine, loyal customer with withheld discounts,
            unwarranted order scrutiny, slower dispatch, reduced shopping trust, and permanent cart abandonment.
            High precision safeguards genuine customer lifetime value (LTV).
        </div>
        """,
        unsafe_allow_html=True,
    )

    if comparison_df is not None:
        with st.expander("📊 All Model Benchmark Scores", expanded=False):
            st.dataframe(
                comparison_df[
                    ["Model", "CV_Precision_Mean", "Accuracy", "F1_Score"]
                ].rename(
                    columns={
                        "CV_Precision_Mean": "CV Precision",
                        "F1_Score": "F1",
                    }
                ),
                hide_index=True,
            )

    st.markdown("---")
    if st.button("🔄 Re-benchmark & Train Models", use_container_width=True):
        with st.spinner("Retraining and compiling pipeline..."):
            st.cache_resource.clear()
            st.cache_data.clear()
            from train_model import run_training_pipeline
            run_training_pipeline(DATA_PATH)
        st.success("Pipeline refreshed successfully!")
        st.rerun()


# ---------------------------------------------------------------------------
# Main Layout: Order Risk Predictor
# ---------------------------------------------------------------------------
st.subheader("🛒 Live Order Risk Assessment")
st.markdown("Configure the order parameters below to evaluate the return probability before fulfillment.")

with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        product_category = st.selectbox(
            "Product Category",
            options=[
                "Electronics",
                "Fashion",
                "Books",
                "Home & Kitchen",
                "Beauty",
                "Grocery",
                "Sports",
            ],
            index=1,
            help="Category of the ordered merchandise.",
        )
        product_price_inr = st.number_input(
            "Product Price (INR ₹)",
            min_value=50,
            max_value=15000,
            value=1850,
            step=50,
            help="Retail price of the product in Indian Rupees.",
        )

    with col2:
        discount_percent = st.slider(
            "Discount Percentage (%)",
            min_value=0.0,
            max_value=70.0,
            value=35.0,
            step=0.5,
            help="Promotional discount applied to this item.",
        )
        delivery_duration_days = st.slider(
            "Estimated Delivery Duration (Days)",
            min_value=1.0,
            max_value=15.0,
            value=5.0,
            step=0.5,
            help="Total anticipated transit days from warehouse to doorstep.",
        )

    with col3:
        customer_purchase_history = st.number_input(
            "Customer Purchase History (Orders)",
            min_value=0,
            max_value=30,
            value=1,
            step=1,
            help="Total previous successful orders completed by this customer.",
        )
        payment_method = st.selectbox(
            "Payment Method",
            options=[
                "Cash on Delivery",
                "Credit Card",
                "Debit Card",
                "UPI",
                "Wallet",
            ],
            index=0,
            help="Payment instrument selected by the buyer at checkout.",
        )

# Predict Button
predict_btn = st.button("🔮 Predict Return Likelihood", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction Inference & Explanation Engine
# ---------------------------------------------------------------------------
if predict_btn:
    input_data = pd.DataFrame(
        [
            {
                "product_category": product_category,
                "product_price_inr": product_price_inr,
                "discount_percent": float(discount_percent),
                "delivery_duration_days": float(delivery_duration_days),
                "customer_purchase_history": float(customer_purchase_history),
                "payment_method": payment_method,
            }
        ]
    )

    try:
        # Run inference through complete pipeline
        pred_label = pipeline.predict(input_data)[0]
        pred_proba = pipeline.predict_proba(input_data)[0, 1]
        return_pct = pred_proba * 100.0

        # Classification result
        # Baseline dataset return rate is 19.3%; standard model threshold is 0.50
        is_high_risk = pred_label == 1 or pred_proba >= 0.50
        likelihood_text = "High" if is_high_risk else "Low"
        result_text = "Likely Returned" if is_high_risk else "Not Likely Returned"

        # Generate contextual driver explanations
        drivers = []

        # Discount influence
        if discount_percent >= 30.0:
            drivers.append(
                f"**Heavily Discounted ({discount_percent:.1f}%)**: In the benchmark data, discounts > 30% yield an elevated 27%-43% return rate due to impulse purchases."
            )
        elif discount_percent < 15.0:
            drivers.append(
                f"**Low Discount ({discount_percent:.1f}%)**: Orders with minimal discounts have a benchmark return rate under 11.5%, signaling high purchase intent."
            )

        # Payment method influence
        if payment_method == "Cash on Delivery":
            drivers.append(
                "**Cash on Delivery (COD)**: Carries the highest baseline return rate (26.7%) due to zero upfront financial commitment."
            )
        elif payment_method in ["Debit Card", "UPI"]:
            drivers.append(
                f"**Prepaid ({payment_method})**: Prepaid orders correlate strongly with lower return probability (14.0% - 17.5%)."
            )

        # Category influence
        if product_category == "Fashion":
            drivers.append(
                "**Fashion Category**: Highest risk sector (29.4% return rate) driven by fit, sizing, and tactile aesthetic expectations."
            )
        elif product_category in ["Books", "Grocery"]:
            drivers.append(
                f"**{product_category} Category**: Low return propensity sector (< 10% returns) with predictable product specifications."
            )
        elif product_category == "Electronics":
            drivers.append(
                "**Electronics Category**: Moderate-to-high risk (20.3% returns), often influenced by buyer remorse or technical complexity."
            )

        # Purchase history influence
        if customer_purchase_history <= 1:
            drivers.append(
                f"**New or Inactive Customer ({customer_purchase_history} past orders)**: Accounts with limited purchase history exhibit higher return variance."
            )
        elif customer_purchase_history >= 4:
            drivers.append(
                f"**Loyal Shopper ({customer_purchase_history} past orders)**: High past order frequency strongly suppresses return risk."
            )

        # Delivery duration influence
        if delivery_duration_days >= 6.0:
            drivers.append(
                f"**Extended Delivery Duration ({delivery_duration_days:.1f} days)**: Longer delivery transit windows increase buyer fatigue and cancellation/return rates."
            )
        elif delivery_duration_days <= 2.5:
            drivers.append(
                f"**Expedited Delivery ({delivery_duration_days:.1f} days)**: Swift transit reduces order cancellation and doorstep return rates."
            )

        # Render Output Card
        if is_high_risk:
            st.markdown(
                f"""
                <div class="risk-high-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                        <h2 style="margin: 0; color: #DC2626; font-size: 1.6rem;">⚠️ High Return Likelihood</h2>
                        <span class="badge-pill badge-high">High Risk</span>
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 500; margin-bottom: 0.5rem;">
                        Predicted Result: <strong>{result_text}</strong>
                    </div>
                    <div style="font-size: 1.05rem; margin-bottom: 1rem;">
                        Estimated Return Probability: <strong style="font-size: 1.35rem; color: #B91C1C;">{return_pct:.1f}%</strong>
                        <span style="color: #6B7280; font-size: 0.9rem;">(Evaluated via <em>{model_name}</em>)</span>
                    </div>
                    <hr style="border: 0; border-top: 1px solid rgba(220, 38, 38, 0.2); margin: 1rem 0;">
                    <div style="font-weight: 600; margin-bottom: 0.5rem; color: #991B1B;">Key Factor Breakdown:</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="risk-low-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                        <h2 style="margin: 0; color: #059669; font-size: 1.6rem;">✅ Low Return Likelihood</h2>
                        <span class="badge-pill badge-low">Low Risk</span>
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 500; margin-bottom: 0.5rem;">
                        Predicted Result: <strong>{result_text}</strong>
                    </div>
                    <div style="font-size: 1.05rem; margin-bottom: 1rem;">
                        Estimated Return Probability: <strong style="font-size: 1.35rem; color: #047857;">{return_pct:.1f}%</strong>
                        <span style="color: #6B7280; font-size: 0.9rem;">(Evaluated via <em>{model_name}</em>)</span>
                    </div>
                    <hr style="border: 0; border-top: 1px solid rgba(5, 150, 105, 0.2); margin: 1rem 0;">
                    <div style="font-weight: 600; margin-bottom: 0.5rem; color: #065F46;">Key Factor Breakdown:</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Render explanatory bullet points
        with st.container():
            for d in drivers:
                st.markdown(f"- {d}")

    except Exception as e:
        st.error(f"Inference execution failed: {e}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Model Insights & Exploratory Analytics Section
# ---------------------------------------------------------------------------
st.subheader("📈 Model Insights & Dataset Intelligence")
st.markdown("Strategic analytics drawn from the historical return patterns across 1,500 e-commerce orders.")

if df_data is not None:
    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Category Return Rates",
            "🏷️ Discount Impact Analysis",
            "🔬 Feature Importance",
        ]
    )

    # Tab 1: Category Return Rates
    with tab1:
        cat_stats = (
            df_data.groupby("product_category")["returned"]
            .agg(Total="count", ReturnRate="mean")
            .reset_index()
        )
        cat_stats["ReturnRatePct"] = cat_stats["ReturnRate"] * 100.0
        cat_stats = cat_stats.sort_values(by="ReturnRatePct", ascending=False)

        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
            colors = [
                "#EF4444"
                if r >= 25
                else "#F59E0B"
                if r >= 18
                else "#3B82F6"
                if r >= 14
                else "#10B981"
                for r in cat_stats["ReturnRatePct"]
            ]
            bars = ax.barh(
                cat_stats["product_category"],
                cat_stats["ReturnRatePct"],
                color=colors,
                height=0.6,
            )
            ax.invert_yaxis()
            ax.set_xlabel("Historical Return Rate (%)", fontsize=10, fontweight="bold")
            ax.set_title(
                "Return Propensity by Product Category",
                fontsize=12,
                fontweight="bold",
                pad=12,
            )
            ax.grid(axis="x", linestyle="--", alpha=0.3)

            for bar in bars:
                w = bar.get_width()
                ax.text(
                    w + 0.6,
                    bar.get_y() + bar.get_height() / 2,
                    f"{w:.1f}%",
                    ha="left",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="#1F2937",
                )
            ax.set_xlim(0, max(cat_stats["ReturnRatePct"]) + 6)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_c2:
            st.markdown("#### Category Risk Ranking")
            st.dataframe(
                cat_stats.rename(
                    columns={
                        "product_category": "Category",
                        "Total": "Order Volume",
                        "ReturnRatePct": "Return Rate (%)",
                    }
                )[["Category", "Order Volume", "Return Rate (%)"]].style.format(
                    {"Return Rate (%)": "{:.1f}%"}
                ),
                hide_index=True,
            )
            st.info(
                "💡 **Logistics Takeaway:** Fashion items exhibit a 29.4% return rate—more than triple that of Books (8.7%) and Grocery (9.6%). Reverse inventory capacity should be allocated accordingly."
            )

    # Tab 2: Discount Impact Analysis & Explicit Statement
    with tab2:
        bins = [0, 15, 30, 45, 100]
        labels = [
            "0-15% (Low Discount)",
            "15-30% (Moderate Discount)",
            "30-45% (High Discount)",
            "45%+ (Deep Clearance)",
        ]
        df_temp = df_data.copy()
        df_temp["discount_group"] = pd.cut(
            df_temp["discount_percent"],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
        disc_stats = (
            df_temp.groupby("discount_group", observed=False)["returned"]
            .agg(Total="count", ReturnRate="mean")
            .reset_index()
        )
        disc_stats["ReturnRatePct"] = disc_stats["ReturnRate"] * 100.0

        col_d1, col_d2 = st.columns([3, 2])
        with col_d1:
            fig2, ax2 = plt.subplots(figsize=(8, 4.5), dpi=150)
            colors2 = ["#10B981", "#3B82F6", "#F59E0B", "#EF4444"]
            bars2 = ax2.bar(
                disc_stats["discount_group"],
                disc_stats["ReturnRatePct"],
                color=colors2,
                width=0.55,
            )
            ax2.set_ylabel("Return Rate (%)", fontsize=10, fontweight="bold")
            ax2.set_title(
                "Order Return Rate Across Discount Tiers",
                fontsize=12,
                fontweight="bold",
                pad=12,
            )
            ax2.grid(axis="y", linestyle="--", alpha=0.3)
            plt.xticks(rotation=15, ha="right", fontsize=9)

            for bar in bars2:
                h = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 1.0,
                    f"{h:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color="#1F2937",
                )
            ax2.set_ylim(0, max(disc_stats["ReturnRatePct"]) + 8)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        with col_d2:
            st.markdown("#### Discount Tier Benchmark")
            st.dataframe(
                disc_stats.rename(
                    columns={
                        "discount_group": "Discount Tier",
                        "Total": "Order Count",
                        "ReturnRatePct": "Return Rate (%)",
                    }
                )[["Discount Tier", "Order Count", "Return Rate (%)"]].style.format(
                    {"Return Rate (%)": "{:.1f}%"}
                ),
                hide_index=True,
            )

        # Clear Statement callout as explicitly requested
        st.markdown(
            """
            <div style="background-color: #FEF3C7; border-left: 5px solid #D97706; padding: 1.25rem; border-radius: 8px; margin-top: 1rem;">
                <h4 style="color: #92400E; margin-top: 0; margin-bottom: 0.5rem;">
                    📢 Definitive Dataset Finding: Heavily Discounted Orders Suffer Significantly Higher Return Rates
                </h4>
                <p style="color: #78350F; margin: 0; line-height: 1.5; font-size: 0.95rem;">
                    <strong>Yes, heavily discounted orders have unequivocally higher return rates in this dataset.</strong><br>
                    While baseline return rates for low-discount orders (0-15%) sit at only <strong>11.5%</strong>,
                    orders with heavy discounts (30-45%) escalate to <strong>27.3%</strong>, and deep clearance orders (45%+)
                    surge to an alarming <strong>43.3%</strong>—representing nearly a <strong>4x increase in return propensity</strong>.
                    Aggressive discounting encourages speculative and low-commitment purchasing behaviors.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Tab 3: Feature Importance
    with tab3:
        if importance_df is not None:
            st.markdown("#### Relative Feature Influence")
            st.markdown(
                "Features driving return probabilities, derived from the trained champion pipeline."
            )

            fig3, ax3 = plt.subplots(figsize=(9, 5.5), dpi=150)
            top_feats = importance_df.head(10).copy()
            top_feats = top_feats.sort_values(by="importance", ascending=True)

            # Color code based on direction if available
            if "direction" in top_feats.columns:
                bar_colors = [
                    "#EF4444"
                    if "Increases" in d
                    else "#10B981"
                    for d in top_feats["direction"]
                ]
            else:
                bar_colors = ["#3B82F6"] * len(top_feats)

            bars3 = ax3.barh(
                top_feats["feature"],
                top_feats["importance"],
                color=bar_colors,
                height=0.6,
            )
            ax3.set_xlabel("Relative Absolute Importance / Coefficient", fontsize=10)
            ax3.set_title(
                "Top 10 Decision Drivers (Red: Increases Returns | Green: Decreases Returns)",
                fontsize=11,
                fontweight="bold",
                pad=10,
            )
            ax3.grid(axis="x", linestyle="--", alpha=0.3)

            for bar in bars3:
                w = bar.get_width()
                ax3.text(
                    w + 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    f"{w:.3f}",
                    ha="left",
                    va="center",
                    fontsize=8.5,
                    color="#374151",
                )
            ax3.set_xlim(0, max(top_feats["importance"]) * 1.18)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

            with st.expander("Detailed Feature Table"):
                st.dataframe(importance_df, hide_index=True)
        else:
            st.info(
                "Feature importance data not found. Please run `python train_model.py`."
            )

else:
    st.warning("Historical dataset `data/ecommerce_returns_dataset.csv` not found.")
