import streamlit as st
import pandas as pd

# Initialize session state
st.set_page_config(page_title="CTR-Based Ad Optimizer", layout="centered")
st.title("🎯 CTR-Based 'Real-Time' Campaign Optimizer")

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame({
        "Product": ["Credit Card", "Premium credit card", "Loan"],
        "Clicks": [0, 0, 0],
        "Impressions": [-1, 0, 0],
        "Status": ["Active"] * 3,
        "Similar_Ads": ["No"] * 3,
        "CTR": [0] * 3

    })
    st.session_state.total_impressions = 0

# Mapping of similar products (customizable)
similar_map = {
    "Credit Card": "Premium credit card"
}

df = st.session_state.df

reset_df = df.copy()
if st.button("r"):
    df["Impressions"] = [-1, 0, 0]
    df["Clicks"] = 0
    df["Status"] = ["Active"] * 3
    df["CTR"] = [0, 0, 0]

st.markdown("Above is the reset button, equivalent to refreshing the page")

total_impressions = st.session_state.total_impressions
idx = 0
placeholder = st.empty()
placeholder2 = st.empty()

if placeholder.button("✅ Click Main Ad"):
    df.at[idx, "Clicks"] += 1

if placeholder2.button("🙈 Ignore Main Ad"):
    pass

active_products = df[df["Status"] == "Active"]

if not active_products.empty:
    current_product = "Credit Card"
    df.at[idx, "Impressions"] += 1
    st.subheader(f"📢 Main Ad : {current_product}")

    clicks = df.at[idx, "Clicks"]
    impressions = df.at[idx, "Impressions"]
    ctr = clicks / impressions if impressions > 0 else 0

    if ctr < 0.1 and impressions > 10:
        df.at[idx, "Status"] = "Inactive"

        active = df[(df["Status"] == "Active") & (df["Product"] == "Credit Card")]
        if len(active) == 0:
            st.subheader("Customer not interested in credit card, stopped ads")
            placeholder.empty()
            placeholder2.empty()

        else:
            prod = active.loc[active["CTR"].idxmax()]
            current_product = prod["Product"]

    elif ctr > 0.4 and impressions > 5:
        df.at[idx, "Similar_Ads"] = "Yes"
        similar_product = similar_map[current_product]
        st.markdown("#### 🎯 Because you engaged, check this out too:")
        if st.button(f"✅ Click Ad 2: {similar_product}"):
            df.at[idx, "Impressions"] -= 1
            # Add or update similar product row
            if similar_product in df["Product"].values:
                pcc_idx = df[df["Product"] == similar_product].index[0]
                df.at[pcc_idx, "Clicks"] += 1
                df.at[pcc_idx, "Impressions"] += 1

        if st.button(f"🙈 Ignore Ad 2: {similar_product}"):
            df.at[idx, "Impressions"] -= 1
            if similar_product in df["Product"].values:
                pcc_idx = df[df["Product"] == similar_product].index[0]
                df.at[pcc_idx, "Impressions"] += 1

        pcc_idx = df[df["Product"] == "Premium credit card"].index[0]
        pcc_ctr = df.at[pcc_idx, "CTR"]
        pcc_impressions = df.at[pcc_idx, "Impressions"]
        if pcc_ctr > 0.4 and pcc_impressions > 10:
            st.markdown(
                "Since you are interested in premium credit cards, here is some information on our private banking services...")
            st.markdown(
                '<span style="text-decoration: underline; color: blue; cursor: pointer;">View our private banking services</span>',
                unsafe_allow_html=True)

    if ctr > 0.6 and impressions > 10:
        st.markdown("#### Due to further engagement, here are more relevant products:")
        loan_idx = df[df["Product"] == "Loan"].index[0]
        if st.button(f"✅ Click ad 3: Loan"):
            df.at[idx, "Impressions"] -= 1
            df.at[loan_idx, "Clicks"] += 1
            df.at[loan_idx, "Impressions"] += 1
        if st.button(f"🙈 Ignore ad 3: Loan"):
            df.at[idx, "Impressions"] -= 1
            df.at[loan_idx, "Impressions"] += 1

        loan_ctr = df.at[loan_idx, "CTR"]
        loan_impressions = df.at[loan_idx, "Impressions"]
        if loan_ctr > 0.4 and loan_impressions > 10:
            st.markdown("Since you are interested in loans, here is some information on our savings accounts...")
            st.markdown(
                '<span style="text-decoration: underline; color: blue; cursor: pointer;">View our savings account benefits</span>',
                unsafe_allow_html=True)

# Show updated CTRs
st.subheader("📊 Ad Campaign Overview")
df["CTR"] = df["Clicks"] / df["Impressions"].replace(0, 1)
st.dataframe(df.style.format({"CTR": "{:.2%}"}))
