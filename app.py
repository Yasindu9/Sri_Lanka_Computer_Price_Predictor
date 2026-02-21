import streamlit as st
import pandas as pd
import xgboost as xgb
import json
import os

# --- Page Config ---
st.set_page_config(
    page_title="Sri Lanka Computer Price Predictor",
    page_icon="💻",
    layout="centered"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #ff4b4b;
        font-family: 'Inter', sans-serif;
    }
    .sub-text {
        text-align: center;
        color: #6c757d;
        font-family: 'Inter', sans-serif;
        margin-bottom: 2rem;
    }
    .pred-box {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #ff4b4b;
    }
    .pred-value {
        font-size: 36px;
        font-weight: bold;
        color: #00fa9a;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model and Assets ---
@st.cache_resource
def load_assets():
    model_path = "src/models/xgb_model.json"
    cat_path = "src/models/categories.json"
    
    if not os.path.exists(model_path) or not os.path.exists(cat_path):
        return None, None
        
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    with open(cat_path, 'r') as f:
        categories = json.load(f)
        
    return model, categories

model, categories = load_assets()

# --- Main UI ---
st.markdown("<h1 class='main-header'> Sri Lanka Computer Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Based on real market data from ikman.lk</p>", unsafe_allow_html=True)

if model is None:
    st.error("Model files not found. Please ensure the model is trained first.")
    st.stop()

# --- Input Form ---
with st.form("prediction_form"):
    st.subheader("Device Specifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        brand = st.selectbox("Brand", options=categories.get('Brand', []))
        cpu_tier = st.selectbox("Processor Tier", options=categories.get('CPU_Tier', []))
        cpu_gen = st.number_input("Processor Generation ", min_value=1, max_value=20, value=11, step=1)
        ram_gb = st.number_input("RAM (GB)", min_value=2, max_value=128, value=8, step=1)
        
    with col2:
        storage_gb = st.number_input("Storage (GB)", min_value=64, max_value=8192, value=256, step=1)
        storage_type = st.selectbox("Storage Type", options=categories.get('Storage_Type', []))
        has_gpu = st.selectbox("Has Dedicated Graphics (GPU)?", options=["No", "Yes"])
        condition = st.selectbox("Condition", options=categories.get('Condition', []))
        
    st.subheader("Listing Details")
    location = st.selectbox("Location", options=categories.get('Location', []))
        
    submit_button = st.form_submit_button("Predict Price", use_container_width=True)

# --- Prediction Logic ---
if submit_button:
    # Build dictionary
    input_data = {
        'Brand': brand,
        'CPU_Tier': cpu_tier,
        'CPU_Gen': float(cpu_gen) if cpu_gen != 'Unknown' else 'Unknown',
        'RAM_GB': float(ram_gb),
        'Storage_GB': float(storage_gb),
        'Storage_Type': storage_type,
        'Has_Dedicated_GPU': 1 if has_gpu == "Yes" else 0,
        'Condition': condition,
        'Location': location,
        'Is_Member': 0  # Default value since it was removed from frontend
    }
    
    # Needs to match exactly the column types used during training
    input_df = pd.DataFrame([input_data])
    
    # Cast to category types matching training data
    for col, cats in categories.items():
        if col in input_df.columns:
            # We must set categories completely matching the strict list used in training
            input_df[col] = pd.Categorical(input_df[col], categories=cats)
            
    # Also handle CPU_Gen if it was categorical or numerical. 
    # In process_features.py, CPU_Gen was mixed (int or 'Unknown') so it became categorical.
    if 'CPU_Gen' in categories:
         input_df['CPU_Gen'] = pd.Categorical([str(cpu_gen)], categories=categories['CPU_Gen'])
    else:
         # If not categorical, it was float
         input_df['CPU_Gen'] = float(cpu_gen)
         
    try:
        prediction = model.predict(input_df)[0]
        
        # Display Prediction
        st.markdown(f"""
        <div class="pred-box">
            <h3 style='color: white; margin-bottom: 0px;'>Estimated Market Value</h3>
            <div class="pred-value">Rs {prediction:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.success("Prediction generated successfully!")
        
        st.info("""
        **Explanation:**
        The XGBoost model calculated this price based on the historical relationships learned from 5,000+ listings. Features like RAM, Processor Tier, Dedicated GPU presence, and overall Storage size influence this estimate the heaviest.
        """)
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
