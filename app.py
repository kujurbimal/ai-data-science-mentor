import streamlit as st
import pandas as pd
import plotly.express as px
import pytesseract
from PIL import Image
from pycaret.regression import setup, compare_models, predict_model
import openai

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="AI Insight Snap", layout="wide")
st.title("📸 AI Insight Snap")
st.caption("Turn Any Data into Instant Insights – Just Snap & Analyze!")

# -------------------------------
# Secure OpenAI API Key Handling
# -------------------------------
st.sidebar.subheader("🔑 OpenAI API Key")

if "openai_key" not in st.session_state:
    st.session_state["openai_key"] = ""

api_key_input = st.sidebar.text_input(
    "Enter your OpenAI API Key",
    value=st.session_state["openai_key"],
    type="password",
    help="Get your key from https://platform.openai.com"
)

if api_key_input:
    st.session_state["openai_key"] = api_key_input
    openai.api_key = st.session_state["openai_key"]

# Show confirmation message
if st.session_state["openai_key"]:
    st.sidebar.success("✅ API Key stored in session")

# -------------------------------
# OCR Section
# -------------------------------
st.subheader("1️⃣ Upload Data Image")
uploaded_image = st.file_uploader(
    "Upload an image of your data (spreadsheet, chart, handwritten table)",
    type=["png", "jpg", "jpeg"]
)

ocr_lang = st.selectbox(
    "OCR Language",
    ["eng", "spa", "fra", "deu", "hin", "jpn", "chi_sim"],
    index=0
)

extracted_text = None
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    extracted_text = pytesseract.image_to_string(image, lang=ocr_lang)
    st.text_area("Extracted Text", extracted_text, height=150)

# -------------------------------
# Dataset Upload Section
# -------------------------------
st.subheader("2️⃣ Or Upload a Dataset File")
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

df = None
if uploaded_file:
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.write("📊 Preview of Uploaded Data:")
    st.dataframe(df.head())

# -------------------------------
# AutoML with PyCaret
# -------------------------------
if df is not None and st.button("🚀 Run AutoML Analysis"):
    st.write("Running AutoML... please wait ⏳")

    # Setup PyCaret regression experiment
    exp = setup(data=df, target=df.columns[-1], silent=True, session_id=42)
    best_model = compare_models()

    st.success("✅ Best Model Selected!")
    st.write(best_model)

    # Predictions
    predictions = predict_model(best_model, data=df)
    st.subheader("📈 Predictions Sample")
    st.dataframe(predictions.head())

    # Plotly visualization
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    if len(numeric_cols) >= 2:
        fig = px.scatter(
            df,
            x=numeric_cols[0],
            y=numeric_cols[1],
            trendline="ols",
            title="Data Visualization"
        )
        st.plotly_chart(fig)

# -------------------------------
# AI Insights with OpenAI
# -------------------------------
if extracted_text and st.session_state["openai_key"]:
    if st.button("✨ Generate Insights with OpenAI"):
        with st.spinner("Analyzing with OpenAI..."):
            prompt = f"Analyze the following data and provide insights:\n\n{extracted_text}"
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst."},
                    {"role": "user", "content": prompt}
                ]
            )
            st.subheader("🤖 AI Insights")
            st.write(response["choices"][0]["message"]["content"])
