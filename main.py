import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
import requests
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Groq
from groq import Groq
from dotenv import load_dotenv


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


st.set_page_config(page_title="Smart Data Analyzer", layout="wide")


from streamlit_lottie import st_lottie

def load_lottie(url):
    try:
        return requests.get(url).json()
    except:
        return None

lottie_data = load_lottie("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")

st.markdown("""
<style>


.block-container {
    padding-top: 0rem !important;
}
header, footer, #MainMenu {
    visibility: hidden;
}
section.main > div {
    padding-top: 0rem !important;
}


.stApp {
    background: linear-gradient(135deg, #e0f2fe, #bae6fd);
    color: #0f172a;
}


section[data-testid="stSidebar"] {
    background: #38bdf8;
    color: white;
}


.fade-in {
    animation: fadeIn 1.2s ease-in;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}


.stButton>button {
    background: #0ea5e9;
    color: white;
    border-radius: 10px;
    border: none;
    transition: 0.3s;
}
.stButton>button:hover {
    background: #0284c7;
    transform: scale(1.05);
}


h1, h2, h3 {
    color: #0369a1;
}

.stTextInput input, .stSelectbox div {
    background-color: #e0f2fe;
    color: #0f172a;
}

[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 10px;
}


[data-testid="stMetric"] {
    background: white;
    padding: 10px;
    border-radius: 10px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)


st.markdown('<div class="fade-in"><h1>📊 Smart Data Analysis Tool</h1></div>', unsafe_allow_html=True)

if lottie_data:
    st_lottie(lottie_data, height=200)


st.sidebar.header("⚙️ Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if file is not None:

    
    with st.spinner("Analyzing your data..."):
        time.sleep(1)

    
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.success("✅ File uploaded successfully!")

    
    st.markdown('<div class="fade-in"><h2>📊 Dataset Overview</h2></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    rows_placeholder = col1.empty()
    for i in range(0, df.shape[0]+1, max(1, df.shape[0]//20)):
        rows_placeholder.metric("Rows", i)
        time.sleep(0.01)

    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.dataframe(df.head())


    st.markdown('<div class="fade-in"><h2>🔍 Data Profiling</h2></div>', unsafe_allow_html=True)

    profile = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum(),
        "Unique Values": df.nunique()
    })
    st.dataframe(profile)


    st.markdown('<div class="fade-in"><h2>🧹 Data Cleaning</h2></div>', unsafe_allow_html=True)

    if st.button("Remove Duplicates"):
        df = df.drop_duplicates()
        st.success("Duplicates removed!")

    if st.button("Fill Missing Values"):
        df = df.fillna(df.mean(numeric_only=True))
        st.success("Missing values filled!")

    
    st.markdown('<div class="fade-in"><h2>📈 Visualization</h2></div>', unsafe_allow_html=True)

    chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Histogram", "Box"])
    x_col = st.selectbox("X-axis", df.columns)
    y_col = st.selectbox("Y-axis", df.select_dtypes(include=np.number).columns)

    if chart_type == "Bar":
        fig = px.bar(df, x=x_col, y=y_col)
    elif chart_type == "Line":
        fig = px.line(df, x=x_col, y=y_col)
    elif chart_type == "Scatter":
        fig = px.scatter(df, x=x_col, y=y_col)
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x_col)
    else:
        fig = px.box(df, x=x_col, y=y_col)

    st.plotly_chart(fig, use_container_width=True)

    
    st.markdown('<div class="fade-in"><h3>🔥 Correlation Heatmap</h3></div>', unsafe_allow_html=True)

    corr = df.select_dtypes(include=np.number).corr()
    st.plotly_chart(px.imshow(corr, text_auto=True), use_container_width=True)

    
    st.markdown('<div class="fade-in"><h2>🤖 Auto Insights</h2></div>', unsafe_allow_html=True)

    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        st.write(f"📌 {col}: Mean={df[col].mean():.2f}, Max={df[col].max()}, Min={df[col].min()}")

    
    st.markdown('<div class="fade-in"><h2>🤖 Ask Your Data</h2></div>', unsafe_allow_html=True)

    user_query = st.text_input("Ask anything about your dataset")

    if user_query:
        with st.spinner("Thinking..."):
            prompt = f"""
            You are a data analyst.

            Dataset columns: {list(df.columns)}

            Sample data:
            {df.head().to_string()}

            Question: {user_query}

            Give a clear and short answer.
            """

            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}]
            )

            st.success(response.choices[0].message.content)

    
    st.markdown('<div class="fade-in"><h2>🔮 Prediction</h2></div>', unsafe_allow_html=True)

    target = st.selectbox("Target Column", num_cols)
    features = st.multiselect("Feature Columns", num_cols)

    if features and target:
        X = df[features]
        y = df[target]

        if len(X) > 2:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = LinearRegression()
            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            result_df = pd.DataFrame({"Actual": y_test, "Predicted": preds})
            st.dataframe(result_df)

            st.plotly_chart(
                px.scatter(result_df, x="Actual", y="Predicted", trendline="ols"),
                use_container_width=True
            )

    
    st.markdown('<div class="fade-in"><h2>📥 Download Data</h2></div>', unsafe_allow_html=True)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "cleaned_data.csv", "text/csv")

else:
    st.info("👈 Upload a dataset to get started!")