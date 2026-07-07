import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. Import Sentiment Libraries
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Initialize VADER analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# ----------------------------------------------------
# Helper Functions for Analysis
# ----------------------------------------------------
def analyze_textblob(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0.05:
        return "Positive", score
    elif score < -0.05:
        return "Negative", score
    else:
        return "Neutral", score

def analyze_vader(text):
    score = vader_analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return "Positive", score
    elif score <= -0.05:
        return "Negative", score
    else:
        return "Neutral", score

def get_sentiment(text, engine):
    if engine == "TextBlob":
        return analyze_textblob(text)
    elif engine == "VADER":
        return analyze_vader(text)

# ----------------------------------------------------
# Streamlit App Layout
# ----------------------------------------------------
st.set_page_config(page_title="Multi-Engine Sentiment Analyzer", layout="wide")

st.title("📊 Multi-Engine Sentiment Analysis Dashboard")
st.markdown("Analyze English text sentiment using **TextBlob** or **VADER** models.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Configuration")
engine_choice = st.sidebar.selectbox(
    "Choose Sentiment Engine", 
    ["VADER", "TextBlob"]
)
analysis_mode = st.sidebar.radio("Select Input Mode", ["Single Text Sandbox", "Bulk File Upload"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Engine Quick Facts:**")
st.sidebar.write("- **VADER:** Best for social media text, emojis, and short idioms.")
st.sidebar.write("- **TextBlob:** Fast, rule-based approach for general text.")

# --- MODE 1: SINGLE TEXT SANDBOX ---
if analysis_mode == "Single Text Sandbox":
    st.subheader("📝 Try-It-Yourself Sandbox")
    user_text = st.text_area("Enter your text below:", "I absolutely love using this dashboard! It makes data science so much easier, though setup takes a minute.")
    
    if st.button("Analyze Text", type="primary"):
        if user_text.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            label, score = get_sentiment(user_text, engine_choice)
            
            # Display colored metric card
            if label == "Positive":
                st.success(f"**Result:** {label} (Polarity Score: {score:.2f})")
            elif label == "Negative":
                st.error(f"**Result:** {label} (Polarity Score: {score:.2f})")
            else:
                st.info(f"**Result:** {label} (Polarity Score: {score:.2f})")

# --- MODE 2: BULK FILE UPLOAD ---
else:
    st.subheader("📁 Bulk File Upload")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file containing text data", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Read file safely
        if uploaded_file.name.endswith('.csv'):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)
            
        # Let user choose which column contains the text
        text_column = st.selectbox("Select the column containing the text data:", raw_df.columns)
        
        # Initialize session state to save data across page redraws
        if "processed_df" not in st.session_state:
            st.session_state.processed_df = None
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.processed_df = None
            st.session_state.last_uploaded_file = uploaded_file.name


        # --- OPTIMIZED BATCH PROCESSING WITH PROGRESS BAR ---
        if st.button("Process Dataset", type="primary"):
            df_clean = raw_df.dropna(subset=[text_column]).reset_index(drop=True)
            total_docs = len(df_clean)
            
            if total_docs == 0:
                st.warning("The selected column has no valid text data.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                text_data = df_clean[text_column].astype(str).tolist()
                all_labels, all_scores = [], []
                
                batch_size = 2000
                num_batches = int(np.ceil(total_docs / batch_size))
                
                for i in range(num_batches):
                    batch = text_data[i*batch_size : (i+1)*batch_size]
                    for text in batch:
                        label, score = get_sentiment(text, engine_choice)
                        all_labels.append(label)
                        all_scores.append(score)
                    
                    progress_bar.progress((i + 1) / num_batches)
                    status_text.text(f"Processed {min((i+1)*batch_size, total_docs)} of {total_docs} rows...")
                
                progress_bar.empty()
                status_text.empty()
                
                df_clean['Sentiment_Label'] = all_labels
                df_clean['Sentiment_Score'] = all_scores
                st.session_state.processed_df = df_clean

        # Load metrics and plots only if data is processed
        if st.session_state.processed_df is not None:
            df = st.session_state.processed_df
            total_docs = len(df)
            
            # 1. KPI Metrics Rows
            st.markdown("### 📈 Dataset Highlights")
            kpi1, kpi2, kpi3 = st.columns(3)
            
            avg_score = df['Sentiment_Score'].mean()
            pos_pct = (df['Sentiment_Label'] == 'Positive').sum() / total_docs
            neg_pct = (df['Sentiment_Label'] == 'Negative').sum() / total_docs
            net_sentiment = (pos_pct - neg_pct) * 100

            kpi1.metric("Total Rows Processed", f"{total_docs:,}")
            kpi2.metric("Average Sentiment Score", f"{avg_score:.2f}", help="Scale spans -1.0 (Negative) to +1.0 (Positive)")
            kpi3.metric("Net Sentiment Score", f"{net_sentiment:.1f}%", help="% Positive minus % Negative")

            # 2. Main High-Level Visualizations
            st.markdown("### 📊 Distribution Plots")
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Distribution Pie Chart
                fig_pie = px.pie(df, names='Sentiment_Label', title='Overall Sentiment Breakdown',
                                 color='Sentiment_Label', 
                                 color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c', 'Neutral':'#f1c40f'})
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with chart_col2:
                # Polarity Score Histogram
                fig_hist = px.histogram(df, x='Sentiment_Score', nbins=20, 
                                        title='Detailed Sentiment Polarity Spread',
                                        labels={'Sentiment_Score': 'Polarity Rating (-1 to +1)'},
                                        color_discrete_sequence=['#3498db'])
                st.plotly_chart(fig_hist, use_container_width=True)

            # 3. Text and Topic Insights (Side-by-Side Wordclouds)
            st.markdown("### ☁️ Theme Wordclouds")
            
            pos_words = " ".join(df[df['Sentiment_Label'] == 'Positive'][text_column].astype(str))
            neg_words = " ".join(df[df['Sentiment_Label'] == 'Negative'][text_column].astype(str))
            
            wc_col1, wc_col2 = st.columns(2)
            
            with wc_col1:
                st.write("**Positive Themes**")
                if len(pos_words.strip()) > 0:
                    wc_pos = WordCloud(width=400, height=250, background_color='white', colormap='Greens').generate(pos_words)
                    fig, ax = plt.subplots()
                    ax.imshow(wc_pos, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.info("No positive sentiment text detected to generate wordcloud.")
                    
            with wc_col2:
                st.write("**Negative Themes**")
                if len(neg_words.strip()) > 0:
                    wc_neg = WordCloud(width=400, height=250, background_color='white', colormap='Reds').generate(neg_words)
                    fig, ax = plt.subplots()
                    ax.imshow(wc_neg, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.info("No negative sentiment text detected to generate wordcloud.")

            # 4. Interactive Raw Data Explorer
            st.markdown("### 🔍 Raw Data Audit Trail")
            st.dataframe(df[[text_column, 'Sentiment_Label', 'Sentiment_Score']], use_container_width=True)

    # # ----------------------------------------------------
    # # Streamlit App Layout
    # # ----------------------------------------------------
    # st.set_page_config(page_title="Multi-Engine Sentiment Analyzer", layout="wide")

    # st.title("📊 Multi-Engine Sentiment Analysis Dashboard")
    # st.markdown("Analyze English text sentiment using **TextBlob** or **VADER** models.")

    # # --- SIDEBAR CONTROLS ---
    # st.sidebar.header("⚙️ Configuration")
    # engine_choice = st.sidebar.selectbox(
    #     "Choose Sentiment Engine", 
    #     ["VADER", "TextBlob"]
    # )
    # analysis_mode = st.sidebar.radio("Select Input Mode", ["Single Text Sandbox", "Bulk File Upload"])

    # st.sidebar.markdown("---")
    # st.sidebar.markdown("**Engine Quick Facts:**")
    # st.sidebar.write("- **VADER:** Best for social media text, emojis, and short idioms.")
    # st.sidebar.write("- **TextBlob:** Fast, rule-based approach for general text.")

    # # --- MODE 1: SINGLE TEXT SANDBOX ---
    # if analysis_mode == "Single Text Sandbox":
    #     st.subheader("📝 Try-It-Yourself Sandbox")
    #     user_text = st.text_area("Enter your text below:", "I absolutely love using this dashboard! It makes data science so much easier, though setup takes a minute.")
        
    #     if st.button("Analyze Text", type="primary"):
    #         if user_text.strip() == "":
    #             st.warning("Please enter some text to analyze.")
    #         else:
    #             label, score = get_sentiment(user_text, engine_choice)
                
    #             # Display colored metric card
    #             if label == "Positive":
    #                 st.success(f"**Result:** {label} (Polarity Score: {score:.2f})")
    #             elif label == "Negative":
    #                 st.error(f"**Result:** {label} (Polarity Score: {score:.2f})")
    #             else:
    #                 st.info(f"**Result:** {label} (Polarity Score: {score:.2f})")

    # # --- MODE 2: BULK FILE UPLOAD ---
    # else:
    #     st.subheader("📁 Bulk File Upload")
    #     uploaded_file = st.file_uploader("Upload a CSV or Excel file containing text data", type=["csv", "xlsx"])
        
    #     if uploaded_file is not None:
    #         # Read file safely
    #         if uploaded_file.name.endswith('.csv'):
    #             df = pd.read_csv(uploaded_file)
    #         else:
    #             df = pd.read_excel(uploaded_file)
                
    #         # Let user choose which column contains the text
    #         text_column = st.selectbox("Select the column containing the text data:", df.columns)
            
    #         if st.button("Process Dataset", type="primary"):
    #             with st.spinner(f"Analyzing rows using {engine_choice}... Please wait."):
    #                 # Drop rows with missing text values
    #                 df = df.dropna(subset=[text_column])
                    
    #                 # Apply sentiment engine function
    #                 results = df[text_column].astype(str).apply(lambda x: get_sentiment(x, engine_choice))
    #                 df['Sentiment_Label'] = [r[0] for r in results]
    #                 df['Sentiment_Score'] = [r[1] for r in results]
                
    #             # 1. KPI Metrics Rows
    #             st.markdown("### 📈 Dataset Highlights")
    #             kpi1, kpi2, kpi3 = st.columns(3)
                
    #             total_docs = len(df)
    #             avg_score = df['Sentiment_Score'].mean()
    #             pos_pct = (df['Sentiment_Label'] == 'Positive').sum() / total_docs
    #             neg_pct = (df['Sentiment_Label'] == 'Negative').sum() / total_docs
    #             net_sentiment = (pos_pct - neg_pct) * 100

    #             kpi1.metric("Total Rows Processed", f"{total_docs:,}")
    #             kpi2.metric("Average Sentiment Score", f"{avg_score:.2f}", help="Scale spans -1.0 (Negative) to +1.0 (Positive)")
    #             kpi3.metric("Net Sentiment Score", f"{net_sentiment:.1f}%", help="% Positive minus % Negative")

    #             # 2. Main High-Level Visualizations
    #             st.markdown("### 📊 Distribution Plots")
    #             chart_col1, chart_col2 = st.columns(2)
                
    #             with chart_col1:
    #                 # Distribution Pie Chart
    #                 fig_pie = px.pie(df, names='Sentiment_Label', title='Overall Sentiment Breakdown',
    #                                 color='Sentiment_Label', 
    #                                 color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c', 'Neutral':'#f1c40f'})
    #                 st.plotly_chart(fig_pie, use_container_width=True)
                    
    #             with chart_col2:
    #                 # Polarity Score Histogram
    #                 fig_hist = px.histogram(df, x='Sentiment_Score', nbins=20, 
    #                                         title='Detailed Sentiment Polarity Spread',
    #                                         labels={'Sentiment_Score': 'Polarity Rating (-1 to +1)'},
    #                                         color_discrete_sequence=['#3498db'])
    #                 st.plotly_chart(fig_hist, use_container_width=True)

    #             # 3. Text and Topic Insights (Side-by-Side Wordclouds)
    #             st.markdown("### ☁️ Theme Wordclouds")
                
    #             pos_words = " ".join(df[df['Sentiment_Label'] == 'Positive'][text_column].astype(str))
    #             neg_words = " ".join(df[df['Sentiment_Label'] == 'Negative'][text_column].astype(str))
                
    #             wc_col1, wc_col2 = st.columns(2)
                
    #             with wc_col1:
    #                 st.write("**Positive Themes**")
    #                 if len(pos_words.strip()) > 0:
    #                     wc_pos = WordCloud(width=400, height=250, background_color='white', colormap='Greens').generate(pos_words)
    #                     fig, ax = plt.subplots()
    #                     ax.imshow(wc_pos, interpolation='bilinear')
    #                     ax.axis('off')
    #                     st.pyplot(fig)
    #                 else:
    #                     st.info("No positive sentiment text detected to generate wordcloud.")
                        
    #             with wc_col2:
    #                 st.write("**Negative Themes**")
    #                 if len(neg_words.strip()) > 0:
    #                     wc_neg = WordCloud(width=400, height=250, background_color='white', colormap='Reds').generate(neg_words)
    #                     fig, ax = plt.subplots()
    #                     ax.imshow(wc_neg, interpolation='bilinear')
    #                     ax.axis('off')
    #                     st.pyplot(fig)
    #                 else:
    #                     st.info("No negative sentiment text detected to generate wordcloud.")

    #             # 4. Interactive Raw Data Explorer
    #             st.markdown("### 🔍 Raw Data Audit Trail")
    #             st.dataframe(df[[text_column, 'Sentiment_Label', 'Sentiment_Score']], use_container_width=True)
