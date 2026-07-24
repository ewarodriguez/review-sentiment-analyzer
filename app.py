import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. Import Sentiment Libraries
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
st.set_page_config(page_title="Multi-Engine Review Sentiment Analyzer", layout="wide")

st.title("📊 Multi-Engine Review Sentiment Analyzer")
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
    
    # Initialize session state variable for text sandbox if it doesn't exist
    if "sandbox_text" not in st.session_state:
        st.session_state.sandbox_text = "I absolutely love using this dashboard! It makes data science so much easier, though setup takes a minute."
    
    # Callback function to clear the text input
    def clear_text():
        st.session_state.sandbox_text = ""

    # Bind the text area value to the session state variable
    user_text = st.text_area("Enter your text below:", key="sandbox_text")
    
    # Create action buttons side-by-side using columns
    btn_col1, btn_col2 = st.columns([1, 10])
    
    with btn_col1:
        analyze_clicked = st.button("Analyze Text", type="primary")
    with btn_col2:
        st.button("Clear Text", type="secondary", on_click=clear_text)
        
    if analyze_clicked:
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
        # Read file safely depending on format
        if uploaded_file.name.endswith('.csv'):
            # Fixes DtypeWarning and allows consistent dynamic casting
            raw_df = pd.read_csv(uploaded_file, low_memory=False)
        else:
            raw_df = pd.read_excel(uploaded_file)

        # FIX 1: Turn whitespace-only fields into true NaNs across the entire file
        raw_df = raw_df.replace(r'^\s*$', np.nan, regex=True)
        
        # FIX 2: Drop rows where ALL columns are completely empty/NaN
        raw_df = raw_df.dropna(how='all').reset_index(drop=True)

        # FIX 3: Dynamic type casting to prevent PyArrow and Streamlit dataframe crashes
        for col in raw_df.columns:
            if raw_df[col].dtype == "object":
                # Converts mixed datetimes/numbers/NaNs into pure string data
                raw_df[col] = raw_df[col].fillna("").astype(str)

        # --- NEW: DATA SET PREVIEW MODULE ---
        with st.expander("👀 Preview Uploaded Dataset", expanded=True):
            max_rows = len(raw_df)
            # Default preview to 5 rows, capped between 1 and total available rows
            preview_rows = st.number_input(
                "Rows to preview:", 
                min_value=1, 
                max_value=max_rows, 
                value=min(5, max_rows), 
                step=1
            )
            st.dataframe(raw_df.head(preview_rows), width='stretch')
            st.caption(f"Showing top {preview_rows} of {max_rows:,} total rows.")
            
        # Let user choose which column contains the text
        text_column = st.selectbox("Select the column containing the text data:", raw_df.columns)
        
        # Initialize session state tracking variables safely
        if "processed_df" not in st.session_state:
            st.session_state.processed_df = None
        if "blanks_df" not in st.session_state:
            st.session_state.blanks_df = None
        if "dataset_processed_clicked" not in st.session_state:
            st.session_state.dataset_processed_clicked = False

        # RESET TRIPPERS: Wipe out stale records and turn off active switch when a new file lands
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.processed_df = None
            st.session_state.blanks_df = None
            st.session_state.dataset_processed_clicked = False
            st.session_state.last_uploaded_file = uploaded_file.name

        # # --- OPTIMIZED BATCH PROCESSING WITH BLANK EXTRACTION ---
        # # Evaluate primary button trigger OR preserve open status via the verified click state flag
        # if st.button("Process Dataset", type="primary") or st.session_state.dataset_processed_clicked:
        #     st.session_state.dataset_processed_clicked = True

        #     # FIX: Normalize text and explicitly catch 'nan' strings alongside placeholders
        #     clean_series = raw_df[text_column].astype(str).str.strip().str.lower()

        #     # 1. Identify rows where the text column is blank, NaN, or just whitespace
        #     is_blank_mask = (
        #         raw_df[text_column].isna() | 
        #         (clean_series == "") |
        #         clean_series.isin(["na", "n/a", "null","nan"]) |
        #         clean_series.str.startswith("#")  # Catches #NAME?, #VALUE!, etc.
        #         )
            
        #     # 2. Split into two separate DataFrames
        #     blanks_df = raw_df[is_blank_mask].reset_index(drop=True)
        #     df_clean = raw_df[~is_blank_mask].reset_index(drop=True)
            
        #     # Save the blanks dataframe to session state so you can use it elsewhere
        #     st.session_state.blanks_df = blanks_df
            
        #     total_docs = len(df_clean)
        #     total_blanks = len(blanks_df)
            
        #     if total_docs == 0:
        #         st.warning("The selected column has no valid text data to process.")
        #         if total_blanks > 0:
        #             st.info(f"Found {total_blanks} completely blank/invalid rows.")
        #             st.dataframe(blanks_df, width='stretch')
        #     else:
        #         # Inform the user immediately about the split distribution and add an interactive preview counter
        #         if total_blanks > 0:
        #             st.warning(f"📊 Found {total_blanks} blank/invalid rows. Moving them to a separate isolated dataframe to protect model accuracy.")
        #             with st.expander(f"👀 Quick View Isolated Blanks (Total: {total_blanks})", expanded=False):
        #                 blank_preview_rows = st.number_input(
        #                     "Rows of blank data to preview:", 
        #                     min_value=1, 
        #                     max_value=total_blanks, 
        #                     value=min(10, total_blanks), 
        #                     key="process_blank_preview_counter"
        #                 )
        #                 # FIXED: Reads directly from stored state array to guarantee interface stability
        #                 st.dataframe(st.session_state.blanks_df.head(blank_preview_rows), width='stretch')

        #         progress_bar = st.progress(0)
        #         status_text = st.empty()
                
        #         text_data = df_clean[text_column].astype(str).tolist()
        #         all_labels, all_scores = [], []
                
        #         batch_size = 2000
        #         num_batches = int(np.ceil(total_docs / batch_size))
                
        #         # GUARD RAIL: Run heavy iteration sequence only if cached array evaluation returns clean
        #         if st.session_state.processed_df is None:
        #             for i in range(num_batches):
        #                 batch = text_data[i*batch_size : (i+1)*batch_size]
        #                 for text in batch:
        #                     label, score = get_sentiment(text, engine_choice)
        #                     all_labels.append(label)
        #                     all_scores.append(score)
                        
        #                 progress_bar.progress((i + 1) / num_batches)
        #                 status_text.text(f"Processed {min((i+1)*batch_size, total_docs)} of {total_docs} valid text rows...")
                    
        #             df_clean['Sentiment_Label'] = all_labels
        #             df_clean['Sentiment_Score'] = all_scores
        #             st.session_state.processed_df = df_clean
                
        #         progress_bar.empty()
        #         status_text.empty()

        # --- NEW: CLEAR BULK ANALYSIS CALLBACK ---
        def clear_bulk_analysis():
            st.session_state.processed_df = None
            st.session_state.blanks_df = None
            st.session_state.dataset_processed_clicked = False
            # Clear the number input key if it was created
            if "process_blank_preview_counter" in st.session_state:
                del st.session_state["process_blank_preview_counter"]

        # --- NEW: ACTION BUTTONS IN COLUMNS ---
        bulk_col1, bulk_col2 = st.columns([1, 10])
        
        with bulk_col1:
            process_btn = st.button("Process Dataset", type="primary")
        with bulk_col2:
            st.button("Clear Processing", type="secondary", on_click=clear_bulk_analysis)

        # --- OPTIMIZED BATCH PROCESSING WITH BLANK EXTRACTION ---
        # Evaluate primary button trigger OR preserve open status via the verified click state flag
        if process_btn or st.session_state.dataset_processed_clicked:
            st.session_state.dataset_processed_clicked = True

            # FIX: Normalize text and explicitly catch 'nan' strings alongside placeholders
            clean_series = raw_df[text_column].astype(str).str.strip().str.lower()

            # 1. Identify rows where the text column is blank, NaN, or just whitespace
            is_blank_mask = (
                raw_df[text_column].isna() | 
                (clean_series == "") |
                clean_series.isin(["na", "n/a", "null","nan"]) |
                clean_series.str.startswith("#")  # Catches #NAME?, #VALUE!, etc.
                )
            
            # 2. Split into two separate DataFrames
            blanks_df = raw_df[is_blank_mask].reset_index(drop=True)
            df_clean = raw_df[~is_blank_mask].reset_index(drop=True)
            
            # Save the blanks dataframe to session state so you can use it elsewhere
            st.session_state.blanks_df = blanks_df
            
            total_docs = len(df_clean)
            total_blanks = len(blanks_df)
            
            if total_docs == 0:
                st.warning("The selected column has no valid text data to process.")
                if total_blanks > 0:
                    st.info(f"Found {total_blanks} completely blank/invalid rows.")
                    st.dataframe(blanks_df, width='stretch')
            else:
                # Inform the user immediately about the split distribution and add an interactive preview counter
                if total_blanks > 0:
                    st.warning(f"📊 Found {total_blanks} blank/invalid rows. Moving them to a separate isolated dataframe to protect model accuracy.")
                    with st.expander(f"👀 Quick View Isolated Blanks (Total: {total_blanks})", expanded=False):
                        blank_preview_rows = st.number_input(
                            "Rows of blank data to preview:", 
                            min_value=1, 
                            max_value=total_blanks, 
                            value=min(10, total_blanks), 
                            key="process_blank_preview_counter"
                        )
                        # FIXED: Reads directly from stored state array to guarantee interface stability
                        st.dataframe(st.session_state.blanks_df.head(blank_preview_rows), width='stretch')

                progress_bar = st.progress(0)
                status_text = st.empty()
                
                text_data = df_clean[text_column].astype(str).tolist()
                all_labels, all_scores = [], []
                
                batch_size = 2000
                num_batches = int(np.ceil(total_docs / batch_size))
                
                # GUARD RAIL: Run heavy iteration sequence only if cached array evaluation returns clean
                if st.session_state.processed_df is None:
                    for i in range(num_batches):
                        batch = text_data[i*batch_size : (i+1)*batch_size]
                        for text in batch:
                            label, score = get_sentiment(text, engine_choice)
                            all_labels.append(label)
                            all_scores.append(score)
                        
                        progress_bar.progress((i + 1) / num_batches)
                        status_text.text(f"Processed {min((i+1)*batch_size, total_docs)} of {total_docs} valid text rows...")
                    
                    df_clean['Sentiment_Label'] = all_labels
                    df_clean['Sentiment_Score'] = all_scores
                    st.session_state.processed_df = df_clean
                
                progress_bar.empty()
                status_text.empty()


        # Load metrics, plots, and data explorer only if data is processed
        if st.session_state.processed_df is not None:
            df = st.session_state.processed_df
            total_docs = len(df)
            
            # 1. KPI Metrics Rows
            st.markdown("### 📈 Dataset Highlights")
            
            # Calculate explicit counts for the metrics
            pos_count = (df['Sentiment_Label'] == 'Positive').sum()
            neg_count = (df['Sentiment_Label'] == 'Negative').sum()
            neu_count = (df['Sentiment_Label'] == 'Neutral').sum()

            avg_score = df['Sentiment_Score'].mean()
            pos_pct = (df['Sentiment_Label'] == 'Positive').sum() / total_docs
            neg_pct = (df['Sentiment_Label'] == 'Negative').sum() / total_docs
            net_sentiment = (pos_pct - neg_pct) * 100

            # Explanatory tooltip hover text strings
            avg_help_text = (
                "The mathematical average of all sentiment scores. "
                "The scale spans from -1.0 (strongly negative) to +1.0 (strongly positive). "
                "Scores near 0 indicate neutral text or evenly balanced polarity."
            )
            net_help_text = (
                "Calculated as (% Positive Rows - % Negative Rows). "
                "Ignores neutral expressions completely to highlight whether positive "
                "or negative sentiment dominates. Range: -100% to +100%."
            )

            # Row 1: Volume & Breakdown Counts
            row1_col1, row1_col2, row1_col3 = st.columns(3)
            row1_col1.metric("Total Rows Processed", f"{total_docs:,}")
            row1_col2.metric("Average Sentiment Score", f"{avg_score:.2f}", help=avg_help_text)
            row1_col3.metric("Net Sentiment Score", f"{net_sentiment:.1f}%", help=net_help_text)
            
            # Row 2: Neutral Volume & Core Indices
            row2_col1, row2_col2, row2_col3 = st.columns(3)
            row2_col1.metric("Rows with POSITIVE Sentiment", f"{pos_count:,}")
            row2_col2.metric("Rows with NEGATIVE Sentiment", f"{neg_count:,}")
            row2_col3.metric("Rows with NEUTRAL Sentiment", f"{neu_count:,}")

            # 2. Main High-Level Visualizations
            st.markdown("### 📊 Distribution Plots")
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                fig_pie = px.pie(df, names='Sentiment_Label', title='Overall Sentiment Breakdown',
                                 color='Sentiment_Label', 
                                 color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c', 'Neutral':'#f1c40f'})
                st.plotly_chart(fig_pie, width='stretch')
                
            with chart_col2:
                fig_hist = px.histogram(df, x='Sentiment_Score', nbins=20, 
                                        title='Detailed Sentiment Polarity Spread',
                                        labels={'Sentiment_Score': 'Polarity Rating (-1 to +1)'},
                                        color_discrete_sequence=['#3498db'])
                st.plotly_chart(fig_hist, width='stretch')

            # 3. Text and Topic Insights (Side-by-Side Wordclouds)
            st.markdown("### ☁️ Theme Wordclouds")
            
            pos_words = " ".join(df[df['Sentiment_Label'] == 'Positive'][text_column].astype(str))
            neg_words = " ".join(df[df['Sentiment_Label'] == 'Negative'][text_column].astype(str))
            
            wc_col1, wc_col2 = st.columns(2)

            with wc_col1:
                st.write("**Positive Themes**")
                if len(pos_words.strip()) > 0:
                    wc_pos = WordCloud(width=400, height=250, background_color='white', colormap='Greens')
                    pos_frequencies = wc_pos.process_text(pos_words)
                    
                    if len(pos_frequencies) > 0:
                        wc_pos.generate_from_frequencies(pos_frequencies)
                        fig, ax = plt.subplots()
                        ax.imshow(wc_pos, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.info("No meaningful words left after filtering stop words.")
                else:
                    st.info("No positive words detected.")
                    
            with wc_col2:
                st.write("**Negative Themes**")
                if len(neg_words.strip()) > 0:
                    wc_neg = WordCloud(width=400, height=250, background_color='white', colormap='Reds')
                    neg_frequencies = wc_neg.process_text(neg_words)
                    
                    if len(neg_frequencies) > 0:
                        wc_neg.generate_from_frequencies(neg_frequencies)
                        fig, ax = plt.subplots()
                        ax.imshow(wc_neg, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.info("No meaningful words left after filtering stop words.")
                else:
                    st.info("No negative words detected.")


            # 4. Interactive Filtered Raw Data Explorer
            st.markdown("### 🔍 Raw Data Audit Trail")

            # Interactive Filter Elements
            filter_col1, filter_col2 = st.columns([1, 2])

            with filter_col1:
                selected_labels = st.multiselect(
                    "Filter by Sentiment Label:",
                    options=["Positive", "Neutral", "Negative"],
                    default=["Positive", "Neutral", "Negative"]
                )

            with filter_col2:
                search_query = st.text_input("Search keywords within text:", "")

            # Apply Filters dynamically
            filtered_df = df[df['Sentiment_Label'].isin(selected_labels)]

            if search_query:
                filtered_df = filtered_df[filtered_df[text_column].astype(str).str.contains(search_query, case=False, na=False)]

            st.caption(f"Showing {len(filtered_df):,} of {len(df):,} records")
            st.dataframe(filtered_df[[text_column, 'Sentiment_Label', 'Sentiment_Score']], width='stretch')

            # # 4. Interactive Raw Data Explorer
            # st.markdown("### 🔍 Raw Data Audit Trail")
            # st.dataframe(df[[text_column, 'Sentiment_Label', 'Sentiment_Score']], width='stretch')




