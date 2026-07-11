# 📊 Multi-Engine Review Sentiment Analyzer 

An interactive **Streamlit** web application designed to evaluate and audit English text sentiment dynamically using **VADER** and **TextBlob** evaluation engines. This application provides dual computational pipelines for isolated sandboxing and industrial-scale bulk file processing.

---

## Dataset

For testing, you may refer to this sample dataset based on global gaming community telemetry data:

*   **Target Application Context**: User feedback, reviews, and community critiques mapping to the award-winning Action RPG **Elden Ring**.
*   **Multilingual Footprint**: Includes global marketplace feedback tracking across multiple source languages, including **English, Italian (italiano), French (français), Spanish (español), and German (Deutsch)**.
*   **Data Provenance & Engine Retrieval**: All data payloads were fetched via automated streaming requests utilizing the official [Steam Web API GET Reviews Documentation Platform](https://partner.steamgames.com/doc/store/getreviews).

### 📥 Access the Repository Test Set
The comprehensive open-source review matrix can be explored, audited, and downloaded natively from Kaggle:
👉 **[Steam Reviews of Elden Ring on Kaggle](https://www.kaggle.com/datasets/lorenzoshylockl/steam-reviews-of-elden-ring)**

---

## 🚀 Key Features

*   **Dual Sentiment Calculators**: Swap between rule-based lexical models natively on the fly:
    *   **VADER**: Explicitly optimized for social media nuances, acronyms, capitalization weights, and emoji sentiment values.
    *   **TextBlob**: A fast, rule-based approach optimized for formal prose, general content, and long-form reviews.
*   **📝 Try-It-Yourself Sandbox**: Instantly evaluate single blocks of text, generate color-coded visual metrics cards, and clear your active scratchpad workspace with a single click.
*   **📁 Bulk File Upload**: Ingest and process high-volume `CSV` and `XLSX` (Excel) formatted files natively using highly computational batched streaming pipelines.
*   **👀 Dataset Previews**: Configure explicit visibility limits to peek inside your raw documents before running complex natural language pipelines.
*   **⚠️ Automatic Ghost & Blank Row Isolation**: Isolate messy data files. The pipeline strips completely empty spreadsheet cells, filters system parsing artifacts (`NA`, `N/A`, `null`), and partitions blanks into an isolated verification table to guarantee your data indices remain completely mathematically unbiased.
*   **📈 High-Impact Data Highlights**: Instantly generate key distribution indicators including **Total Dataset Volume**, **Average Sentiment Score**, and a **Net Sentiment Dominance Index**.
*   **📊 Data Visualizations**: Render crisp high-resolution Plotly Pie Charts, Polar Spread Histograms, and multi-threaded, side-by-side positive and negative theme **Word Clouds**.
*   **🔍 Raw Data Audit Trail**: Filter production strings down by conditional sentiment tags or perform localized regular-expression keyword lookups inside an interactive data browser.

---

## 🛠️ Tech Stack & Dependencies

This system operates entirely in **Python** using clean execution utilities managed via modern project dependency workflows (such as `uv` or `pip`).

*   **Streamlit**: Front-end engine and reactive visual component framework.
*   **Pandas & NumPy**: Optimized dataframe vectorized slicing and batch matrix calculations.
*   **VADER Sentiment & TextBlob**: Lexical dictionary evaluation libraries.
*   **Plotly Express**: Fully scalable interactive vector graphics.
*   **WordCloud & Matplotlib**: Theme keyword frequencies and image rendering engines.

---

## ⚙️ Installation & Workspace Setup

Clone this repository and configure your runtime environment using the instructions below:

### 1. Clone the Target Repository
```bash
git clone https://github.com
cd review-sentiment-analyzer
```

### 2. Configure Your Virtual Workspace

**Using standard pip:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

**Using the `uv` package manager (Recommended):**
```bash
uv sync
```

### 3. Launch the Application Locally
Run the Streamlit entry script to mount the dashboard server on your local port:
```bash
streamlit run app.py
```

---

## 📖 How To Use The App

### Mode 1: Single Text Sandbox
1. Open the sidebar navigation menu and select **Single Text Sandbox**.
2. Type or paste your evaluation target text into the sandbox text field.
3. Click **Analyze Text** to render contextual semantic metric alerts, or click **Clear Text** to flush your state back to empty.

### Mode 2: Bulk File Upload
1. Open the sidebar navigation menu and toggle over to **Bulk File Upload**.
2. Drag and drop any data log file (`.csv` or `.xlsx`).
3. Set your preferred row limit within the input box to instantly inspect the incoming dataset frame layout.
4. Select the specific column header containing your target string values via the drop-down menu.
5. Click **Process Dataset** to trigger batch processing. The dashboard will automatically filter out corrupted ghost entries and render your data visualizations instantly.
