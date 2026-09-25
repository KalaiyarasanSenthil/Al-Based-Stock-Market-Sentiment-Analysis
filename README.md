# AI-Based Stock Market Sentiment Analysis 📈

An AI-driven system that analyzes financial news articles to classify market sentiment (positive, neutral, negative) and studies its relationship with stock price movement.

## 📌 Problem Statement

Stock prices are influenced by many factors — company performance, innovation, and market sentiment shaped by news and media. With the sheer volume of financial news, investors struggle to interpret its impact on stock prices in real time.

This project builds an AI-driven sentiment analysis system that automatically processes historical news articles for a NASDAQ-listed company, classifies their sentiment, and studies the relationship between news sentiment and stock price/volume trends — helping analysts make more informed investment decisions.

## 🗂️ Dataset

The dataset contains daily news articles along with corresponding stock market data:

| Column | Description |
|--------|-------------|
| `Date` | Date the news was released |
| `News` | News article content |
| `Open` | Opening stock price ($) |
| `High` | Highest stock price ($) during the day |
| `Low` | Lowest stock price ($) during the day |
| `Close` | Closing stock price ($) |
| `Volume` | Number of shares traded |
| `Label` | Sentiment polarity (1 = positive, 0 = neutral, -1 = negative) |

## 🏗️ Project Pipeline

```
News Data
   ↓
Exploratory Data Analysis
   ↓
Text Embeddings
   ├── Word2Vec
   └── Sentence Transformers (BAAI/bge-base-en-v1.5, all-MiniLM-L6-v2)
        ↓
Classification Models
   ├── Random Forest
   └── Neural Network
        ↓
Model Evaluation
   ├── Accuracy, Precision, Recall, F1-Score
   └── Confusion Matrix
        ↓
Final Model Selection
```

## 🔍 Exploratory Data Analysis

- Distribution of news sentiment and its class balance
- Distribution of stock prices, trading volume, and news article length
- Correlation between price variables, volume, and sentiment
- Monthwise and time-series trends in stock prices
- Sentiment polarity vs. price behavior (boxplots)

## 🧠 Feature Engineering — Text Embeddings

News articles were converted into numerical vectors using three approaches:

1. **Word2Vec** — trained from scratch on the news corpus, generating 100-dimensional averaged word vectors.
2. **Sentence Transformer (BAAI/bge-base-en-v1.5)** — contextual sentence-level embeddings.
3. **Sentence Transformer (all-MiniLM-L6-v2)** — lightweight contextual sentence-level embeddings.

## 🤖 Models Built

For each embedding type, two classifiers were trained and evaluated:

- **Random Forest Classifier**
- **Neural Network** (Dense layers with ReLU activations, softmax output for 3 sentiment classes)

This resulted in 6 model variants compared on training and test performance.

## 📊 Evaluation

Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Test-set F1-Score was used as the primary criterion for final model selection, since it best reflects performance on unseen data.

## ✅ Key Findings

- The dataset's sentiment classes show varying distribution, which affects model performance across categories.
- Sentence Transformer embeddings capture contextual meaning better than Word2Vec's averaged word vectors.
- Random Forest and Neural Network models were compared across all embedding types to identify the best-performing combination.
- Model performance suggests scope for improvement through hyperparameter tuning, richer text preprocessing, and more training data.

## 🚀 Recommendations / Future Work

- Tune hyperparameters (Random Forest estimators, Neural Network architecture/layers).
- Apply advanced text preprocessing to improve embedding quality.
- Expand the dataset for better generalization.
- Extend the analysis with time-series modeling to study sentiment's predictive power on future price movement.

## 🛠️ Tech Stack

- **Python**, **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn** — visualization
- **Gensim** (Word2Vec)
- **Sentence-Transformers**, **Transformers**
- **Scikit-learn** — Random Forest, evaluation metrics
- **TensorFlow / Keras** — Neural Network

## 📁 Repository Structure

```
├── AI_Based_StockMarket_Sentiment_Analysis.ipynb   # Main notebook
└── README.md                                       # Project documentation
```

## ▶️ How to Run

1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn scipy gensim sentence-transformers tensorflow matplotlib seaborn
   ```
3. Open `AI_Based_StockMarket_Sentiment_Analysis.ipynb` in Jupyter Notebook / Google Colab.
4. Update the dataset path to point to your local/Drive copy of `stock_news.csv`.
5. Run all cells sequentially.

## 👤 Author

**S. Kalaiyarasan**
BSc IT, RVS College of Arts and Science, Sulur, Coimbatore
