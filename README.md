# Understanding Customer Sentiment Through Natural Language Processing  

The project focused on building a robust pipeline to extract actionable insights from large-scale Amazon Office Products reviews by combining **exploratory data analysis (EDA)**, **topic modelling**, and **sentiment classification**.

---

## Project Overview  

Online reviews have exploded in recent years and are now critical to both customers and companies. However, the informal, noisy, and context-dependent nature of these reviews makes large-scale analysis challenging.  

I built an **end-to-end NLP pipeline** that:

- Cleans and pre-processes raw customer reviews  
- Discovers hidden themes/topics in the data  
- Classifies sentiments (positive/neutral/negative) at scale  

This pipeline is intended to help product teams understand customer pain points, discover unmet needs, and improve product design—especially for eco-friendly items.

---

## Methodology  

### 1. Data Collection  
I used the [Amazon Office Products reviews dataset](https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Office_Products_5.json.gz) (~800,000 reviews).  

### 2. Data Preprocessing  

- **Text Cleaning:** Removed special characters, non-alphabetic tokens, lowercased text  
- **Lemmatization & Stop-word Removal:** Used NLTK’s `WordNetLemmatizer` and English stopword list  
- **Parallel Processing:** Accelerated cleaning with the `swifter` library  

### 3. Exploratory Data Analysis (EDA)  

- Distribution of star ratings  
- Review lengths (word counts)  
- Year-wise trends in reviews and average ratings  

### 4. Topic Modelling  

- Used **BERTopic**, which combines:
  - Sentence-Transformers (`all-MiniLM-L6-v2`) for embeddings  
  - UMAP for dimensionality reduction  
  - HDBSCAN for clustering  

Identified **32 topics**, e.g. *“pen & ink”*, *“printer performance”*, etc.  

### 5. Sentiment Analysis  

- **VADER** for initial sentiment scoring  
- Labeled reviews as Positive (>3 stars), Neutral (=3 stars), or Negative (≤2 stars)  

### 6. Sentiment Classification  

- Balanced the dataset by upsampling minority classes and downsampling the majority class  
- Generated multiple feature sets:
  - Sentence embeddings  
  - TF-IDF + LSA  
  - CountVectorizer + LDA  
- Trained and evaluated multiple models:
  - Logistic Regression  
  - Multinomial Naive Bayes  
  - Linear SVM  
  - Random Forest  
  - XGBoost  
  - **Stacked Ensemble (best performer)**  

---

## Key Results  

| Model | Accuracy | F1-Score |
|-------|---------:|---------:|
| **Stacked Ensemble (tuned)** | **0.8500** | **0.8498** |
| Random Forest (tuned) | 0.8233 | 0.8214 |
| Linear SVM | 0.8200 | 0.8188 |
| Multinomial Naive Bayes | 0.7467 | 0.7416 |
| XGBoost (tuned) | 0.8000 | 0.7988 |

- BERTopic successfully uncovered 32 coherent themes  
- The tuned **Stacked Ensemble** outperformed all other classifiers  
- EDA revealed clear shifts in review trends and ratings over time  

---

## Impact  

This project shows how **aspect-based sentiment analysis** can give product teams a clear map of strengths, weaknesses, and unmet needs hidden inside unstructured text. It also provides a foundation for:

- Fine-tuning transformer models (BERT, RoBERTa) on labelled review data  
- Extending to multilingual or real-time streaming reviews  
- Handling mixed-sentiment reviews more precisely  
- Deploying lightweight models for large-scale processing  

---

## Resources  
 
- **Code (Google Colab):** [Colab Notebook](https://colab.research.google.com/drive/1q5Up0VLwI6aipPoPg0dlxCeS8iCgwksk?usp=sharing)  

---

## Technologies Used  

- **Python (Google Colab)**  
- **NLTK** (text preprocessing, VADER sentiment)  
- **Sentence-Transformers** (embeddings)  
- **BERTopic / UMAP / HDBSCAN** (topic modelling)  
- **scikit-learn** (Logistic Regression, SVM, Random Forest, Naive Bayes, metrics)  
- **XGBoost**  
- **Pandas / NumPy / Matplotlib** (EDA and visualization)  
- **Swifter** (parallel processing)  

---

## Future Work  

- Fine-tune BERT/RoBERTa for direct end-to-end sentiment classification and aspect extraction  
- Extend pipeline to multi-lingual analysis and real-time streaming data  
- Introduce multi-aspect scoring for mixed reviews  
- Deploy lightweight models for large-scale, on-device inference  

---

### If you find this project useful, feel free to fork it or star it on GitHub!
