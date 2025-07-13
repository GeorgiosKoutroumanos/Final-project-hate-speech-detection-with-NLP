# 🛡️ Hate Speech Detection with NLP

A robust NLP pipeline that leverages **transformers (RoBERTa)** to classify text into *hate speech*, *offensive language*, or *neutral content*. The model has been trained on a combination of public and subtle hate speech datasets, cleaned and enhanced using linguistic preprocessing and advanced feature engineering techniques. The final model is deployed through a **Streamlit app**.

---

## 📌 Project Overview

- **Goal**: Automatically detect hate speech and offensive language in social media posts using deep learning and NLP.
- **Model Used**: `distilroberta-base` (lightweight version of RoBERTa)
- **Frameworks**: HuggingFace `transformers`, `scikit-learn`, `NLTK`, `Streamlit`, `SymSpell`, `pyLDAvis`
- **Deployment**: Streamlit App
- Code executed on Google Colab

---

## 📁 Dataset Sources

| Dataset                          | Description                                           |
|----------------------------------|-------------------------------------------------------|
| `Dataset-Hate-Speech-Detection.csv` | Primary dataset of labeled tweets                     |
| `subtle_hp_dataset.csv`         | Additional dataset with examples of subtle hate       |
| `reddit_comments.csv`           | Real-world Reddit data used for inference & testing   |

### Class Labels:
- `0`: Hate Speech
- `1`: Offensive Language
- `2`: Neutral/Neither

---

## 🧹 Data Preprocessing Pipeline

1. **Text Cleaning**:
   - Remove URLs, emojis, mentions, hashtags
   - Normalize punctuation
   - Convert text to lowercase

2. **Spelling Correction**:
   - Powered by `SymSpell` using `frequency_dictionary_en_82_765.txt`

3. **Tokenization & Filtering**:
   - `TweetTokenizer` from NLTK
   - Stopword removal
   
## 📊 Exploratory Data Analysis (EDA)

1. **Class Distribution**:
   - Visualized with bar and pie charts
   - Showed imbalance across the three classes: hate, offensive, and neutral

2. **Word and Phrase Patterns**:
   - Most frequent words identified per class
   - Word clouds generated for quick visual overview
   - Top bigrams extracted using `CountVectorizer`

3. **TF-IDF Analysis**:
   - Highlighted key offensive and hateful terms across classes
   - Heatmaps used to compare term usage between labels

4. **Sentiment Scoring**:
   - Used `VADER` to assign sentiment scores to each text
   - Plotted average sentiment per class to reveal tone differences

5. **Topic Modeling**:
   - Applied LDA (Latent Dirichlet Allocation) to discover underlying topics
   - Visualized topic distributions with `pyLDAvis`

## ⚖️ Handling Class Imbalance

- Used **upsampling** on minority classes (`class 0` and `class 2`) to match the majority class size using `sklearn.utils.resample`.
- Ensured deduplication after merging datasets.

---

## 🤖 Model Training

### Model Architecture

| Component         | Value                      |
|------------------|----------------------------|
| Base Model       | `distilroberta-base`       |
| Tokenizer        | `AutoTokenizer`            |
| Classifier Head  | Sequence Classification Head (3 labels) |
| Dropout          | 0.3                        |

### Training Details

- Split: 80% training, 20% testing
- Batch size: 20
- Epochs: 8 (with **EarlyStoppingCallback**)
- Max sequence length: 190 tokens
- Evaluation Metric: `eval_loss`
- Optimizer Config: AdamW with weight decay (`0.01`)

### Trainer Configuration
```python
TrainingArguments(
    output_dir='./results',
    num_train_epochs=8,
    per_device_train_batch_size=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    weight_decay=0.01
)
```

---

## 📈 Evaluation Metrics

```python
{'eval_loss': 0.2138, 'accuracy': ~0.94}
```
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	0.633800	0.432050	0.847181	0.846327	0.853060	0.847181
2	0.403200	0.318154	0.887820	0.888004	0.890274	0.887820
3	0.324400	0.301429	0.908875	0.908895	0.911882	0.908875
4	0.272700	0.264181	0.925741	0.925340	0.928912	0.925741
5	0.238900	0.235427	0.937967	0.937496	0.939959	0.937967
6	0.209400	0.230335	0.947136	0.946908	0.947584	0.947136
7	0.183400	0.202349	0.949287	0.949011	0.950131	0.949287
8	0.172300	0.228162	0.949174	0.948830	0.950173	0.949174
```

### Classification Report (Example)
| Class         | Precision | Recall | F1-score |
|---------------|-----------|--------|----------|
| Hate Speech   | 0.95      | 0.96   | 0.95     |
| Offensive     | 0.91      | 0.89   | 0.92     |
| Neutral       | 0.97      | 0.97   | 0.97     |

- **Confusion Matrix** visualized with Seaborn
- Misclassified examples printed for each class

---

## 🧪 Subtle Hate Speech Test

Used a custom pipeline to evaluate the model on tricky examples like:

```
"Oh sure, let's put another brilliant woman in charge, what could possibly go wrong."
```

The model detects many subtle cues, but still **misclassifies sarcasm** or disguised hate, suggesting room for fine-tuning or additional data.

---

## 🧼 Reddit Comment Analysis

- Cleaned Reddit comments using the same pipeline
- Batch prediction using the trained model
- Generated predictions and exported top 30 samples per class

```python
df_combined.to_csv('sampled_comments_per_class.csv')
```

---

## 🖥️ Streamlit App (UI)

### Features:
- Input raw text
- Predicts class label and confidence
- Deployable locally or via Hugging Face

### Launch Locally:
```bash
streamlit run app.py
```

### Hugging Face Hosted Version:
```python
MODEL_REPO = "GeorgiosKoutroumanos/NLP-Roberta-HP-detection"
```
Live App: [Streamlit App](https://final-project-hate-speech-detection-with-nlp.streamlit.app)

---

## 💾 Model Saving & Loading

```python
model.save_pretrained('/content/drive/MyDrive/best_model')
tokenizer.save_pretrained('/content/drive/MyDrive/best_model')
```

Can be reloaded anytime using:
```python
AutoModelForSequenceClassification.from_pretrained("best_model")
```

---

## 📁 Project Structure

```
├── MAIN.py                      # Full model pipeline (Colab)
├── app.py                       # Streamlit interface
├── best_model/                  # Saved model and tokenizer
├── Dataset-Hate-Speech-Detection.csv
├── subtle_hp_dataset.csv
├── reddit_comments.csv
├── sampled_comments_per_class.csv
└── README.md
```

---

## 🙋‍♂️ Future Work

- Integrate sarcasm detection module & additional data on subtle Hate Speech
- Expand to multilingual datasets
- Classify specific hate categories

---

## 👨‍💻 Author

**Koutroumanos Giorgos**  
📧 g.koutroumanos@outlook.com  
[GitHub](https://github.com/GeorgiosKoutroumanos/Final-project-hate-speech-detection-with-NLP.git)
