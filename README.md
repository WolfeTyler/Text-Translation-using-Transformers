# 🌍 Text Translation and Sentiment Analysis using Transformers

A comprehensive NLP project that translates movie reviews from multiple languages and performs sentiment analysis using state-of-the-art Transformer models from HuggingFace.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.35%2B-yellow.svg)](https://huggingface.co/docs/transformers/index)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This project analyzes the sentiment of **30 movie reviews** in three different languages:
- 🇬🇧 **English** (10 movies)
- 🇫🇷 **French** (10 movies)
- 🇪🇸 **Spanish** (10 movies)

The pipeline performs:
1. **Data Preprocessing** - Consolidates multi-language datasets
2. **Neural Machine Translation** - Translates French and Spanish reviews to English
3. **Sentiment Analysis** - Classifies reviews as Positive or Negative

## 🎯 Key Features

- ✅ **Multi-language Support** - Processes English, French, and Spanish text
- ✅ **State-of-the-art Models** - Uses HuggingFace's pre-trained Transformers
- ✅ **Batch Processing** - Efficiently handles multiple translations
- ✅ **Comprehensive Analysis** - Includes both reviews and synopses
- ✅ **Preserves Metadata** - Maintains original language information

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Core programming language |
| **HuggingFace Transformers** | NLP models (MarianMT, DistilBERT) |
| **PyTorch** | Deep learning framework |
| **Pandas** | Data manipulation and analysis |
| **Jupyter Notebook** | Interactive development environment |

## 🚀 Quick Start

### Prerequisites

```bash
pip install transformers sentencepiece torch pandas
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/WolfeTyler/Text-Translation-using-Transformers.git
   cd Text-Translation-using-Transformers
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook project.ipynb
   ```

### Project Structure

```
Text-Translation-using-Transformers/
├── data/
│   ├── movie_reviews_eng.csv    # English reviews
│   ├── movie_reviews_fr.csv     # French reviews
│   └── movie_reviews_sp.csv     # Spanish reviews
├── result/
│   └── reviews_with_sentiment.csv   # Output file
├── project.ipynb                # Main Jupyter notebook
└── README.md                    # This file
```

## 📊 Models Used

### Translation Models (MarianMT)
- **French → English**: `Helsinki-NLP/opus-mt-fr-en`
- **Spanish → English**: `Helsinki-NLP/opus-mt-es-en`

### Sentiment Analysis Model
- **DistilBERT**: `distilbert-base-uncased-finetuned-sst-2-english`
  - Fine-tuned on Stanford Sentiment Treebank (SST-2)
  - Binary classification: Positive/Negative

## 💡 How It Works

### 1. Data Preprocessing
```python
def preprocess_data() -> pd.DataFrame:
    # Reads CSVs, standardizes columns, adds language tags
    # Combines all datasets into single DataFrame
```

### 2. Translation Pipeline
```python
def translate(text: str, model, tokenizer) -> str:
    # Tokenizes input text
    # Generates translation using MarianMT
    # Decodes and returns translated text
```

### 3. Sentiment Analysis
```python
def analyze_sentiment(text, classifier):
    # Classifies review sentiment
    # Returns: "Positive" or "Negative"
```

## 📈 Output Format

The final CSV file contains:

| Column | Description |
|--------|-------------|
| **Title** | Movie/TV series name |
| **Year** | Release year |
| **Synopsis** | Plot summary (translated to English) |
| **Review** | Review text (translated to English) |
| **Original Language** | Source language (`en`/`fr`/`sp`) |
| **Sentiment** | Classification result (`Positive`/`Negative`) |

## 🔍 Example Results

| Title | Original Language | Sentiment |
|-------|------------------|-----------|
| The Shawshank Redemption | en | Positive |
| Intouchables | fr | Positive |
| Roma | sp | Positive |
| Blade Runner 2049 | en | Negative |
| Le Dîner de Cons | fr | Negative |
| Torrente | sp | Negative |

## 📚 Learning Outcomes

This project demonstrates proficiency in:

- 🧠 **Natural Language Processing** - Translation, sentiment analysis
- 🔧 **ML Pipeline Development** - Data preprocessing, model integration
- 📊 **Data Manipulation** - Pandas operations, multi-source data handling
- 🤖 **Transfer Learning** - Leveraging pre-trained models
- 📝 **Best Practices** - Clean code, documentation, reproducibility

## 🔧 Troubleshooting

### Common Issues

**ImportError: No module named 'sentencepiece'**
```bash
pip install sentencepiece
# Then restart your Jupyter kernel
```

**CUDA out of memory**
```python
# Use CPU instead
device = torch.device('cpu')
model.to(device)
```

**Slow first run**
- Models download ~650MB on first execution
- Subsequent runs use cached models (much faster)

## 📊 Performance Metrics

- **Translation Speed**: ~1-2 seconds per text
- **Total Processing Time**: ~2-3 minutes for 30 movies
- **Model Downloads**: ~650MB (first run only)
- **Accuracy**: Leverages state-of-the-art pre-trained models

## 👤 Author

**Tyler Wolfe**
- GitHub: [@WolfeTyler](https://github.com/WolfeTyler)
- LinkedIn: [Tyler Wolfe](https://www.linkedin.com/in/tyler-wolfe/)

## 🙏 Acknowledgments

- [HuggingFace](https://huggingface.co/) for providing pre-trained models
- [Helsinki-NLP](https://github.com/Helsinki-NLP) for MarianMT translation models
- Dataset sourced from various movie review platforms

---
