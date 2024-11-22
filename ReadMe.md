# Sentiment Analysis on Twitter Data

This project performs sentiment analysis on tweets from the Sentiment140 dataset. It classifies tweets as **Positive**, **Negative**, or **Neutral** and generates visualizations like sentiment distribution bar charts and word clouds.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the required NLTK resources by running the following in Python:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

## Usage

1. Place the **Sentiment140** dataset (`sentiment140.csv`) in the `data/` folder.
2. Run the main script:

```bash
python main.py
```

This will:
- Clean the data
- Perform sentiment analysis
- Generate visualizations (sentiment bar charts, word clouds)

## Project Structure

```
sentiment-analysis/
├── data/                    # Contains the dataset
│   └── sentiment140.csv      # Sentiment140 dataset
├── main.py                  # Main script for analysis
├── requirements.txt         # Dependencies
└── README.md                # Project instructions
```

