# 🎭 Sentiment Analyzer

A comprehensive sentiment analysis application built with Python, featuring both a command-line interface and an interactive Streamlit web application.

## 🚀 Features

- **Machine Learning-Based**: Uses Multinomial Naive Bayes classifier
- **Dual Interface**: Command-line tool and web application
- **Interactive Web UI**: Built with Streamlit for user-friendly experience
- **Batch Analysis**: Analyze multiple texts simultaneously
- **Confidence Scores**: Get probability-based confidence metrics
- **Visual Analytics**: Interactive charts and graphs
- **Export Results**: Download analysis results as CSV
- **Real-time Processing**: Instant sentiment prediction

## 📊 Sentiment Classifications

- 😊 **Positive**: Happy, satisfied, enthusiastic sentiments
- 😞 **Negative**: Sad, disappointed, frustrated sentiments

## 🛠️ Technology Stack

- **Python 3.7+**
- **Scikit-learn**: Machine learning library
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

## 🔧 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AnandShadow/sentiment-analyzer.git
   cd sentiment-analyzer
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Command Line Interface

Run the basic sentiment analyzer:

```bash
python sentiment_analyzer.py
```

This will:
- Display model training information
- Show evaluation results
- Test with sample texts
- Launch an interactive demo

### Streamlit Web Application

Launch the web interface:

```bash
streamlit run streamlit_sentiment_app.py
```

Then open your browser to `http://localhost:8501`

### Web Application Features

#### 🏠 Home Page
- Quick overview of the application
- Instant demo with sample text
- Feature highlights

#### 📝 Single Analysis
- Analyze individual texts
- Real-time sentiment prediction
- Confidence scores and visualizations
- Text statistics

#### 📋 Batch Analysis
- Upload text files or enter multiple texts
- Process multiple sentiments simultaneously
- Summary statistics and visualizations
- Export results to CSV

#### 📈 Model Performance
- View training data evaluation
- Model accuracy metrics
- Vocabulary analysis
- Technical details

#### ℹ️ About
- Technical documentation
- Algorithm explanations
- Future improvements
- Library information

## 📈 Model Details

### Algorithm: Multinomial Naive Bayes
- **Feature Extraction**: CountVectorizer (Bag of Words)
- **Training Data**: 6 labeled examples (expandable)
- **Classes**: Binary classification (Positive/Negative)
- **Current Accuracy**: 100% on training data

### Model Pipeline
1. **Text Preprocessing**: Basic tokenization
2. **Feature Extraction**: Convert text to numerical vectors
3. **Classification**: Multinomial Naive Bayes prediction
4. **Confidence Scoring**: Probability-based confidence metrics

## 📝 Example Usage

```python
from sentiment_analyzer import predict_sentiment, get_sentiment_label

# Analyze a single text
text = "I love this amazing product!"
prediction, confidence = predict_sentiment(text)
sentiment = get_sentiment_label(prediction)

print(f"Text: {text}")
print(f"Sentiment: {sentiment}")
print(f"Confidence: {confidence:.2%}")
```

## 📋 File Structure

```
sentiment-analyzer/
│
├── sentiment_analyzer.py          # Command-line sentiment analyzer
├── streamlit_sentiment_app.py     # Streamlit web application
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .gitignore                     # Git ignore rules
└── venv/                          # Virtual environment (excluded from git)
```

## 🔮 Future Improvements

- [ ] Larger and more diverse training dataset
- [ ] Advanced text preprocessing (stemming, lemmatization)
- [ ] Deep learning models (LSTM, BERT, Transformers)
- [ ] Support for neutral sentiment classification
- [ ] Multiple language support
- [ ] Real-time social media sentiment analysis
- [ ] API endpoint for integration
- [ ] Model persistence and loading
- [ ] A/B testing for different algorithms

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**AnandShadow**
- GitHub: [@AnandShadow](https://github.com/AnandShadow)

## 🙏 Acknowledgments

- Scikit-learn team for the excellent machine learning library
- Streamlit team for the amazing web framework
- Plotly for interactive visualizations
- Python community for continuous support

## 📞 Support

If you have any questions or run into issues, please:
1. Check the existing issues on GitHub
2. Create a new issue with detailed information
3. Provide steps to reproduce any bugs

---

⭐ **Star this repository if you found it helpful!** ⭐