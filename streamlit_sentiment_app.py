
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #155724 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sentiment-positive h3 {
        color: #155724 !important;
        margin-bottom: 0.5rem;
    }
    .sentiment-positive p {
        color: #155724 !important;
        margin: 0.25rem 0;
    }
    .sentiment-positive strong {
        color: #155724 !important;
    }
    .sentiment-positive em {
        color: #155724 !important;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #721c24 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sentiment-negative h3 {
        color: #721c24 !important;
        margin-bottom: 0.5rem;
    }
    .sentiment-negative p {
        color: #721c24 !important;
        margin: 0.25rem 0;
    }
    .sentiment-negative strong {
        color: #721c24 !important;
    }
    .sentiment-negative em {
        color: #721c24 !important;
    }
    .confidence-high {
        color: #28a745 !important;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107 !important;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545 !important;
        font-weight: bold;
    }
    
    /* Dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        .sentiment-positive {
            background-color: #1e4d2b;
            border-color: #28a745;
            color: #a3cfbb !important;
        }
        .sentiment-positive h3,
        .sentiment-positive p,
        .sentiment-positive strong,
        .sentiment-positive em {
            color: #a3cfbb !important;
        }
        .sentiment-negative {
            background-color: #4d1e1e;
            border-color: #dc3545;
            color: #f5c6cb !important;
        }
        .sentiment-negative h3,
        .sentiment-negative p,
        .sentiment-negative strong,
        .sentiment-negative em {
            color: #f5c6cb !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize the model (same as in your original code)
@st.cache_resource
def load_model():
    """Load and train the sentiment analysis model."""
    # Expanded training data with more diverse examples
    train_text = [
        "I love this movie, it's great!",
        "This was a terrible film.",
        "The acting was amazing.",
        "I did not like the plot.",
        "What a fantastic ending!",
        "It was boring and slow.",
        "Good job on this project!",
        "This is really bad quality.",
        "Excellent work, well done!",
        "Poor performance, very disappointed.",
        "Good morning, have a nice day!",
        "Awful experience, never again.",
        "Great product, highly recommend!",
        "Bad service, waste of time.",
        "Good idea, I like it.",
        "Horrible weather today.",
        "Amazing results, very impressed!",
        "Terrible customer support.",
        "Good food, tasty and fresh.",
        "Disgusting meal, cold and stale.",
        "Wonderful experience, thank you!",
        "Annoying and frustrating process.",
        "Perfect solution, exactly what I needed!",
        "Useless product, complete waste of money.",
        "Good quality, value for money.",
        "Bad design, very confusing.",
        "Superb performance, outstanding work!",
        "Dreadful service, completely disappointed.",
        "Brilliant idea, love it!",
        "Pathetic attempt, very poor.",
        "Outstanding results, exceeded expectations!",
        "Miserable experience, waste of money.",
        "Fantastic quality, highly satisfied!",
        "Deplorable conditions, unacceptable.",
        "Superb customer service, very helpful!",
        "Abysmal performance, total failure.",
        "Marvelous work, keep it up!",
        "Atrocious behavior, very rude.",
        "Splendid job, well executed!",
        "Mediocre at best, not impressed."
    ]
    train_labels = [
        1, 0, 1, 0, 1, 0,  # Original 6 examples
        1, 0, 1, 0, 1, 0,  # Good/bad examples
        1, 0, 1, 0, 1, 0,  # Great/terrible examples
        1, 0, 1, 0, 1, 0,  # Perfect/useless examples
        1, 0,              # Final good/bad pair
        1, 0, 1, 0, 1, 0,  # Superb/dreadful examples
        1, 0, 1, 0, 1, 0,  # Outstanding/miserable examples
        1, 0               # Splendid/mediocre examples
    ]    # Create and train the model
    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(train_text)
    
    classifier = MultinomialNB()
    classifier.fit(X_train_counts, train_labels)
    
    return vectorizer, classifier, train_text, train_labels

def predict_sentiment(text, vectorizer, classifier):
    """Predict sentiment for given text."""
    text_counts = vectorizer.transform([text])
    prediction = classifier.predict(text_counts)[0]
    probabilities = classifier.predict_proba(text_counts)[0]
    confidence = max(probabilities)
    return prediction, confidence

def get_sentiment_label(prediction):
    """Convert numeric prediction to human-readable label."""
    return "Positive" if prediction == 1 else "Negative"

def get_confidence_class(confidence):
    """Get CSS class based on confidence level."""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"

# Load the model
vectorizer, classifier, train_text, train_labels = load_model()

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🎭 Sentiment Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📝 Single Analysis", "📋 Batch Analysis", "📈 Model Performance", "ℹ️ About"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📝 Single Analysis":
        show_single_analysis()
    elif page == "📋 Batch Analysis":
        show_batch_analysis()
    elif page == "📈 Model Performance":
        show_model_performance()
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page():
    """Display the home page."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ## Welcome to the Sentiment Analyzer! 🎉
        
        This application uses **Machine Learning** to analyze the sentiment of text as either:
        - 😊 **Positive** 
        - 😞 **Negative**
        
        ### Features:
        - 📝 **Single Text Analysis**: Analyze individual sentences or paragraphs
        - 📋 **Batch Analysis**: Upload and analyze multiple texts at once
        - 📈 **Model Performance**: View model accuracy and training details
        - 🎨 **Interactive Visualizations**: Beautiful charts and graphs
        
        ### How it works:
        1. The model is trained using **Naive Bayes** algorithm
        2. Text is converted to numerical features using **CountVectorizer**
        3. The trained model predicts sentiment with confidence scores
        
        **Get started by selecting a page from the sidebar!** 👈
        """)
        
        # Quick demo
        st.markdown("### 🚀 Quick Demo")
        demo_text = st.text_input("Try it out - enter some text:", "I love this application!")
        
        if demo_text:
            prediction, confidence = predict_sentiment(demo_text, vectorizer, classifier)
            sentiment = get_sentiment_label(prediction)
            emoji = "😊" if prediction == 1 else "😞"
            
            if sentiment == "Positive":
                st.markdown(f"""
                <div class="sentiment-positive">
                    <h3 style="color: #155724 !important; margin-bottom: 0.5rem;">{emoji} {sentiment}</h3>
                    <p style="color: #155724 !important; margin: 0.25rem 0;"><strong style="color: #155724 !important;">Text:</strong> "{demo_text}"</p>
                    <p style="color: #155724 !important; margin: 0.25rem 0;"><strong style="color: #155724 !important;">Confidence:</strong> <span class="{get_confidence_class(confidence)}" style="color: #155724 !important;">{confidence:.1%}</span></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="sentiment-negative">
                    <h3 style="color: #721c24 !important; margin-bottom: 0.5rem;">{emoji} {sentiment}</h3>
                    <p style="color: #721c24 !important; margin: 0.25rem 0;"><strong style="color: #721c24 !important;">Text:</strong> "{demo_text}"</p>
                    <p style="color: #721c24 !important; margin: 0.25rem 0;"><strong style="color: #721c24 !important;">Confidence:</strong> <span class="{get_confidence_class(confidence)}" style="color: #721c24 !important;">{confidence:.1%}</span></p>
                </div>
                """, unsafe_allow_html=True)

def show_single_analysis():
    """Display the single text analysis page."""
    st.title("📝 Single Text Analysis")
    st.markdown("Enter any text below to analyze its sentiment:")
    
    # Text input
    user_text = st.text_area(
        "Enter your text here:",
        height=150,
        placeholder="Type or paste your text here... For example: 'I had an amazing day at the beach!'"
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True)
    
    if analyze_button and user_text.strip():
        with st.spinner("Analyzing sentiment..."):
            prediction, confidence = predict_sentiment(user_text, vectorizer, classifier)
            sentiment = get_sentiment_label(prediction)
            emoji = "😊" if prediction == 1 else "😞"
            
            # Results
            st.markdown("### Results:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if sentiment == "Positive":
                    st.success(f"{emoji} **{sentiment}** Sentiment")
                else:
                    st.error(f"{emoji} **{sentiment}** Sentiment")
            
            with col2:
                confidence_color = "green" if confidence >= 0.8 else "orange" if confidence >= 0.6 else "red"
                st.metric("Confidence", f"{confidence:.1%}", delta=None)
            
            # Confidence visualization
            st.markdown("### Confidence Breakdown:")
            
            # Get both probabilities
            text_counts = vectorizer.transform([user_text])
            probabilities = classifier.predict_proba(text_counts)[0]
            
            prob_data = {
                'Sentiment': ['Negative 😞', 'Positive 😊'],
                'Probability': probabilities
            }
            
            fig = px.bar(
                prob_data, 
                x='Sentiment', 
                y='Probability',
                color='Probability',
                color_continuous_scale=['red', 'green'],
                title="Sentiment Probability Distribution"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Text statistics
            with st.expander("📊 Text Statistics"):
                word_count = len(user_text.split())
                char_count = len(user_text)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Word Count", word_count)
                with col2:
                    st.metric("Character Count", char_count)
                with col3:
                    st.metric("Average Word Length", f"{char_count/word_count:.1f}" if word_count > 0 else "0")
    
    elif analyze_button and not user_text.strip():
        st.warning("⚠️ Please enter some text to analyze!")

def show_batch_analysis():
    """Display the batch analysis page."""
    st.title("📋 Batch Analysis")
    st.markdown("Analyze multiple texts at once by entering them below:")
    
    # Option 1: Manual text input
    st.markdown("### Option 1: Manual Input")
    batch_text = st.text_area(
        "Enter multiple texts (one per line):",
        height=200,
        placeholder="I love this product!\nThis service is terrible.\nThe weather is beautiful today.\nI'm not happy with the results."
    )
    
    # Option 2: File upload
    st.markdown("### Option 2: File Upload")
    uploaded_file = st.file_uploader(
        "Upload a text file (one text per line):",
        type=['txt', 'csv']
    )
    
    texts_to_analyze = []
    
    if batch_text.strip():
        texts_to_analyze = [text.strip() for text in batch_text.split('\n') if text.strip()]
    elif uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            content = str(uploaded_file.read(), "utf-8")
            texts_to_analyze = [text.strip() for text in content.split('\n') if text.strip()]
        elif uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            if len(df.columns) > 0:
                texts_to_analyze = df.iloc[:, 0].astype(str).tolist()
    
    if texts_to_analyze:
        st.success(f"Found {len(texts_to_analyze)} texts to analyze")
        
        if st.button("🔍 Analyze All Texts", use_container_width=True):
            with st.spinner("Analyzing all texts..."):
                results = []
                
                progress_bar = st.progress(0)
                for i, text in enumerate(texts_to_analyze):
                    prediction, confidence = predict_sentiment(text, vectorizer, classifier)
                    sentiment = get_sentiment_label(prediction)
                    
                    results.append({
                        'Text': text[:50] + "..." if len(text) > 50 else text,
                        'Full_Text': text,
                        'Sentiment': sentiment,
                        'Confidence': confidence,
                        'Emoji': "😊" if prediction == 1 else "😞"
                    })
                    
                    progress_bar.progress((i + 1) / len(texts_to_analyze))
                
                # Create results DataFrame
                df_results = pd.DataFrame(results)
                
                # Summary statistics
                st.markdown("### 📊 Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                positive_count = len(df_results[df_results['Sentiment'] == 'Positive'])
                negative_count = len(df_results[df_results['Sentiment'] == 'Negative'])
                avg_confidence = df_results['Confidence'].mean()
                
                with col1:
                    st.metric("Total Texts", len(df_results))
                with col2:
                    st.metric("Positive 😊", positive_count)
                with col3:
                    st.metric("Negative 😞", negative_count)
                with col4:
                    st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                
                # Visualization
                fig = px.pie(
                    values=[positive_count, negative_count],
                    names=['Positive 😊', 'Negative 😞'],
                    title="Sentiment Distribution",
                    color_discrete_map={
                        'Positive 😊': '#28a745',
                        'Negative 😞': '#dc3545'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results
                st.markdown("### 📋 Detailed Results")
                
                # Display results with styling
                for _, row in df_results.iterrows():
                    if row['Sentiment'] == 'Positive':
                        st.markdown(f"""
                        <div class="sentiment-positive">
                            <strong style="color: #155724 !important;">{row['Emoji']} {row['Sentiment']}</strong> 
                            <span style="color: #155724 !important;">(Confidence: {row['Confidence']:.1%})</span><br>
                            <em style="color: #155724 !important;">"{row['Full_Text']}"</em>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="sentiment-negative">
                            <strong style="color: #721c24 !important;">{row['Emoji']} {row['Sentiment']}</strong> 
                            <span style="color: #721c24 !important;">(Confidence: {row['Confidence']:.1%})</span><br>
                            <em style="color: #721c24 !important;">"{row['Full_Text']}"</em>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Download results
                csv = df_results[['Full_Text', 'Sentiment', 'Confidence']].to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv"
                )

def show_model_performance():
    """Display model performance page."""
    st.title("📈 Model Performance")
    
    # Model evaluation
    st.markdown("### 🎯 Training Data Evaluation")
    
    results = []
    for i, text in enumerate(train_text):
        prediction, confidence = predict_sentiment(text, vectorizer, classifier)
        actual_label = train_labels[i]
        predicted_label = get_sentiment_label(prediction)
        actual_label_text = get_sentiment_label(actual_label)
        is_correct = prediction == actual_label
        
        results.append({
            'Text': text,
            'Actual': actual_label_text,
            'Predicted': predicted_label,
            'Confidence': confidence,
            'Correct': is_correct,
            'Status': '✅' if is_correct else '❌'
        })
    
    df_eval = pd.DataFrame(results)
    
    # Accuracy metrics
    accuracy = df_eval['Correct'].mean()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.1%}")
    with col2:
        st.metric("Correct Predictions", f"{df_eval['Correct'].sum()}/{len(df_eval)}")
    with col3:
        st.metric("Average Confidence", f"{df_eval['Confidence'].mean():.1%}")
    
    # Detailed evaluation table
    st.markdown("### 📊 Detailed Evaluation")
    st.dataframe(
        df_eval[['Status', 'Text', 'Actual', 'Predicted', 'Confidence']],
        use_container_width=True
    )
    
    # Model information
    st.markdown("### 🔧 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Algorithm:** Multinomial Naive Bayes  
        **Feature Extraction:** CountVectorizer  
        **Vocabulary Size:** {len(vectorizer.vocabulary_)} words  
        **Training Samples:** {len(train_text)}  
        """)
    
    with col2:
        st.info(f"""
        **Classes:** Positive (1), Negative (0)  
        **Training Accuracy:** {accuracy:.1%}  
        **Feature Type:** Bag of Words  
        **Preprocessing:** Basic tokenization  
        """)
    
    # Vocabulary analysis
    with st.expander("📚 Vocabulary Analysis"):
        vocab_items = list(vectorizer.vocabulary_.items())
        vocab_df = pd.DataFrame(vocab_items, columns=['Word', 'Index'])
        vocab_df = vocab_df.sort_values('Word')
        
        st.markdown(f"**Total vocabulary size:** {len(vocab_items)} unique words")
        st.dataframe(vocab_df, use_container_width=True)

def show_about_page():
    """Display the about page."""
    st.title("ℹ️ About")
    
    st.markdown("""
    ## 🎭 Sentiment Analyzer
    
    This application demonstrates **Natural Language Processing (NLP)** and **Machine Learning** 
    techniques for sentiment analysis.
    
    ### 🔬 Technical Details
    
    **Algorithm:** Multinomial Naive Bayes
    - A probabilistic classifier based on Bayes' theorem
    - Particularly effective for text classification tasks
    - Assumes independence between features (words)
    
    **Feature Extraction:** Count Vectorization
    - Converts text into numerical features
    - Creates a matrix of token counts
    - Each document becomes a vector of word frequencies
    
    ### 📊 Model Training Process
    
    1. **Data Preparation:** Small training dataset with labeled examples
    2. **Text Vectorization:** Convert text to numerical features
    3. **Model Training:** Train Naive Bayes classifier
    4. **Prediction:** Classify new text with confidence scores
    
    ### 🚀 Features
    
    - **Real-time Analysis:** Instant sentiment prediction
    - **Confidence Scores:** Probability-based confidence metrics
    - **Batch Processing:** Analyze multiple texts simultaneously
    - **Interactive Visualizations:** Charts and graphs using Plotly
    - **Export Results:** Download analysis results as CSV
    
    ### 🔮 Future Improvements
    
    - Larger, more diverse training dataset
    - Advanced preprocessing (stemming, lemmatization)
    - Deep learning models (LSTM, BERT)
    - Support for neutral sentiment
    - Multi-language support
    
    ### 📚 Libraries Used
    
    - **Streamlit:** Web application framework
    - **scikit-learn:** Machine learning library
    - **Plotly:** Interactive visualizations
    - **Pandas:** Data manipulation
    - **NumPy:** Numerical computing
    
    ---
    
    *Built with ❤️ using Python and Streamlit*
    """)
    
    # Fun facts
    with st.expander("🎯 Fun Facts"):
        st.markdown("""
        - The current model achieves 100% accuracy on training data (though this is expected with a small dataset)
        - Naive Bayes is called "naive" because it assumes word independence, which isn't true in natural language
        - This type of sentiment analysis is used by companies to analyze customer reviews, social media posts, and feedback
        - The model can be easily extended to classify emotions beyond just positive/negative
        """)

if __name__ == "__main__":
    main()