# Import the tools we need
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------- 1. The Data -------------------- #

# Some sample text data for our model to learn from
train_text = [
    "I love this movie, it's great!",
    "This was a terrible film.",
    "The acting was amazing.",
    "I did not like the plot.",
    "What a fantastic ending!",
    "It was boring and slow."
]

# The labels for our text data (1 for positive, 0 for negative)
train_labels = [1, 0, 1, 0, 1, 0]

# -------------------- 2. Feature Extraction (Vectorization) -------------------- #

# Create a CountVectorizer object to convert text into a matrix of token counts
vectorizer = CountVectorizer()

# Learn the vocabulary and transform our training data into a matrix
X_train_counts = vectorizer.fit_transform(train_text)

# -------------------- 3. Model Training -------------------- #

# Create a Multinomial Naive Bayes classifier
classifier = MultinomialNB()

# Train the classifier with our vectorized data and labels
classifier.fit(X_train_counts, train_labels)

print("Model trained successfully!")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")


# -------------------- 4. Prediction Function -------------------- #

def predict_sentiment(text):
    """
    Predict the sentiment of a given text.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        tuple: (prediction, confidence) where prediction is 1 for positive,
               0 for negative and confidence is the probability of prediction
    """
    # Transform the input text using the same vectorizer
    text_counts = vectorizer.transform([text])
    
    # Make prediction
    prediction = classifier.predict(text_counts)[0]
    
    # Get prediction probabilities for confidence
    probabilities = classifier.predict_proba(text_counts)[0]
    confidence = max(probabilities)
    
    return prediction, confidence


def get_sentiment_label(prediction):
    """Convert numeric prediction to human-readable label."""
    return "Positive" if prediction == 1 else "Negative"


# -------------------- 5. Model Evaluation -------------------- #

def evaluate_model():
    """Evaluate the model on the training data and display results."""
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Test on training data (in real scenarios, use separate test data)
    correct_predictions = 0
    total_predictions = len(train_text)
    
    print("Testing on training samples:")
    print("-" * 30)
    
    for i, text in enumerate(train_text):
        prediction, confidence = predict_sentiment(text)
        actual_label = train_labels[i]
        predicted_label = get_sentiment_label(prediction)
        actual_label_text = get_sentiment_label(actual_label)
        
        # Check if prediction is correct
        is_correct = prediction == actual_label
        if is_correct:
            correct_predictions += 1
            
        status = "✓" if is_correct else "✗"
        
        print(f"{status} Text: '{text}'")
        print(f"  Predicted: {predicted_label} (confidence: {confidence:.2f})")
        print(f"  Actual: {actual_label_text}")
        print()
    
    # Calculate and display accuracy
    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {correct_predictions}/{total_predictions} "
          f"({accuracy:.2%})")
    print("="*50)


# -------------------- 6. Interactive Demo -------------------- #

def interactive_demo():
    """Run an interactive demo where users can test their own text."""
    print("\n" + "="*50)
    print("SENTIMENT ANALYZER - INTERACTIVE DEMO")
    print("="*50)
    print("Enter text to analyze sentiment (or 'quit' to exit)")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nEnter text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Thank you for using the sentiment analyzer!")
                break
            
            if not user_input:
                print("Please enter some text to analyze.")
                continue
            
            # Analyze sentiment
            prediction, confidence = predict_sentiment(user_input)
            sentiment = get_sentiment_label(prediction)
            
            # Display results
            print(f"\nText: '{user_input}'")
            print(f"Sentiment: {sentiment}")
            print(f"Confidence: {confidence:.2%}")
            
            # Add emoji for fun
            emoji = "😊" if prediction == 1 else "😞"
            print(f"Result: {sentiment} {emoji}")
            
        except KeyboardInterrupt:
            print("\n\nExiting sentiment analyzer...")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")


def test_sample_texts():
    """Test the analyzer with some sample texts."""
    test_texts = [
        "I absolutely love this product!",
        "This is the worst experience ever.",
        "The movie was okay, nothing special.",
        "Amazing performance by the actors!",
        "I hate waiting in long queues.",
        "What a beautiful sunny day!"
    ]
    
    print("\n" + "="*50)
    print("TESTING WITH SAMPLE TEXTS")
    print("="*50)
    
    for text in test_texts:
        prediction, confidence = predict_sentiment(text)
        sentiment = get_sentiment_label(prediction)
        emoji = "😊" if prediction == 1 else "😞"
        
        print(f"Text: '{text}'")
        print(f"Sentiment: {sentiment} {emoji} (confidence: {confidence:.2%})")
        print("-" * 30)


# -------------------- 7. Main Execution -------------------- #

def main():
    """Main function to run the sentiment analyzer."""
    try:
        print("="*60)
        print("SENTIMENT ANALYZER")
        print("="*60)
        print("This program uses a Naive Bayes classifier to analyze")
        print("the sentiment of text as either Positive or Negative.")
        print("="*60)
        
        # Run evaluation
        evaluate_model()
        
        # Test with sample texts
        test_sample_texts()
        
        # Run interactive demo
        interactive_demo()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check that all required libraries are installed:")
        print("pip install scikit-learn")


if __name__ == "__main__":
    main()
