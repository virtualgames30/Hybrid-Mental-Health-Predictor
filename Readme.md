MENTAL HEALTH PREDICTION USING HYBRID MACHINE LEARNING TECHNIQUES

This project implements a hybrid Natural Language Processing (NLP) pipeline that analyzes mental health-related textual statements to detect the underlying mental health status. It leverages sentiment analysis, topic modeling (LDA), TF-IDF vectorization, and supervised machine learning classifiers (XGBoost, SVM, Logistic Regression).

Dataset
•	File: mental_health.xlsx
•	Target Column: status (categories include Depression, Anxiety, Normal, Suicidal, Stress)
•	Text Column: statement

Project Pipeline
1. Setup
•	Imports essential Python libraries including such as pandas, numpy, seaborn, matplotlib, sklearn, xgboost, gensim, plotly.
•	Downloads NLTK resources datasets like punkt (for tokenization), stopwords (for removing common words), wordnet (for lemmatization), and vader_lexicon (for sentiment analysis).

3. Data Loading
•	Loads the dataset using pandas.read_excel().
•	Drops unnecessary index column and missing statements.


5. Data Cleaning and  Preprocessing
A preprocess_text function is defined to perform the following
•	Lowercasing
•	Punctuation removal
•	Tokenization
•	Stopword removal
•	Lemmatization
The preprocess_text function is then applied to the 'statement' column of the dataset, creating a new 'cleaned_text' column.

6. Exploratory Data Analysis (EDA)
•	Distribution of mental health status classes using seaborn and plotly bar plots.
•	Word cloud visualization from preprocessed text.


8. Sentiment Analysis
•	SentimentIntensityAnalyzer (VADER) from nltk.sentiment is employed to compute the compound sentiment scores (polarity scores: negative, neutral, positive, compound) for each cleaned text.
•	The score is then stored in sentiment_score column.


10. Sentiment Score Visualization
•	Histogram and boxplots of sentiment scores across mental health statuses.


12. N-gram Analysis
•	Generates frequency counts of:
•	Unigrams
•	Bigrams
•	Trigrams
•	Quadgrams
•	Uses nltk and collections.Counter


14. Topic Modeling (LDA)
•	Additional preprocessing for LDA (removes numbers, special characters, contractions).
•	Uses CountVectorizer (with custom stopwords) to create input features.
•	Applies Latent Dirichlet Allocation (LDA) using sklearn.
•	Extracts topics and assigns topic labels based on the list of  word that appears in each topics. The following labels were assigned
        i.	        0: "Temporal Patterns & Pain Signals",
        ii.	        1: "Life Perspectives & Aspirations",
        iii.	    2: "Interpersonal Relationships & Emotional Support",
        iv.	        3: "Depressive Symptoms & Familial Impact",
        v.	        4: "Suicide & Self-Harm Ideation",
        vi.	        5: "Crisis & Intense Mental Health Struggles",
        vii.	    6: "Work Stress & Financial Pressures",
        viii        7: "Bipolar Experiences & Treatment"
•	Computes and visualizes coherence scores using gensim.


16. Topic Distribution Analysis
•	Uses plotly.express to visualize:
        o	Heatmap of topic-word importance
        o	Bar chart of topic distribution per status
        o	Radar chart comparing topic prevalence
        o	Faceted bar charts by topic
        o	Treemap of topic-word weights

    
18. First prediction system: Based LDA and VADER
•	Saves trained LDA model, TF-IDF, CountVectorizer, and LabelEncoder using joblib.
•	Create a MentalHealthPredictor class to:
        o	Load models
        o	Preprocess new texts
        o	Output: cleaned text, sentiment score, dominant topic & topic label

    
20. Hybrid Model Building
•	Combines features from:
        o	LDA topic probabilities
        o	VADER sentiment scores
        o	TF-IDF vectors
•	Merges into a single feature matrix (named- combined_x_features). This now serve as the new X-variable i.e independent features
•	Target: status column. Which is the Y-variable. i.e dependent feature / what we are aiming to predict


20.1. Classification Models
•	XGBoost:
        o	Encoded the target column and stored in “y_encoded”
        o	Created a seperate train_test_split for this model. Using combined_x_features and y_encoded
        o	Evaluated the performance using accuracy, precision, recall, F1-score
        o	Visualized top 20 feature importances
        o	Saved model: mentalhealth_xgboost.pkl
•	Support Vector Machine (SVC)
        o	Trained on combined_x_features
        o	Evaluated the performance using accuracy, precision, recall, F1-score
•	Logistic Regression
        o	Baseline model with evaluation metrics
        o	Evaluated the performance using accuracy, precision, recall, F1-score

21. Testing the Full Hybrid Pipeline
A MentalHealthHybridPredictor class is defined to encapsulate the entire prediction pipeline, making it easy to use the trained model for new text inputs.
•	Initialization: The class loads all the saved models and components.
•	Predict Method: This method takes new text as input, applies the same preprocessing and feature engineering steps (sentiment, TF-IDF, topic modeling), combines the features, and then uses the loaded XGBoost model to predict the mental health status. It returns a dictionary containing the original text, sentiment score, dominant topic, topic label, and predicted status

Example Usage
The if __name__ == "__main__": block provides an example of how to use the MentalHealthHybridPredictor class. It creates an instance of the predictor, and then tests it with several sample input texts, printing the prediction results for each.

Test on sample input

predictor = MentalHealthHybridPredictor()
input_text = "I feel lost and tired all the time"
result = predictor.predict(input_text)
print(result)

Batch testing with CSV

import pandas as pd
# Load new dataset
new_data = pd.read_csv('new_dataset.csv')['text_column'].tolist()
# Predict
results = predictor.predict_batch(new_data)
results.to_csv('prediction_output.csv', index=False)

Dependencies
•	nltk, gensim, xgboost, sklearn, pandas, numpy, seaborn, matplotlib, plotly, joblib, contractions, 

Author
Omowaye Joshua


