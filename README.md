# Drug-Review-Analysis

Here’s how each task in our project was achieved:

## Sentiment Analysis:

Objective: Determine the sentiment (positive, negative, neutral) of drug reviews using NLTK's VADER sentiment analyzer.

Implementation:
+ Imported necessary libraries including nltk and nltk.sentiment.vader.
+ Initialized SentimentIntensityAnalyzer() from nltk.sentiment.vader.
+ Iterated through each review in the training dataset (train.review), calculated the compound sentiment score using sid.polarity_scores(i).get('compound'), and stored these scores in an array (sentiments).
+ Converted the list of sentiment scores (sentiments) into a numpy array (np.asarray(sentiments)).

## Rating Prediction:

Objective: Predict the rating of drugs based on textual reviews using machine learning models.

Implementation:
+ Loaded and prepared the dataset (drugsComTrain_raw.csv and drugsComTest_raw.csv) using Pandas (pd.read_csv()).
+ Explored and cleaned the data to handle missing values and basic text cleaning.
+ Utilized TF-IDF vectorization (TfidfVectorizer()) from sklearn.feature_extraction.text to convert text data into numerical features (X_train, X_test).
+ Implemented several classifiers (MultinomialNB, LogisticRegression, KNeighborsClassifier, RandomForestClassifier) from sklearn to train models on the TF-IDF vectorized data.
+ Evaluated model performance using accuracy metrics (accuracy_score), confusion matrices (confusion_matrix), and other metrics (precision_score, recall_score, f1_score).

## Insight Generation:

Objective: Gain insights into the relationships between drug ratings, conditions treated, and sentiment expressed in reviews.

Implementation:
+ Conducted exploratory data analysis (EDA) using Matplotlib (plt) and Seaborn (sns) to visualize distributions of ratings, conditions, and drug names.
+ Generated insights into popular conditions, common drugs, and relationships between ratings, useful counts, and sentiment using descriptive statistics and visualizations (e.g., histograms, scatter plots, heatmaps).

## Model Comparison:

Objective: Compare the performance of traditional machine learning models with a deep learning approach using Keras.

Implementation:

+ Designed a deep learning model using Keras Sequential API (Sequential(), Dense()) with multiple layers for classification of drug ratings.
+ Prepared text data for deep learning using one-hot encoding (CountVectorizer() or TfidfVectorizer()).
+ Trained the deep learning model (model.fit()) on the one-hot encoded text data and evaluated its performance using categorical cross-entropy loss (loss='categorical_crossentropy') and accuracy metrics (metrics=['accuracy']).
+ Monitored and visualized training progress over epochs using Matplotlib (plt.plot()).

## Visualization:

Objective: Create visualizations to better understand data distributions and model predictions.

Implementation:
+ Used Matplotlib (plt) and Seaborn (sns) to create various visualizations, including histograms, scatter plots, and heatmaps.
+ Visualized distributions of ratings, conditions, and drug names.
+ Plotted relationships between sentiment scores, ratings, and usefulness counts (plt.scatter()).
+ Displayed performance metrics such as confusion matrices (sns.heatmap()) to visualize model evaluation results.

Each task was achieved through systematic data handling, application of appropriate libraries and techniques, and thorough analysis to meet the project objectives effectively.





