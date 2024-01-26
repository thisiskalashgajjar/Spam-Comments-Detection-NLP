# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.pipeline import Pipeline


# Load data into a Pandas DataFrame
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00380/YouTube-Spam-Collection-v1.zip"
# Download and extract the data from the UCI repository

# Assuming you've already downloaded and extracted the data, load the CSV file
data = pd.read_csv("/Users/kalashgajjar/Desktop/Spam Comments Detecntion_NLP/Youtube02-KatyPerry.csv")

# Basic Data Exploration
print(data.head())
print(data.info())

# Prepare Data for Model Building
X = data['CONTENT']  # Assuming 'CONTENT' is the column containing comments
y = data['CLASS']    # Assuming 'CLASS' is the column containing the class labels

# Text preprocessing using NLTK
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    words = word_tokenize(text)
    words = [ps.stem(word) for word in words if word.isalnum() and word.lower() not in stop_words]
    return ' '.join(words)

X = X.apply(preprocess_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Build a pipeline with CountVectorizer, TF-IDF, and Naive Bayes Classifier
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Cross-validate the model
cross_val_results = cross_val_score(model, X_train, y_train, cv=5)
print("Cross-validation results:", cross_val_results)
print("Mean Accuracy:", cross_val_results.mean())

# Test the model on the test data
y_pred = model.predict(X_test)

# Print confusion matrix and accuracy
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classify new comments
new_comments = [
    "I love Katy Perry's music!",
    "This video is amazing!",
    "Subscribe to my channel!",
    "Check out this cool website!",
    "Free iPhone giveaway!",
    "Click this link for a discount on sunglasses!"
]

new_comments = [preprocess_text(comment) for comment in new_comments]
new_predictions = model.predict(new_comments)

# Display the results
for comment, prediction in zip(new_comments, new_predictions):
    print(f"Comment: {comment}\nPrediction: {prediction}\n")

