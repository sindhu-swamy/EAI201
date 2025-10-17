import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# --- Task 1: Load and Explore Data ---

# Load the SMS Spam Collection dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[['v1', 'v2']]
df.columns = ['label', 'message']

# Dataset shape and columns
print("Dataset shape:", df.shape)
print("Columns:", df.columns)

# Count spam and ham messages
spam_count = df[df['label'] == 'spam'].shape[0]
ham_count = df[df['label'] == 'ham'].shape[0]
total_count = df.shape[0]
spam_percent = (spam_count / total_count) * 100

print(f"Spam messages: {spam_count}, Ham messages: {ham_count}")
print(f"Spam percentage: {spam_percent:.2f}%")

# Display 3 spam and 3 ham examples
print("\nSample Spam Messages:")
print(df[df['label']=='spam']['message'].head(3))
print("\nSample Ham Messages:")
print(df[df['label']=='ham']['message'].head(3))


# --- Task 2: Bayes Theorem ---

# Prior probabilities
P_spam = spam_count / total_count
P_ham = ham_count / total_count
print(f"\nP(spam): {P_spam:.3f}, P(ham): {P_ham:.3f}")

# Split data for training/testing
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Count "free" in spam and ham training messages
def word_count(messages, word):
    return sum(word.lower() in message.lower() for message in messages)

free_spam = word_count(train[train['label']=='spam']['message'], 'free')
free_ham = word_count(train[train['label']=='ham']['message'], 'free')
spam_msgs = train[train['label']=='spam'].shape[0]
ham_msgs = train[train['label']=='ham'].shape[0]
P_free_given_spam = free_spam / spam_msgs
P_free_given_ham = free_ham / ham_msgs

print(f"P('free'|spam): {P_free_given_spam:.3f}")
print(f"P('free'|ham): {P_free_given_ham:.3f}")

# --- Task 3: Build Naive Bayes Classifier ---

# Text to features
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train['message'])
X_test = vectorizer.transform(test['message'])
y_train = train['label']
y_test = test['label']

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy: {accuracy:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
print("Confusion Matrix:\n", cm)

# --- Task 4: Analyze Feature Importance ---

# Feature names and spam class word probabilities
feature_names = vectorizer.get_feature_names_out()
spam_word_logprobs = model.feature_log_prob_[1]

# Top 5 spam indicator words
top5_indices = np.argsort(spam_word_logprobs)[-5:]
top5_words = feature_names[top5_indices]
print("\nTop 5 spam-indicator words:", top5_words)

# Test classifier on custom messages
custom_messages = ["Call me", "Free call", "Win money!"]
custom_vectors = vectorizer.transform(custom_messages)
preds = model.predict(custom_vectors)
print("Custom message classifications:")
for msg, label in zip(custom_messages, preds):
    print(f"'{msg}' -> {label}")

print("Words with highest spam probability ratios:", top5_words)

# --- Task 5: Test the 'Naive' Assumption ---

# Sample messages
sample1 = "Call me"
sample2 = "Free call"
samples = [sample1, sample2]
vec_samples = vectorizer.transform(samples)
sample_probs = model.predict_proba(vec_samples)

for i, s in enumerate(samples):
    print(f"'{s}' - Spam probability: {sample_probs[i][1]:.3f}, Ham probability: {sample_probs[i][0]:.3f}")

print("""
Even though words in messages aren't truly independent, Naive Bayes works well for spam detection because strong indicator words (like 'free', 'win', 'money') commonly appear in spam, allowing robust separation of classes.
""")
