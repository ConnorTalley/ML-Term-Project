from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import re
import os
import glob
from collections import Counter, defaultdict
import math
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import chi2
from mpl_toolkits.mplot3d import Axes3D


def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # Remove punctuation but keep spaces
    text = " ".join(text.split())  # Remove extra spaces
    return text


training_folder_path = os.path.join(os.getcwd(), "Train Chapters")

Train_Chapters = sorted(glob.glob(os.path.join(training_folder_path, "ch*.txt")))

chapter_texts = []
for file in Train_Chapters:
    with open(file, "r", encoding="utf-8") as f:
        chapter_texts.append(f.read().lower())

num_documents = len(chapter_texts)

preprocessed_texts = [preprocess_text(text) for text in chapter_texts]

vectorizer = CountVectorizer(ngram_range=(1,2), stop_words="english", token_pattern=r"(?u)\b[a-zA-Z]{2,}\b")
X_counts = vectorizer.fit_transform(chapter_texts)
X_counts = X_counts.sum(axis=0).A1
words_phrases = vectorizer.get_feature_names_out()

tf_values = X_counts / sum(X_counts)
# Show sample non-zero term frequencies
tf = [{words_phrases[i]: tf_values[i] for i in range(len(words_phrases))}]

print("🔹 TF Matrix Shape:", X_counts.shape)
print("🔹 Sample TF values:", list(tf[0].items())[:10])
print("🔹 Sample extracted words/phrases:", words_phrases[:20])

df = Counter()
for text in chapter_texts:
    phrases_in_doc = {phrase for phrase in words_phrases if phrase in text}
    df.update(phrases_in_doc)

idf = {phrase: math.log(num_documents / (df[phrase] + 1)) for phrase in words_phrases}

term_doc_counts = defaultdict(int)
for text in chapter_texts:
    words = set(text.split())
    for word in words:
        term_doc_counts[word] += 1

p_w = np.array([X_counts[i] / sum(X_counts) for i in range(len(words_phrases))])
p_d = 1 / num_documents
p_w_d = np.array([term_doc_counts[word] / num_documents for word in words_phrases])

PMI_values = np.log1p(p_w_d / (p_w * p_d))

def entropy(probs):
    return -np.sum([p * np.log2(p) for p in probs if p > 0])

# Build document presence matrix
phrase_entropy = {}
for phrase in words_phrases:
    presence = []
    for text in preprocessed_texts:
        presence.append(1 if phrase in text else 0)
    if sum(presence) > 0: 
        prob = np.array(presence) / sum(presence)
        phrase_entropy[phrase] = entropy(prob)
    else:
        phrase_entropy[phrase] = 0

Train_Index_folder_path = os.path.join(os.getcwd(), "Train Chapters\\Train_Index_by_chapters.txt")
with open(Train_Index_folder_path, "r", encoding="utf-8") as file:
    Train_Index = file.read().lower()

def normalize_phrase(phrase):
    phrase = phrase.lower().strip()  # Convert to lowercase and strip spaces
    phrase = re.sub(r"[^a-z0-9\s]", "", phrase)  # Remove punctuation (keep only words & spaces)
    phrase = " ".join(phrase.split())  # Normalize multiple spaces to single space
    return phrase

# Extract indexed phrases for each chapter and normalize them
indexed_phrases_by_chapter = {}
index_lines = Train_Index.split("\n")

current_chapter = None
for line in index_lines:
    line = line.strip().lower()
    
    if line.startswith("chapter "):  # Identify chapter headers
        current_chapter = f"ch{re.search(r'\d+', line).group()}.txt"
        indexed_phrases_by_chapter[current_chapter] = set()
    
    elif current_chapter and line:  # Add normalized phrases to the corresponding chapter
        indexed_phrases_by_chapter[current_chapter].add(normalize_phrase(line))

phrase_data = []
for chapter_idx, tf in enumerate(tf):
    chapter_name = f"ch{chapter_idx+1}.txt"
    indexed_phrases = indexed_phrases_by_chapter.get(chapter_name, set())

    for phrase, tf_value in tf.items():
        phrase_data.append({"Word/Phrase": phrase, "TF": tf_value, "IDF": idf.get(phrase, 0), "TF-IDF": tf_value * idf.get(phrase, 0), "Keyword Label": 1 if phrase in indexed_phrases else 0, "Document": chapter_name, "Entropy": phrase_entropy.get(phrase, 0), "PMI": PMI_values[words_phrases.tolist().index(phrase)] if phrase in words_phrases else 0})


phrase_df = pd.DataFrame(phrase_data)
phrase_df = phrase_df.sort_values(by="Keyword Label", ascending=False)

scaler_pmi = MinMaxScaler()
phrase_df["PMI"] = scaler_pmi.fit_transform(phrase_df["PMI"].values.reshape(-1, 1)).flatten()

display(phrase_df.head(20))

# Separate keyword and non-keyword entries
keyword_df = phrase_df[phrase_df["Keyword Label"] == 1]
non_keyword_df = phrase_df[phrase_df["Keyword Label"] == 0]

# Sample non-keywords to match keyword count
non_keyword_sampled = non_keyword_df.sample(n=len(keyword_df) * 5, random_state=10)

# Combine and shuffle the dataset
balanced_df = pd.concat([keyword_df, non_keyword_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

X = balanced_df[["TF-IDF", "Entropy", "PMI"]].values
Y = balanced_df["Keyword Label"].values 

#X[:, 0] = np.log1p(X[:, 0])
for i in range(X.shape[1]):
    X[:, i] = MinMaxScaler().fit_transform(X[:, i].reshape(-1, 1)).flatten()

print("creating knn classifier")
knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
knn.fit(X, Y)

testing_folder_path = os.path.join(os.getcwd(), "Test Chapters")
Test_Chapters = sorted(glob.glob(os.path.join(testing_folder_path, "ch*.txt")))

print("sorting test chapters")
test_chapter_texts = []
for file in Test_Chapters:
    with open(file, "r", encoding="utf-8") as f:
        test_chapter_texts.append(f.read().lower())

num_test_documents = len(test_chapter_texts)

preprocessed_test_texts = [preprocess_text(text) for text in test_chapter_texts]

test_vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words="english", token_pattern=r"(?u)\b[a-zA-Z]{2,}\b")

X_test_counts = test_vectorizer.fit_transform(test_chapter_texts)
X_test_counts = X_test_counts.sum(axis=0).A1
test_words_phrases = test_vectorizer.get_feature_names_out()

test_tf_values = X_test_counts / sum(X_test_counts)

test_term_doc_counts = defaultdict(int)
for text in test_chapter_texts:
    words = set(text.split())
    for word in words:
        test_term_doc_counts[word] += 1

test_p_w = np.array([X_test_counts[i] / sum(X_test_counts) for i in range(len(test_words_phrases))])
test_p_d = 1 / num_test_documents
test_p_w_d = np.array([test_term_doc_counts[word] / num_test_documents if word in test_term_doc_counts else 0 for word in test_words_phrases])
test_PMI_values = np.log1p(test_p_w_d / (test_p_w * p_d))

test_df = Counter()
for text in test_chapter_texts:
    phrases_in_doc = {phrase for phrase in test_words_phrases if phrase in text}
    test_df.update(phrases_in_doc)

test_idf = {phrase: math.log(num_test_documents / (test_df[phrase] + 1)) for phrase in test_words_phrases}


phrase_entropy = {}
for phrase in test_words_phrases:
    presence = []
    for text in test_chapter_texts:
        presence.append(1 if phrase in text else 0)
    if sum(presence) > 0: 
        prob = np.array(presence) / sum(presence)
        phrase_entropy[phrase] = entropy(prob)
    else:
        phrase_entropy[phrase] = 0

test_phrase_data = []
for phrase, tf_value in zip(test_words_phrases, test_tf_values):
    test_phrase_data.append({"Word/Phrase": phrase, "TF": tf_value, "IDF": test_idf.get(phrase, 0), "TF-IDF": tf_value * test_idf.get(phrase, 0), "Entropy": phrase_entropy.get(phrase, 0), "PMI": test_PMI_values[test_words_phrases.tolist().index(phrase)] if phrase in test_words_phrases else 0})

test_phrase_df = pd.DataFrame(test_phrase_data)


scaler_pmi = MinMaxScaler()
test_phrase_df["PMI"] = scaler_pmi.fit_transform(test_phrase_df["PMI"].values.reshape(-1, 1)).flatten()

X_test = test_phrase_df[["TF-IDF", "Entropy", "PMI"]].values

scalers = [MinMaxScaler().fit(X[:, i].reshape(-1, 1)) for i in range(X.shape[1])]
for i in range(X_test.shape[1]):
    X_test[:, i] = scalers[i].transform(X_test[:, i].reshape(-1, 1)).flatten()

print("Creating test predictions")
test_predictions = knn.predict(X_test)
test_phrase_df["Predicted Keyword Label"] = test_predictions

Test_Index_folder_path = os.path.join(os.getcwd(), "Test Chapters\\Test_Index_by_chapters.txt")
with open(Test_Index_folder_path, "r", encoding="utf-8") as file:
    Test_Index = file.read().lower()

indexed_test_phrases = {normalize_phrase(line) for line in Test_Index.split("\n") if line.strip()}

test_phrase_df["Actual Keyword Label"] = test_phrase_df["Word/Phrase"].apply(lambda x: 1 if x in indexed_test_phrases else 0)

y_true = test_phrase_df["Actual Keyword Label"].values
y_pred = test_phrase_df["Predicted Keyword Label"].values

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=1)
recall = recall_score(y_true, y_pred, zero_division=1)
f1 = f1_score(y_true, y_pred, zero_division=1)

print("🔹 Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

x_range = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 30)
y_range = np.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, 30)
z_range = np.linspace(X[:, 2].min() - 0.1, X[:, 2].max() + 0.1, 30)

xx, yy, zz = np.meshgrid(x_range, y_range, z_range)

# Flatten the grid and predict
grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
Z = knn.predict(grid_points)
Z = Z.reshape(xx.shape)

# 3D Scatter Plot of Training Data
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot original data points
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=plt.cm.coolwarm, edgecolor='k', s=40)

# Plot the decision boundary as scatter of predicted labels (optional: reduce for clarity)
ax.scatter(xx.ravel(), yy.ravel(), zz.ravel(), c=Z.ravel(), alpha=0.03, cmap=plt.cm.coolwarm)

ax.set_xlabel('TF')
ax.set_ylabel('IDF')
ax.set_zlabel('PMI')
ax.set_title("3D kNN Classification Surface (TF, IDF, PMI)")

plt.show()