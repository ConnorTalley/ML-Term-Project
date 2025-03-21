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
        phrase_data.append({"Word/Phrase": phrase, "TF": tf_value, "IDF": idf.get(phrase, 0), "TF-IDF": tf_value * idf.get(phrase, 0), "Keyword Label": 1 if phrase in indexed_phrases else 0, "Document": chapter_name, "PMI": PMI_values[words_phrases.tolist().index(phrase)] if phrase in words_phrases else 0})


phrase_df = pd.DataFrame(phrase_data)
phrase_df = phrase_df.sort_values(by="Keyword Label", ascending=False)

scaler_pmi = MinMaxScaler()
phrase_df["PMI"] = scaler_pmi.fit_transform(phrase_df["PMI"].values.reshape(-1, 1)).flatten()

display(phrase_df.head(20))

X = phrase_df[["TF-IDF", "PMI"]].values
Y = phrase_df["Keyword Label"].values 

#X[:, 0] = np.log1p(X[:, 0])
scaler = MinMaxScaler()
X[:, 0] = scaler.fit_transform(X[:, 0].reshape(-1, 1)).flatten()
X[:, 1] = scaler.fit_transform(X[:, 1].reshape(-1, 1)).flatten()

knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
knn.fit(X, Y)

testing_folder_path = os.path.join(os.getcwd(), "Test Chapters")
Test_Chapters = sorted(glob.glob(os.path.join(testing_folder_path, "ch*.txt")))

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

test_phrase_data = []
for phrase, tf_value in zip(test_words_phrases, test_tf_values):
    test_phrase_data.append({"Word/Phrase": phrase, "TF": tf_value, "IDF": test_idf.get(phrase, 0), "TF-IDF": tf_value * test_idf.get(phrase, 0), "PMI": test_PMI_values[test_words_phrases.tolist().index(phrase)] if phrase in test_words_phrases else 0})

test_phrase_df = pd.DataFrame(test_phrase_data)

scaler_pmi = MinMaxScaler()
test_phrase_df["PMI"] = scaler_pmi.fit_transform(test_phrase_df["PMI"].values.reshape(-1, 1)).flatten()

X_test = test_phrase_df[["TF-IDF", "PMI"]].values

X_test[:, 0] = scaler.transform(X_test[:, 0].reshape(-1, 1)).flatten()
X_test[:, 1] = scaler.transform(X_test[:, 1].reshape(-1, 1)).flatten()

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

xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 300),  
    np.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, 300)
)

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
contour = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)

scatter = plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm, edgecolors="k", s=50)

legend_labels = {0: "Not Keyword (0)", 1: "Keyword (1)"}
handles = [plt.Line2D([0], [0], marker='o', color='w', markersize=10, markerfacecolor=plt.cm.coolwarm(i/1)) for i in range(2)]
plt.legend(handles, [legend_labels[i] for i in range(2)], title="Keyword Label")

plt.xlabel("TF-IDF (Term Frequency - Inverse Document Frequency)")
plt.ylabel("PMI (Pointwise Mutual Information)")
plt.title("kNN Classification (TF-IDF vs. PMI) with k=3")

plt.show()

