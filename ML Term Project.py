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
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
import yake


data1 = "YAKE_Score"
data2 = "Position_Score"
data3 = "PMI"
data4 = "TF-IDF"
data5 = "NB_Prob"
data6 = "Entropy"
k = 7

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

print("TF Matrix Shape:", X_counts.shape)
print("Sample TF values:", list(tf[0].items())[:10])
print("Sample extracted words/phrases:", words_phrases[:20])

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



phrase_first_pos = []

# Flatten chapter_texts to align with chapter_name references
chapter_lookup = {f"ch{i+1}.txt": text for i, text in enumerate(chapter_texts)}

for _, row in phrase_df.iterrows():
    phrase = row["Word/Phrase"]
    chapter_name = row["Document"]
    chapter_text = chapter_lookup.get(chapter_name, "")

    index = chapter_text.find(phrase)
    if index != -1:
        normalized_position = 1 - (index / len(chapter_text))  # earlier = higher
    else:
        normalized_position = 0  # not found, treat as least important

    phrase_first_pos.append(normalized_position)

phrase_df["Position_Score"] = phrase_first_pos

# --- NB STEP 1: Vectorize phrases using TF-IDF ---
train_phrases = phrase_df["Word/Phrase"].tolist()
train_labels = phrase_df["Keyword Label"].values

vectorizer_nb = CountVectorizer(
    ngram_range=(1, 2),
    stop_words='english',
    token_pattern=r"(?u)\b[a-zA-Z]{2,}\b"
)

X_train_nb = vectorizer_nb.fit_transform(train_phrases)

# --- NB STEP 2: Out-of-fold NB probability for training set ---
oof_probs = np.zeros(len(train_phrases))
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kf.split(X_train_nb, train_labels):
    X_tr, X_val = X_train_nb[train_idx], X_train_nb[val_idx]
    y_tr = train_labels[train_idx]

    nb_model = MultinomialNB()
    nb_model.fit(X_tr, y_tr)

    oof_probs[val_idx] = nb_model.predict_proba(X_val)[:, 1]  # probability of being a keyword

# Add NB score to phrase_df
phrase_df["NB_Prob"] = oof_probs

# Configure YAKE
yake_extractor = yake.KeywordExtractor(
    lan="en", 
    n=2,  # same as your ngram_range
    dedupLim=0.9, 
    top=100000, 
    features=None
)

# --- Score training phrases with YAKE ---
yake_scores_train = []
yake_cache = {}

print("Obtaining Yake! Scores")

for _, row in phrase_df.iterrows():
    phrase = row["Word/Phrase"]
    chapter = row["Document"]
    chapter_text = chapter_lookup.get(chapter, "")

    if chapter not in yake_cache:
        yake_cache[chapter] = dict(yake_extractor.extract_keywords(chapter_text))

    score = yake_cache[chapter].get(phrase, 1.0)
    yake_scores_train.append(score)

phrase_df["YAKE_Score"] = yake_scores_train

display(phrase_df.head(20))

# Separate keyword and non-keyword entries
keyword_df = phrase_df[phrase_df["Keyword Label"] == 1]
non_keyword_df = phrase_df[phrase_df["Keyword Label"] == 0]

# Sample non-keywords to match keyword count
non_keyword_sampled = non_keyword_df.sample(n=len(keyword_df), random_state=10)

# Combine and shuffle the dataset
balanced_df = pd.concat([keyword_df, non_keyword_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

X = balanced_df[[data1, data5]].values
Y = balanced_df["Keyword Label"].values 

#X[:, 0] = np.log1p(X[:, 0])
for i in range(X.shape[1]):
    X[:, i] = MinMaxScaler().fit_transform(X[:, i].reshape(-1, 1)).flatten()

print("creating knn classifier")
knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
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

test_phrase_first_pos = []

# Create lookup for test chapter content
test_chapter_lookup = {os.path.basename(f): text for f, text in zip(Test_Chapters, test_chapter_texts)}

# Assign Document column if not yet assigned
if "Document" not in test_phrase_df.columns:
    doc_names = [os.path.basename(f) for f in Test_Chapters]
    chunk_size = len(test_phrase_df) // len(doc_names)
    test_phrase_df["Document"] = [doc_names[min(i // chunk_size, len(doc_names) - 1)] for i in range(len(test_phrase_df))]

for _, row in test_phrase_df.iterrows():
    phrase = row["Word/Phrase"]
    chapter_name = row["Document"]
    chapter_text = test_chapter_lookup.get(chapter_name, "")

    index = chapter_text.find(phrase)
    if index != -1:
        normalized_position = 1 - (index / len(chapter_text))  # earlier = higher
    else:
        normalized_position = 0  # not found

    test_phrase_first_pos.append(normalized_position)

test_phrase_df["Position_Score"] = test_phrase_first_pos

yake_scores_test = []

for _, row in test_phrase_df.iterrows():
    phrase = row["Word/Phrase"]
    chapter = row["Document"]
    chapter_text = test_chapter_lookup.get(chapter, "")

    if chapter not in yake_cache:
        yake_cache[chapter] = dict(yake_extractor.extract_keywords(chapter_text))

    score = yake_cache[chapter].get(phrase, 1.0)
    yake_scores_test.append(score)

test_phrase_df["YAKE_Score"] = yake_scores_test

scaler_pmi = MinMaxScaler()
test_phrase_df["PMI"] = scaler_pmi.fit_transform(test_phrase_df["PMI"].values.reshape(-1, 1)).flatten()

nb_final = MultinomialNB()
nb_final.fit(X_train_nb, train_labels)

test_phrases = test_phrase_df["Word/Phrase"].tolist()
X_test_nb = vectorizer_nb.transform(test_phrases)
test_nb_probs = nb_final.predict_proba(X_test_nb)[:, 1]

# Add NB score to test_phrase_df
test_phrase_df["NB_Prob"] = test_nb_probs

Test_Index_folder_path = os.path.join(os.getcwd(), "Test Chapters\\Test_Index_by_chapters.txt")
with open(Test_Index_folder_path, "r", encoding="utf-8") as file:
    Test_Index = file.read().lower()

indexed_test_phrases = {normalize_phrase(line) for line in Test_Index.split("\n") if line.strip()}

test_phrase_df["Actual Keyword Label"] = test_phrase_df["Word/Phrase"].apply(lambda x: 1 if x in indexed_test_phrases else 0)


# Step 1: Add NB predicted labels based on threshold (default 0.5)
test_phrase_df["NB_Prediction"] = (test_phrase_df["NB_Prob"] > 0.3).astype(int)

# Step 2: Compare with actual keyword labels
y_true_nb = test_phrase_df["Actual Keyword Label"].values
y_pred_nb = test_phrase_df["NB_Prediction"].values

# Step 3: Calculate metrics
accuracy_nb = accuracy_score(y_true_nb, y_pred_nb)
precision_nb = precision_score(y_true_nb, y_pred_nb, zero_division=1)
recall_nb = recall_score(y_true_nb, y_pred_nb, zero_division=1)
f1_nb = f1_score(y_true_nb, y_pred_nb, zero_division=1)

# Step 4: Print results
print("\nEvaluation Metrics for Naïve Bayes Classifier Only:")
print(f"Accuracy:  {accuracy_nb:.4f}")
print(f"Precision: {precision_nb:.4f}")
print(f"Recall:    {recall_nb:.4f}")
print(f"F1 Score:  {f1_nb:.4f}")

X_test = test_phrase_df[[data1, data5]].values

for i in range(X_test.shape[1]):
    X_test[:, i] = MinMaxScaler().fit_transform(X_test[:, i].reshape(-1, 1)).flatten()

print("Creating test predictions")
test_predictions = knn.predict(X_test)
test_phrase_df["Predicted Keyword Label"] = test_predictions

y_true = test_phrase_df["Actual Keyword Label"].values
y_pred = test_phrase_df["Predicted Keyword Label"].values

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=1)
recall = recall_score(y_true, y_pred, zero_division=1)
f1 = f1_score(y_true, y_pred, zero_division=1)

print("Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# x_range = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 30)
# y_range = np.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, 30)
# z_range = np.linspace(X[:, 2].min() - 0.1, X[:, 2].max() + 0.1, 30)

# xx, yy, zz = np.meshgrid(x_range, y_range, z_range)

# # Flatten the grid and predict
# grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
# Z = knn.predict(grid_points)
# Z = Z.reshape(xx.shape)

# # 3D Scatter Plot of Training Data
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot original data points
# scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=plt.cm.coolwarm, edgecolor='k', s=40)

# # Plot the decision boundary as scatter of predicted labels (optional: reduce for clarity)
# ax.scatter(xx.ravel(), yy.ravel(), zz.ravel(), c=Z.ravel(), alpha=0.03, cmap=plt.cm.coolwarm)

# ax.set_xlabel(data1)
# ax.set_ylabel(data2)
# ax.set_zlabel(data3)
# ax.set_title(f"3D kNN Classification Surface ({data1}, {data2}, {data3}) Where k = {k}")

# plt.show()
# STEP 1: Compute confidence from inverse kNN distances
distances, _ = knn.kneighbors(X_test)
confidence_scores = 1 / (np.mean(distances, axis=1) + 1e-5)
test_phrase_df["Confidence"] = confidence_scores
test_phrase_df["Normalized Confidence"] = MinMaxScaler().fit_transform(confidence_scores.reshape(-1, 1)).flatten()

# STEP 2: Assign Document if missing
if "Document" not in test_phrase_df.columns:
    doc_names = [os.path.basename(f) for f in Test_Chapters]
    chunk_size = len(test_phrase_df) // len(doc_names)
    test_phrase_df["Document"] = [doc_names[min(i // chunk_size, len(doc_names) - 1)] for i in range(len(test_phrase_df))]

# STEP 3: Select top 15 keywords per chapter
top_keywords_df = (
    test_phrase_df[test_phrase_df["Predicted Keyword Label"] == 1]
    .groupby("Document")
    .apply(lambda df: df.sort_values("Normalized Confidence", ascending=False).head(15))
    .reset_index(drop=True)
)

# STEP 4: Build concept map using networkx
# Rebuild graph with unique keywords only
concept_graph = nx.Graph()
concept_graph.add_node("CONCEPT MAP", color='gold', size=1500)

# Step 1: Add unique keyword nodes with style
keyword_set = set()
for _, row in top_keywords_df.iterrows():
    phrase = row["Word/Phrase"]
    conf = row["Normalized Confidence"]
    
    if phrase not in keyword_set:
        if conf > 0.8:
            color = "lightcoral"
            size = 1000
        elif conf > 0.6:
            color = "lightblue"
            size = 800
        else:
            color = "lightgreen"
            size = 600

        concept_graph.add_node(phrase, color=color, size=size)
        concept_graph.add_edge("CONCEPT MAP", phrase)
        keyword_set.add(phrase)

# Step 2: Connect keywords based on co-occurrence per chapter
grouped_keywords = top_keywords_df.groupby("Document")["Word/Phrase"].apply(list)

for doc_name, keyword_list in grouped_keywords.items():
    for i in range(len(keyword_list)):
        for j in range(i + 1, len(keyword_list)):
            concept_graph.add_edge(keyword_list[i], keyword_list[j])

# (Optional) Step 3: Connect keyword to chapter group nodes (for traceability)
# This adds chapter nodes and connects them to the keywords from that chapter
add_chapter_nodes = False  # Set to True if you want visible chapter clusters
if add_chapter_nodes:
    for doc_name, keyword_list in grouped_keywords.items():
        chapter_node = f"[{doc_name}]"
        concept_graph.add_node(chapter_node, shape='diamond', color='gray', size=500)
        for phrase in keyword_list:
            concept_graph.add_edge(chapter_node, phrase)

# Step 4: Draw final concept map
plt.figure(figsize=(16, 12))
pos = nx.spring_layout(concept_graph, k=0.8)

colors = [concept_graph.nodes[n].get('color', 'gray') for n in concept_graph.nodes()]
sizes = [concept_graph.nodes[n].get('size', 500) for n in concept_graph.nodes()]

nx.draw(concept_graph, pos,
        with_labels=True,
        node_color=colors,
        node_size=sizes,
        edge_color='gray',
        font_size=8,
        font_weight='bold',
        width=1.2)

plt.title("Concept Map (Unique Keywords Across Chapters)", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.savefig("concept_map_styled.png", dpi=300)
plt.show()

y_true_concept = top_keywords_df["Actual Keyword Label"].values
y_pred_concept = top_keywords_df["Predicted Keyword Label"].values  # should all be 1s

accuracy_concept = accuracy_score(y_true_concept, y_pred_concept)
precision_concept = precision_score(y_true_concept, y_pred_concept, zero_division=1)
recall_concept = recall_score(y_true_concept, y_pred_concept, zero_division=1)
f1_concept = f1_score(y_true_concept, y_pred_concept, zero_division=1)

print("\nConcept Map Evaluation Metrics:")
print(f"Accuracy (Concept Map Keywords): {accuracy_concept:.4f}")
print(f"Precision: {precision_concept:.4f}")
print(f"Recall: {recall_concept:.4f}")
print(f"F1 Score: {f1_concept:.4f}")