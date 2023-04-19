import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.cluster.hierarchy import linkage, dendrogram

import matplotlib.pyplot as plt

ignore_files = set(["LICENSE.txt", ".DS_Store",".git_ignore"])

color_dict = {"Unknown":"grey", "Madison":"magenta", "Hamilton":"cyan", "Jay":"blue"}

authors = []
titles = []
texts = []
for root, dirs, files in os.walk('fedpapers'):
    files = [f for f in files if f not in ignore_files]
    for f in files:
        with open(os.path.join(root, f),'r', encoding='utf8') as rf:
            text = rf.read()

        texts.append(text.lower())
        authors.append(f[:-4].split("_")[1])
        titles.append(f[:-4])

print(authors)
# calculates occurance of term divided by length of document (then normalized)
vectorizer = TfidfVectorizer(use_idf=False, max_features=100, analyzer="word", ngram_range=(1,1))
counts = vectorizer.fit_transform(texts)

similarity = cosine_similarity(counts)

# cluster the documents
linkages = linkage(similarity, 'ward')

# viz with scipy
dendrogram(linkages, labels=authors, orientation="right", leaf_font_size=8, leaf_rotation=45)
plt.tick_params(axis="x", which='both', bottom=False, top=False, labelbottom=False)
plt.title('Dendrogram of Federalist Papers')
plt.tight_layout()

ax = plt.gca()
labels = ax.get_ymajorticklabels()
for label in labels:
    label.set_color(color_dict[label.get_text()])


plt.show()