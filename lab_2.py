import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import plotly.express as px
from plotly import graph_objects as go
import pandas as pd


ignore_files = set(["LICENSE.txt"])

color_dict = {"Unknown":"grey", "Madison":"magenta", "Hamilton":"cyan", "Jay":"blue"}

authors = []
titles = []
texts = []

for root, dirs, files in os.walk('fedpapers'):
    files = [f for f in files if f not in ignore_files]
    files = [f for f in files if f[0] != "."] 
    for f in files:
        with open(os.path.join(root, f),'r', encoding='utf8') as rf:
            text = rf.read()
        
        texts.append(text.lower())
        authors.append(f[:-4].split("_")[1])
        titles.append(f[:-4])

print(authors)
# calculates occurance of term divided by length of document (then normalized)
vectorizer = TfidfVectorizer(use_idf=False, max_features=100, analyzer="word", ngram_range=(1,1))

# fit()
# transform()
# fit_transform()
counts = vectorizer.fit_transform(texts)

# change counts to a dense matrix
counts = counts.toarray()

pca = PCA(n_components=2)
my_pca = pca.fit_transform(counts)

print(my_pca)

vocab = vectorizer.vocabulary_
# print(vocab)

loadings = pca.components_

# print(loadings)

df = pd.DataFrame({"pc1":my_pca[:,0], "pc2":my_pca[:,1], "authors":authors})

loadings_df = pd.DataFrame({"vocab":list(vocab), "pc1":loadings[0], "pc2":loadings[1]})


fig = px.scatter(df, x="pc1", y="pc2", color="authors")
fig.add_trace(go.Scatter(x=loadings_df["pc1"], y=loadings_df["pc2"], text=loadings_df["vocab"], mode="text"))
fig.show()