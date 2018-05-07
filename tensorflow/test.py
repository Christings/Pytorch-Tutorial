import gensim, re
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import matplotlib
# matplotlib.use("TkAgg")

# get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# text = open('4095_01.txt', 'rb', ).read()

stop_words = set(stopwords.words('english'))


def clean(word):
    word = word.strip()
    word = word.lower()
    word = re.sub('[^A-Za-z0-9]+', '', word)
    if word not in stop_words:
        return word
    else:
        return ''


# clean("king's")
# 'kings'

line_count = 0
sentences = []

with open('4095_01.txt', 'r', encoding='utf-8-sig') as inpFile:
    x = inpFile.readlines()
    for line in x:
        if line is not None or line != '\n':
            words = line.split()
            words = map(lambda x: clean(x), words)
            words = list(filter(lambda x: True if len(x) > 0 else False, words))
            sentences.append(words)
        print("111")

# sentences[100:110]


# In[ ]:

model = Word2Vec(sentences, window=5, size=500, workers=-1, min_count=5)
labels = []
tokens = []

for word in model.wv.vocab:
    print("111")
    tokens.append(model[word])
    labels.append(word)

# TSNE plot to find the similarity of words
tsne_model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
new_values = tsne_model.fit_transform(tokens)

x = []
y = []
for value in new_values:
    x.append(value[0])
    y.append(value[1])
plt.figure(figsize=(16, 12))
for i in range(100, 200):
    plt.scatter(x[i], y[i])
    plt.annotate(labels[i],
                 xy=(x[i], y[i]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
plt.show()
