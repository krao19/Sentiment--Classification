#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import random
import pickle
from nltk.tokenize import word_tokenize
import re
import os

from nltk.classify.scikitlearn import SklearnClassifier


# In[3]:


data = pd.read_csv("C:\\Users\\Keshav Rao\\Desktop\\Keshav\\Kaam\\cpp_tasks\\nlp+ocr\\data.csv")


# In[4]:


data.info()


# In[ ]:





# In[5]:


all_words = []
from nltk.corpus import stopwords
nltk.download('stopwords')
import re

stop_words = list(set(stopwords.words('english')))
allowed_word_types = ["J"]


# In[6]:


from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
#nltk.download('all')

from nltk import sent_tokenize
for t in data.review:
    
    clean = re.sub(r'[^(a-zA-Z)\s]','',t)
    
    tokenized = word_tokenize(clean)
    
    stop = [w for w in tokenized if not w in stop_words]
    
    docs = nltk.pos_tag(stop)
    
    for w in docs:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())


# In[11]:


words_freq = nltk.FreqDist(all_words)
type(words_freq)


# In[9]:


words_freq.plot(30, cumulative = False)
plt.show()


# In[13]:


print (len(all_words))


# In[14]:


words_freq.values()


# In[15]:


words_most = list(words_freq.keys())[:1000]
words_most


# In[ ]:





# In[16]:


documents = []

for r,s in zip(data.review, data.sentiment):
    documents.append((r,s))
    
documents


# In[17]:



def rev_features (document):
    words = word_tokenize(document)
    features = {}
    for w in words_most:
        features[w] = (w in words)
    return features

sets_features = [(rev_features(r), s) for (r,s) in documents]
random.shuffle(sets_features)


# In[18]:


sets_features[1]


# In[31]:


x = [x[0] for x in sets_features]
y = [y[1] for y in sets_features]

df = pd.DataFrame(list(zip(x,y)), columns = ['features', 'sent'])


# In[ ]:


new_doc = ' worst movie'
features_new = rev_features(new_doc)
model2.


# In[23]:


train = sets_features[:700]
test = sets_features[700:]


# In[34]:


model0 = nltk.NaiveBayesClassifier.train(train)
print("acc:", (nltk.classify.accuracy(model0, test)))
model0.show_most_informative_features(10)


# In[35]:


model0.predict(features_new)


# In[31]:


from sklearn.naive_bayes import MultinomialNB
model2 = SklearnClassifier(MultinomialNB())
model2 = model2.train(train)

print("acc:", (nltk.classify.accuracy(model2, test)))


# In[60]:


model3 = MultinomialNB()
#x1 = x.reshape(-1,1)
for i,j in zip(x,y):
    model3.fit(i, j)
    


# In[41]:


from sklearn.naive_bayes import MultinomialNB
model2 = SklearnClassifier(MultinomialNB())
model2 = model2.train(sets_features)


# In[40]:


new_doc = ' worst movie'
features_new = rev_features(new_doc)
model2.predict(X_test) 


# In[39]:


print(model2.predict(features_new))


# In[33]:


from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
model1 = SklearnClassifier(LogisticRegression())
model1 = model1.train(train)

print("acc:", (nltk.classify.accuracy(model1, test)))


# In[32]:


from sklearn.ensemble import RandomForestClassifier
model5 = RandomForestClassifier(n_estimators=100,bootstrap = True,max_features = 'sqrt')

model5.fit(df.features, df.sent)

