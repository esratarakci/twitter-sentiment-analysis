import re 
import pandas as pd 
pd.set_option("display.max_colwidth", 200)
import numpy as np 
import string
import nltk 
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

train  = pd.read_csv('train_E6oV3lV.csv')
test = pd.read_csv('test_tweets_anuFYb8.csv')

train[train['label'] == 0].head(10)
train[train['label'] == 1].head(10)

train.shape, test.shape
train["label"].value_counts()

combi = train.append(test, ignore_index=True,sort=True)
combi.shape

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt

combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*") 
combi.head()

combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")

combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
combi.head()

tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) 

stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 
tokenized_tweet.head()

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    
combi['tidy_tweet'] = tokenized_tweet

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])
tfidf.shape

train_tfidf = tfidf[:31962,:]
test_tfidf = tfidf[31962:,:]
xtrain_tfidf, xvalid_tfidf, ytrain, yvalid = train_test_split(train_tfidf, train['label'],  
                                                          random_state=42, 
                                                          test_size=0.3)
svc = svm.SVC(kernel='linear', C=1, probability=True).fit(xtrain_tfidf, ytrain)
prediction = svc.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)
f1_score(yvalid, prediction_int)

test_pred = svc.predict_proba(test_tfidf)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id','label']]
submission.to_csv('predicted_test.csv', index=False)

positive = 0
negative = 0
data = pd.read_csv('predicted_test.csv')
#data = submission
#row = data.label.tolist()
lines = data.label
    
for i in lines:
    if(lines[i]==1):
        negative+=1
    if(lines[i]==0):
         positive+=1
#for i in range(len(row)):
  #  if (row[i] == 0):
  #      positive+=1
    #elif(row[i] == 1):
      #  negative+=1
        
labels=['Positive [' +str(positive)+'%]' , 'Negative [' +str(negative)+'%]']
sizes=[positive,negative]
colors=['pink','yellow']
patches,texts=plt.pie(sizes, colors=colors,startangle=90)
plt.legend(patches,labels,loc="best")
plt.axis("equal")
plt.tight_layout()
plt.show();

