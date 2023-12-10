import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
#LoadGmailDataSet
data=pd.read_csv('C:\Users\DELL\Downloads\mail_data.csv')
#splitdatasetintotrainingandtestingset
train_data,test_data,train_labels,test_label=train_test_split(data['Message'],data['Category'],test_size=0.2,random_state=42)
#Createcountervectorizeroftextdata
vector=CountVectorizer()
train_vectors=vector.fit_transform(train_data)
test_vector=vector.transform(test_data)
print("printingtraindata")
print(train_vectors)
print("printingtestdata")
print(test_vector)
#TrainNAiveBayesalgorithm
classifier=MultinomialNB()
print("printingCalssification")
print(classifier.fit(train_vectors,train_labels))
#Makepredictionontestdata
print(test_label)
print(train_labels)
predictions=classifier.predict(test_vector)
print(predictions)
#Evaluatethemodel
accuracy=accuracy_score(test_label,predictions)
print(accuracy)
print('Accuracy:{:.2f}'.format(accuracy))

#Displayclassificationreport
print(classification_report(test_label,predictions))
