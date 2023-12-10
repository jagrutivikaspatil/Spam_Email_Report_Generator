import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
#Load Gmail DataSet
data=pd.read_csv('C:\Users\DELL\Downloads\mail_data.csv')
#split data set into training and testing set
train_data,test_data,train_labels,test_label=train_test_split(data['Message'],data['Category'],test_size=0.2,random_state=42)
#Create counter vectorizer of text data
vector=CountVectorizer()
train_vectors=vector.fit_transform(train_data)
test_vector=vector.transform(test_data)
print("printing train data")
print(train_vectors)
print("printing test data")
print(test_vector)
#TrainNAiveBayesalgorithm
classifier=MultinomialNB()
print("printing Calssification")
print(classifier.fit(train_vectors,train_labels))
#Make prediction on test data
print(test_label)
print(train_labels)
predictions=classifier.predict(test_vector)
print(predictions)
#Evaluate the model
accuracy=accuracy_score(test_label,predictions)
print(accuracy)
print('Accuracy:{:.2f}'.format(accuracy))
#Display classification report
print(classification_report(test_label,predictions))
