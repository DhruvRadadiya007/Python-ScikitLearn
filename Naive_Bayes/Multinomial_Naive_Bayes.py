import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
message = [
"win money now",
"win a free prize",
"free money offer",
"meeting at office",
"project meeting today",
"office project update"]

labels = [
"spam",
"Spam",
"Spam",
"Not Spam",
"Not Spam",
"Not Spam"]

vectorizer = CountVectorizer() # creating a vector for word counts 

x = vectorizer.fit_transform(message)
print(x)
print(vectorizer.get_feature_names_out())
print(x.toarray())

model = MultinomialNB()
model.fit(x,labels)

new_message = ["free cash prize,win upto 50rs cash "]
new_X = vectorizer.transform(new_message)
print(model.predict(new_X))
    