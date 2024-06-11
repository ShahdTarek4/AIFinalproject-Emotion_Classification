#shahd tarek-202202018
#mira emad-202200319
#alya mohamed-202202900
#aliaa haggag-202202026
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

data = pd.read_csv("Emotion_classify_Data.csv")
x = data['Comment']
y = data['Emotion']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
vectorizer = CountVectorizer().fit(x_train)
text_train_vectorized = vectorizer.transform(x_train)
text_test_vectorized = vectorizer.transform(x_test)
clfr = MultinomialNB()
clfr.fit(text_train_vectorized, y_train)
predicted = clfr.predict(text_test_vectorized)

st.title("Emotions Classification App")
st.image("emotion_picture.png", use_column_width=True)
st.text('Model Description: Naive Bayes Model, trained on emotions classification data')
st.text('Anger, joy, or fear')

text = st.text_input("Enter Text Here", "Type Here...")
predict = st.button('Predict')

if predict:
    new_test_data = vectorizer.transform([text])
    predicted_label = clfr.predict(new_test_data)[0]
    if predicted_label == 'fear':
        prediction_text = "Fearful"
        st.success(f"'{text}' is classified as {prediction_text}")
    elif predicted_label == 'anger':
        prediction_text = "Angry"
        st.success(f"'{text}' is classified as {prediction_text}")
    elif predicted_label == 'joy':
        prediction_text = "Joyful"
        st.success(f"'{text}' is classified as {prediction_text}")


