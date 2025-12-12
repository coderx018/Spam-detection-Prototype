import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Load dataset
data = pd.read_csv("/Users/bhuwanverma/Documents/python/Aiml Project Secondary/spam.csv", encoding='latin-1')

# Clean data
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

# Split data
mess = data['Message']
cat = data['Category']

mess_train, mess_test, cat_train, cat_test = train_test_split(mess, cat, test_size=0.2, random_state=42)

# Vectorization
cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

# Train model
model = MultinomialNB()
model.fit(features, cat_train)

# Prediction function
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result[0]

# Streamlit UI
st.title("📩 Spam Detection App")

input_mess = st.text_input("Enter a message to check:")

if st.button("Validate"):
    if input_mess.strip() == "":
        st.warning("Please enter a message before validating.")
    else:
        output = predict(input_mess)
        if output == "Spam":
            st.error("🚨 This message is likely Spam!")
        else:
            st.success("✅ This message is Not Spam.")
