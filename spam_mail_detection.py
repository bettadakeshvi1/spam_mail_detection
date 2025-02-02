import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Load data
data = pd.read_csv("spam.csv", encoding='latin-1')

# Remove duplicates
data.drop_duplicates(inplace=True)

# Replace category labels
data['Category'] = data['Category'].replace({'ham': 'Not Spam', 'spam': 'Spam'})

# Balance dataset (if needed)
spam_count = data['Category'].value_counts()
print("Dataset distribution before balancing:\n", spam_count)

# Split into input (messages) and output (labels)
mess = data['Message']
cat = data['Category']

# Split into training and test sets (80-20 split)
mess_train, mess_test, cat_train, cat_test = train_test_split(mess, cat, test_size=0.2, random_state=1)

# Convert text to numerical features using CountVectorizer with bigrams
cv = CountVectorizer(ngram_range=(1, 2))  # Using bigrams to capture spam patterns
features_train = cv.fit_transform(mess_train)

# Create and train the model
model = MultinomialNB()
model.fit(features_train, cat_train)

# Evaluate the model
features_test = cv.transform(mess_test)
print("Model Accuracy:", model.score(features_test, cat_test))

# # Predict on a new spammy message manually
# new_message = ['Congratulation, you won a 5 crore lottery! Click here to claim now.']
# message_transformed = cv.transform(new_message)
# result = model.predict(message_transformed)
# print("Prediction:", result[0])  # Expected output: "Spam"

# building interface
def predict(message):
    input_message=cv.transform([message]).toarray()
    result=model.predict(input_message)
    return result

st.header('SPAM MAIL DETECTION')


output=predict('Congratulations,you won a lottery')
input_mess=st.text_input('enter mail here')
if st.button('validate'):
    output = predict(input_mess)  # This returns 'Spam' or 'Not Spam' as a string

if output == "Spam":
    st.error("🚨 This message is **Spam**!")
else:
    st.success("✅ This message is **Not Spam**.")

# finished







