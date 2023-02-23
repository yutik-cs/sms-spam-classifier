import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
import base64
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer


ps = PorterStemmer()

# function to add background image to the web app.
def add_bg(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# function for text processing
def transform_text(text):
    text = text.lower()  # lowercase
    text = nltk.word_tokenize(text)  # tokenization i.e. converting into words

    y = []
    for i in text:  # Removing special characters
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


add_bg('sms.jpg')
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("SMS Spam Classifier")
st.markdown('<label style="color:white; font-weight:bold; font-size:18px;">Enter the message</label>', unsafe_allow_html=True)
input_sms = st.text_area(" ")

if st.button('Predict'):
    if len(input_sms)>0:
        # 1. Preprocess
        transform_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transform_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display
        if result == 1:
            display = '<p style="font-family:Verdana;' \
                      'background-color:#242C2F; ' \
                      'height:45px; ' \
                      'border-radius:5px; ' \
                      'color:White; ' \
                      'font-size: 28px;' \
                      'text-align:center">' \
                      '<b>Classified as : </b> <b style="color:#C61B03; font-size:28px;">Spam</b></p>'
            st.markdown(display, unsafe_allow_html=True)
        elif result == 0:
            display = '<p style="font-family:Verdana;' \
                      'background-color:#242C2F; ' \
                      'height:45px; ' \
                      'border-radius:5px; ' \
                      'color:White; ' \
                      'font-size: 28px;' \
                      'text-align:center">' \
                      '<b>Classified as : </b> <b style="color:#1F9300; font-size:28px;">Not Spam</b></p>'
            st.markdown(display, unsafe_allow_html=True)
    else:
        st.header("")
