import re
import string

def preprocess_title(text):
    text = text.lower()
    text = re.sub(r'\(\d{4}\)', '', text)  
    text = text.translate(str.maketrans('', '', string.punctuation))  
    text = re.sub(r'\s+', ' ', text).strip()
    return text