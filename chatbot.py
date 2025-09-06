import nltk

# Auto-download if not already installed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

import random
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data (only needed once)
nltk.download('punkt')
nltk.download('wordnet')

# Sample corpus - Replace with your business FAQ or knowledge base
corpus = """
Hello, how can I help you?
What are your business hours?
We are open from 9 AM to 6 PM, Monday to Friday.
Where are you located?
Our office is in Bangalore, India.
How do I reset my password?
Click on the 'Forgot Password' link on the login page and follow instructions.
Do you offer technical support?
Yes, we offer 24/7 technical support.
How can I contact customer care?
You can call our support line or email us at support@example.com.
"""

# Preprocessing
sent_tokens = nltk.sent_tokenize(corpus.lower())  # Convert to lower-case sentences
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Greeting inputs/outputs
greet_inputs = ("hello", "hi", "greetings", "sup", "what's up", "hey")
greet_responses = ["Hi there!", "Hello!", "Greetings!", "Hey! How can I assist you?"]

def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)

# Generate response using TF-IDF
def generate_response(user_input):
    user_input = user_input.lower()
    sent_tokens.append(user_input)

    tfidf_vec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = tfidf_vec.fit_transform(sent_tokens)

    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = vals.argsort()[0][-1]
    flat = vals.flatten()
    flat.sort()
    score = flat[-1]

    sent_tokens.pop()

    if score == 0:
        return "I'm sorry, I don't understand that. Could you rephrase?"
    else:
        return sent_tokens[idx]

# Chat loop
def chatbot():
    print("BOT: Hello! I'm your AI assistant. Type 'bye' to exit.")
    while True:
        user_input = input("YOU: ")
        if user_input.lower() == 'bye':
            print("BOT: Goodbye! Have a nice day.")
            break
        elif greet(user_input) is not None:
            print(f"BOT: {greet(user_input)}")
        else:
            print(f"BOT: {generate_response(user_input)}")

# Run the chatbot
if __name__ == "__main__":
    chatbot()
