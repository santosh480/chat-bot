import random
import json
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download required NLTK data
nltk.download('punkt')

# Load training data from JSON
with open("train_data.json") as file:
    training_data = json.load(file)

# Initialize stemmer and data holders
stemmer = PorterStemmer()
training_sentences = []
intent_labels = []
intent_to_responses = {}

# Preprocess the training data
for intent in training_data['intents']:
    label = intent['tag']
    intent_to_responses[label] = intent['responses']
    for sentence in intent['patterns']:
        tokenized = nltk.word_tokenize(sentence.lower())
        stemmed = [stemmer.stem(word) for word in tokenized]
        processed_sentence = " ".join(stemmed)
        training_sentences.append(processed_sentence)
        intent_labels.append(label)

# Convert text data to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_sentences)
y = intent_labels

# Train the model
model = MultinomialNB()
model.fit(X, y)

# Function to generate chatbot response
def chatbot_response(user_message):
    tokens = nltk.word_tokenize(user_message.lower())
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    input_vector = vectorizer.transform([" ".join(stemmed_tokens)])
    predicted_intent = model.predict(input_vector)[0]
    return random.choice(intent_to_responses[predicted_intent])

# Chat loop
print("🤖 AI Chatbot: Hello! Type 'exit' to stop.")
while True:
    user_text = input("You: ")
    if user_text.lower() == "exit":
        print("🤖 Chatbot: Bye! Have a nice day.")
        break
    response = chatbot_response(user_text)
    print("🤖 Chatbot:", response)

