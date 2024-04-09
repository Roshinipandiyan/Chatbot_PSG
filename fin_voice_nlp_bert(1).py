import csv
import os
import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import speech_recognition as sr
import pyttsx3

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Function to process natural language query and return 1-D embedding
def process_query(query):
    inputs = tokenizer(query, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze()  # Ensure the output is 1-D

# Preprocess dataset to generate 1-D embeddings for each question
def preprocess_dataset(file_name):
    question_embeddings = []
    questions = []
    answers = []
    if os.path.exists(file_name):
        with open(file_name, 'r', encoding='ISO-8859-1') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # Skip the header row
            for row in csv_reader:
                questions.append(row[0])
                answers.append(row[1])
                # Process each question to get 1-D embedding
                embeddings = process_query(row[0]).numpy()
                question_embeddings.append(embeddings)
    else:
        print(f"File '{file_name}' not found.")
    return questions, answers, question_embeddings

# Find the most similar question based on 1-D embeddings
def find_most_similar_question(query_embedding, question_embeddings, questions, answers):
    min_distance = float('inf')
    most_similar_index = -1
    for index, q_embedding in enumerate(question_embeddings):
        # Ensure both embeddings are 1-D before comparing
        distance = cosine(query_embedding, q_embedding)
        if distance < min_distance:
            min_distance = distance
            most_similar_index = index
    if most_similar_index != -1:
        return answers[most_similar_index]
    else:
        return None

# Function to convert text to speech
def speak_text(command):
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

# Initialize the recognizer
r = sr.Recognizer()

# Load your dataset and generate 1-D embeddings for questions
file_name = 'Questions_answers.csv'  # Adjust to your CSV file path
questions, answers, question_embeddings = preprocess_dataset(file_name)

# Function to process text input
def process_text_input(user_input):
    # Process the user's query to get 1-D embedding
    query_embedding = process_query(user_input).numpy()

    # Find the most similar question and its answer based on embeddings
    found_answer = find_most_similar_question(query_embedding, question_embeddings, questions, answers)

    # Display the answer
    if found_answer:
        print("Answer:", found_answer)
        speak_text("The answer is: " + found_answer)
    else:
        print("No answer found.")
        speak_text("Sorry, I couldn't find an answer.")

# Function to process voice input
def process_voice_input():
    try:
        print("Listening:")
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio = r.listen(source)
            text = r.recognize_google(audio).lower()
            print("You said:", text)
            speak_text("You said: " + text)
            process_text_input(text)  # Process the recognized text
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
    except sr.UnknownValueError:
        print("Unknown error occurred.")

# Save the trained BERT model
model_save_path = "bert_voice_model.pth"
torch.save(model.state_dict(), model_save_path)
print("Model saved at:", model_save_path)

# Ask for input only once
user_input = input("Type your query or type 'voice' to switch to voice input: ").lower()
if user_input == "voice":
    process_voice_input()  # Start voice input mode
else:
    process_text_input(user_input)  # Process text input
