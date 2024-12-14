import spacy
import pandas as pd
import requests
from transformers import pipeline
import streamlit as st

from pro import chunking, dependency_parsing, fetch_books_from_dataset, fetch_books_from_google, fetch_books_from_open_library, named_entity_recognition, pos_tagging, semantic_verification, syntax_verification

# Load SpaCy's small English model
nlp = spacy.load('en_core_web_sm')

# Load BERT-based models for semantic tasks
semantic_model = pipeline("text-classification", model="distilbert-base-uncased")
qa_model = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Load your dataset
df = pd.read_csv('BooksDatasetClean.csv')

# Function definitions remain the same as before for syntax verification, semantic verification, etc.

# Adding Question Answering using BERT
def answer_question(question, context):
    result = qa_model(question=question, context=context)
    return result['answer']

# Adding Sentiment Analysis using BERT
def sentiment_analysis(text):
    result = sentiment_model(text)[0]
    return f"Label: {result['label']}, Score: {result['score']:.2f}"

# Streamlit Interface
def main():
    st.title("NLP Exam Preparation Chatbot with BERT Enhancements")

    user_input = st.text_input("Ask me a question or input a sentence:")

    if st.button("Analyze"):
        if user_input:
            # Step 1: Syntax Verification
            syntax_feedback = syntax_verification(user_input)
            st.subheader("Syntax Verification")
            st.write(syntax_feedback)

            # Step 2: Semantic Verification
            semantic_feedback = semantic_verification(user_input)
            st.subheader("Semantic Verification")
            st.write(semantic_feedback)

            # Step 3: POS Tagging
            pos_tags = pos_tagging(user_input)
            st.subheader("Part-of-Speech Tagging")
            for word, tag in pos_tags:
                st.write(f"{word}: {tag}")

            # Step 4: Named Entity Recognition
            entities = named_entity_recognition(user_input)
            st.subheader("Named Entity Recognition")
            for entity, label in entities:
                st.write(f"{entity}: {label}")

            # Step 5: Chunking
            chunks = chunking(user_input)
            st.subheader("Chunking")
            for chunk, dep in chunks:
                st.write(f"{chunk}: {dep}")

            # Step 6: Dependency Parsing
            dependencies = dependency_parsing(user_input)
            st.subheader("Dependency Parsing")
            for word, dep, head in dependencies:
                st.write(f"{word}: {dep} → {head}")

            # Step 7: Sentiment Analysis
            sentiment_feedback = sentiment_analysis(user_input)
            st.subheader("Sentiment Analysis")
            st.write(sentiment_feedback)

            # Step 8: Book Recommendations from Dataset
            st.subheader("Book Recommendations from Dataset")
            books_from_dataset = fetch_books_from_dataset(user_input)
            for book in books_from_dataset:
                if isinstance(book, dict):
                    st.write(f"Title: {book['title']}, Author: {book['author']}")
                else:
                    st.write(book)

            # Step 9: Book Recommendations from Google Books API
            st.subheader("Book Recommendations from Google Books API")
            books_from_google = fetch_books_from_google(user_input)
            for book in books_from_google:
                if isinstance(book, dict):
                    st.write(f"Title: {book['title']}, Author: {book['author']}")
                else:
                    st.write(book)

            # Step 10: Book Recommendations from Open Library API
            st.subheader("Book Recommendations from Open Library API")
            books_from_open_library = fetch_books_from_open_library(user_input)
            for book in books_from_open_library:
                if isinstance(book, dict):
                    st.write(f"Title: {book['title']}, Author: {book['author']}")
                else:
                    st.write(book)

            # Step 11: Question Answering using BERT
            st.subheader("Question Answering with BERT")
            context = "Add a large block of text here for question-answering context."  # Replace with relevant context
            answer = answer_question(user_input, context)
            st.write(f"Answer: {answer}")
            
        else:
            st.write("Please enter a sentence or question.")

if __name__ == "__main__":
    main()