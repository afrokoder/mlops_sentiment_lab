from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

sentiment_pipeline = pipeline("sentiment-analysis")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define the default file path for classes
DEFAULT_CLASSES_FILE = 'email_classes.json'

def load_email_classes(file_path=DEFAULT_CLASSES_FILE):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Error loading {file_path}: {str(e)}. Using default classes.")
        return ["Work", "Sports", "Food"]  # Fallback to default classes

def save_email_classes(classes, file_path=DEFAULT_CLASSES_FILE):
    """Save email classes to a JSON file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(classes, f, indent=2)

def add_email_class(new_class, file_path=DEFAULT_CLASSES_FILE):
    """Add a new email class and update the file.
    
    Returns:
        tuple: (success, message)
    """
    global EMAIL_CLASSES
    
    # Check if class already exists
    if new_class in EMAIL_CLASSES:
        return False, f"Class '{new_class}' already exists"
    
    # Add new class and update file
    try:
        EMAIL_CLASSES.append(new_class)
        save_email_classes(EMAIL_CLASSES, file_path)
        return True, f"Class '{new_class}' added successfully"
    except Exception as e:
        # If saving fails, roll back the change
        EMAIL_CLASSES.remove(new_class)
        return False, f"Failed to add class: {str(e)}"

# Load classes when module is imported
EMAIL_CLASSES = load_email_classes()

def get_sentiment(text):
    response = sentiment_pipeline(text)
    return response

def compute_embeddings(embeddings = EMAIL_CLASSES):
    embeddings = model.encode(embeddings)
    return zip(EMAIL_CLASSES, embeddings)

def classify_email(text):
    # Encode the input text
    text_embedding = model.encode([text])[0]
    
    # Get embeddings for all classes
    class_embeddings = compute_embeddings()
    
    # Calculate distances and return results
    results = []
    for class_name, class_embedding in class_embeddings:
        # Compute cosine similarity between text and class embedding
        similarity = np.dot(text_embedding, class_embedding) / (np.linalg.norm(text_embedding) * np.linalg.norm(class_embedding))
        results.append({
            "class": class_name,
            "similarity": float(similarity)  # Convert tensor to float for JSON serialization
        })
    
    # Sort by similarity score descending
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return results