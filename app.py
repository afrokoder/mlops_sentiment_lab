from flask import Flask, request, jsonify, render_template
from analyze import get_sentiment, compute_embeddings, classify_email, add_email_class, EMAIL_CLASSES
app = Flask(__name__, template_folder='templates')

@app.route("/")
def home():
    print("Home page")
    return render_template('index.html')


@app.route("/api/v1/sentiment-analysis/", methods=['POST'])
def analysis():
    if request.is_json:
        data = request.get_json()
        sentiment = get_sentiment(data['text'])
        return jsonify({"message": "Data received", "data": data, "sentiment": sentiment}), 200
    else:
        return jsonify({"error": "Invalid Content-Type"}), 400


@app.route("/api/v1/valid-embeddings/", methods=['GET'])
def valid_embeddings():
    embeddings = compute_embeddings()
    formatted_embeddings = []
    for text, vector in embeddings:
        formatted_embeddings.append({
            "text": text,
            "vector": vector.tolist() if hasattr(vector, 'tolist') else vector
        })
    embeddings = formatted_embeddings
    return jsonify({"message": "Valid embeddings fetched", "embeddings": embeddings}), 200


@app.route("/api/v1/classify/", methods=['POST'])
def classify():
    if request.is_json:
        data = request.get_json()
        text = data['text']
        classifications = classify_email(text)
        return jsonify({"message": "Email classified", "classifications": classifications}), 200
    else:
        return jsonify({"error": "Invalid Content-Type"}), 400


@app.route("/api/v1/classify-email/", methods=['GET'])
def classify_with_get():
    text = request.args.get('text')
    classifications = classify_email(text)
    return jsonify({"message": "Email classified", "classifications": classifications}), 200


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get all available email classes."""
    return jsonify(EMAIL_CLASSES)


@app.route('/api/classes', methods=['POST'])
def add_class():
    """Add a new email class."""
    data = request.json
    if not data or 'class' not in data:
        return jsonify({"success": False, "message": "Missing 'class' field"}), 400
    
    success, message = add_email_class(data['class'])
    return jsonify({"success": success, "message": message}), 200 if success else 400

# Add a new email class using query parameters in the url:http://127.0.0.1:3000/api/classes/add?class=Entertainment
@app.route('/api/classes/add', methods=['GET'])
def add_class_via_query():
    """Add a new email class using query parameters."""
    new_class = request.args.get('class')
    if not new_class:
        return jsonify({"success": False, "message": "Missing 'class' parameter"}), 400
    
    success, message = add_email_class(new_class)
    return jsonify({"success": success, "message": message}), 200 if success else 400
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)