Lab 1: Sentiment Analysis

Explore flask and huggingface transformers.

To run the app:

```
python app.py
```

To install the dependencies:

```
pip install -r requirements.txt
```

This Service allows you add a Sentiment Analysis class: Think of a class as a category in which you want your Model to.
You can Add Classes via two options.

Option 1: Add Via Curl Command

```
curl -X POST http://your-api-url/api/classes \
 -H "Content-Type: application/json" \
 -d '{"class": "New Category Name"}'
```

image.png

Option 2: Add Via Url Param:

```
http://127.0.0.1:3000/api/classes/add?class=YourClassName
```

image.png
