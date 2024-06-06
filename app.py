from flask import Flask, request
from utils import get_model, get_tokenizer, predict_sentiment

app = Flask(__name__)

MODEL_NAME = 'cahya/bert-base-indonesian-522M'
MAX_LENGTH = 100
PRETRAINED_PATH = 'transformers-bert'

tokenizer = get_tokenizer(MODEL_NAME)
model = get_model(PRETRAINED_PATH)

@app.route("/predict/", methods=['GET'])
def predict():
    statement = request.json.get("statement")
    sentiment = predict_sentiment(statement, tokenizer, model, MAX_LENGTH)
    return {
        "sentiment": sentiment
    }

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)