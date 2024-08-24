from flask import Flask, render_template, request
import joblib
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = Flask(__name__)

nb_model = joblib.load(
    r'/Users/zainnofal/Desktop/lab04 hometask/sentiment.joblib')

bert_model, bert_tokenizer = joblib.load(
    r'/Users/zainnofal/Desktop/lab04 hometask/bert.joblib')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        prediction_method = request.form['prediction_method']

        if prediction_method == 'MultinomialNB':
            prediction = nb_model.predict([text])[0]
        elif prediction_method == 'BERT':
            tokenized_text = bert_tokenizer.encode(
                text, add_special_tokens=True)

            padded_sequence = torch.tensor(
                tokenized_text + [0]*(512-len(tokenized_text)))

            attention_mask = torch.where(
                padded_sequence != 0, torch.tensor(1), torch.tensor(0))

            with torch.no_grad():
                output = bert_model(padded_sequence.unsqueeze(
                    0), attention_mask.unsqueeze(0))
                logits = output.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities).item()

        return render_template('index.html', prediction=prediction, text=text)


if __name__ == '__main__':
    app.run(debug=True)
