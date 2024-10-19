from flask import Flask, request, jsonify, render_template
from main import predict_top_n_words_lstm

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/api/suggestions', methods=['POST'])
def suggestions():
    data = request.json
    input_text = data.get('input_text', '')

    try:
        top_words, relevant_sentences = predict_top_n_words_lstm(input_text)
        return jsonify({
            'words': top_words,
            'sentences': relevant_sentences
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=4000)