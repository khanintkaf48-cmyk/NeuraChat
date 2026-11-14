from flask import Flask, render_template, request
from encoder import Encoder
from decoder import Decoder
from process import Initializer
import pickle
import tensorflow as tf

''' Load the tokenizer data '''
with open('train_tokenizer.pickle', 'rb') as handle:
    train_tokenizer = pickle.load(handle)

with open('label_tokenizer.pickle', 'rb') as handle:
    label_tokenizer = pickle.load(handle)

''' Load the maximum training feature length '''
with open('max_data.txt', 'r') as file:
    max_len = int(file.read())

''' Process the data '''
BATCH_SIZE = 64
embedding_dim = 256
units = 1024
vocab_train = len(train_tokenizer.word_index) + 1
vocab_label = len(label_tokenizer.word_index) + 1

# Initialize models
encoder = Encoder(vocab_train, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_label, embedding_dim, units, BATCH_SIZE)

# ✅ CORRECT way to build models (fixes warning)
sample_input = tf.zeros((1, 1))
sample_hidden = tf.zeros((1, units))
sample_enc_output = tf.zeros((1, 1, units))

encoder(sample_input, sample_hidden)
decoder(sample_input, sample_hidden, sample_enc_output)

# ✅ FINAL FIX: load weights directly (NO checkpoint usage)
encoder.load_weights('enc/model')
decoder.load_weights('dec/model')

''' Start the sequence '''
seq = Initializer(units, train_tokenizer, max_len, label_tokenizer, encoder, decoder)

''' Flask app '''
app = Flask('NeuraChat')

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(seq.test(userText))

if __name__ == "__main__":
    app.run(debug=True)