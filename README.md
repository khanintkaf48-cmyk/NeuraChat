# NeuraChat — AI Conversational Chatbot

NeuraChat is a sequence-to-sequence conversational chatbot built with **TensorFlow/Keras**. It uses an **Encoder–Decoder architecture with an Attention mechanism** to generate responses from a small conversational dataset. A lightweight **Flask** web application provides the chat interface.

## Features

- Web-based chat interface powered by Flask
- Encoder–Decoder model using GRU layers
- Attention mechanism for response generation
- Text preprocessing and tokenization
- Saved model weights and tokenizer files for inference

## Tech Stack

- Python
- TensorFlow / Keras
- Flask
- NumPy
- Pandas
- HTML, CSS and jQuery

## Project Structure

```text
NeuraChat/
├── attention.py             # Attention layer
├── encoder.py               # Encoder network
├── decoder.py               # Decoder network
├── process.py               # Input processing and response generation
├── run.py                   # Flask application
├── train.py                 # Training-step logic
├── model_train.py           # Model training script
├── data_lit.py              # Dataset loading and preprocessing
├── chatbot.txt              # Conversation training data
├── train_tokenizer.pickle   # Input tokenizer
├── label_tokenizer.pickle   # Output tokenizer
├── max_data.txt             # Maximum input sequence length
├── enc/                     # Saved encoder weights
├── dec/                     # Saved decoder weights
└── templates/
    └── home.html            # Chat interface
```

## Setup

Use a Python environment compatible with the TensorFlow version used to train the supplied model weights.

1. Clone or download this repository.
2. Open a terminal in the project directory.
3. Create and activate a virtual environment:

   **Windows**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS / Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install the dependencies:

   ```bash
   pip install flask tensorflow numpy pandas
   ```

## Run the Chatbot

Start the Flask application:

```bash
python run.py
```

Open the local URL printed in the terminal (usually `http://127.0.0.1:5000`) in your browser and send a message.

The application loads the tokenizer files and pretrained weights from `enc/` and `dec/`. Keep these files and the Python source files in their expected relative locations.

## Train the Model

The training script is `model_train.py`. It reads the conversation pairs from `chatbot.txt`, preprocesses and tokenizes them, trains the Encoder–Decoder model, and saves the weights.

```bash
python model_train.py
```

Training may require additional hardware resources. The training script currently configures GPU memory growth and expects a compatible GPU to be available; review that configuration if you intend to train on a CPU-only machine.

## Dataset and Limitations

The chatbot was trained on a relatively small conversational dataset, so its responses may be limited, repetitive, or inaccurate. Response quality depends on the coverage and quality of the training data. This project is an educational demonstration, not a production-ready assistant.

## License

See the `LICENSE` file included in this repository.
