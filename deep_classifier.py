import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix

def train_bilstm():
    os.makedirs("agent", exist_ok=True)
    os.makedirs("assets/plots", exist_ok=True)
    df = pd.read_csv("crime_data.csv")
    if 'description' in df.columns:
        df = df.rename(columns={'description': 'text'})
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(df['label'])
    x_train, x_test, y_train, y_test = train_test_split(df['text'], y_encoded, test_size=0.2, random_state=42)
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train)
    x_train_pad = pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=100, padding='post', truncating='post')
    x_test_pad = pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=100, padding='post', truncating='post')
    model = Sequential([
        Embedding(5000, 64, input_length=100),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(patience=3, restore_best_weights=True)
    history = model.fit(x_train_pad, y_train, epochs=15, batch_size=32, validation_data=(x_test_pad, y_test), callbacks=[early_stop])
    loss, accuracy = model.evaluate(x_test_pad, y_test)
    model.save("agent/bilstm_evidence.h5")
    with open("agent/bilstm_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    with open("agent/bilstm_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    plt.figure()
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.legend()
    plt.savefig("assets/plots/bilstm_accuracy.png")
    plt.close()
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.savefig("assets/plots/bilstm_loss.png")
    plt.close()
    y_pred = np.argmax(model.predict(x_test_pad), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#1a1a2e"})
    sns.heatmap(cm, annot=True, fmt='d', cmap='flare')
    plt.savefig("assets/plots/bilstm_confusion_matrix.png")
    plt.close()
    return model, history, float(accuracy)

def load_bilstm():
    model = load_model("agent/bilstm_evidence.h5")
    with open("agent/bilstm_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("agent/bilstm_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, tokenizer, encoder

def predict_bilstm(text):
    model, tokenizer, encoder = load_bilstm()
    seq = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=100, padding='post', truncating='post')
    pred = model.predict(seq)[0]
    idx = np.argmax(pred)
    return str(encoder.inverse_transform([idx])[0]), float(pred[idx] * 100)

if __name__ == "__main__":
    _, _, test_accuracy = train_bilstm()
    print(test_accuracy)
