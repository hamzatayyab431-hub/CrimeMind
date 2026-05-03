import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

def finetune_distilbert():
    os.makedirs("agent/distilbert_deception", exist_ok=True)
    os.makedirs("assets/plots", exist_ok=True)
    df = pd.read_csv("deception_data.csv")
    if 'statement' in df.columns:
        df = df.rename(columns={'statement': 'text'})
    x_train, x_test, y_train, y_test = train_test_split(df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)
    train_encodings = tokenizer(x_train, truncation=True, padding='max_length', max_length=128)
    test_encodings = tokenizer(x_test, truncation=True, padding='max_length', max_length=128)
    train_dataset = Dataset.from_dict({'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask'], 'labels': y_train})
    test_dataset = Dataset.from_dict({'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask'], 'labels': y_test})
    training_args = TrainingArguments(
        output_dir="agent/distilbert_deception",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="logs"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    trainer.train()
    eval_results = trainer.evaluate()
    trainer.save_model("agent/distilbert_deception")
    tokenizer.save_pretrained("agent/distilbert_deception")
    log_history = trainer.state.log_history
    eval_loss = [x['eval_loss'] for x in log_history if 'eval_loss' in x]
    plt.figure()
    plt.plot(eval_loss)
    plt.savefig("assets/plots/distilbert_eval_loss.png")
    plt.close()
    preds = trainer.predict(test_dataset)
    y_pred = np.argmax(preds.predictions, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#1a1a2e"})
    sns.heatmap(cm, annot=True, fmt='d', cmap='flare')
    plt.savefig("assets/plots/distilbert_confusion_matrix.png")
    plt.close()
    return trainer, eval_results

def load_distilbert():
    tokenizer = DistilBertTokenizerFast.from_pretrained("agent/distilbert_deception")
    model = DistilBertForSequenceClassification.from_pretrained("agent/distilbert_deception")
    return model, tokenizer

def predict_distilbert(text):
    model, tokenizer = load_distilbert()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label_idx = torch.argmax(probs).item()
    classes = ["Likely Truthful", "Evasive", "Defensive/Hostile", "Fabricated/Over-justified"]
    return classes[label_idx], float(probs[0][label_idx] * 100)

if __name__ == "__main__":
    _, eval_results = finetune_distilbert()
    print(eval_results)
