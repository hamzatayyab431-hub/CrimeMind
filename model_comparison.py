import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

def load_all_models():
    m1 = joblib.load("agent/clue_classifier.pkl")
    m2 = joblib.load("agent/deception_model.pkl")
    m3 = load_model("agent/bilstm_evidence.h5")
    with open("agent/bilstm_tokenizer.pkl", "rb") as f:
        t3 = pickle.load(f)
    with open("agent/bilstm_encoder.pkl", "rb") as f:
        e3 = pickle.load(f)
    t4 = DistilBertTokenizerFast.from_pretrained("agent/distilbert_deception")
    m4 = DistilBertForSequenceClassification.from_pretrained("agent/distilbert_deception")
    return m1, m2, m3, t3, e3, m4, t4

def plot_metrics(y_true, y_pred, y_prob, name, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#1a1a2e"})
    sns.heatmap(cm, annot=True, fmt='d', cmap=sns.light_palette("#e94560", as_cmap=True), xticklabels=classes, yticklabels=classes)
    plt.savefig(f"assets/plots/{name}_cm.png", dpi=150)
    plt.close()
    y_bin = label_binarize(y_true, classes=range(len(classes)))
    n_classes = y_bin.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.figure(figsize=(8,6))
    plt.plot(fpr["macro"], tpr["macro"], color='#e94560', lw=2, label=f'macro auc = {roc_auc["macro"]:.2f}')
    plt.legend(loc="lower right")
    plt.savefig(f"assets/plots/{name}_roc.png", dpi=150)
    plt.close()
    plt.figure(figsize=(8,6))
    for i in range(n_classes):
        p, r, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        plt.plot(r, p, lw=2, label=f'class {i}')
    plt.legend(loc="lower left")
    plt.savefig(f"assets/plots/{name}_prc.png", dpi=150)
    plt.close()
    return roc_auc["macro"]

def run_full_evaluation():
    os.makedirs("assets/plots", exist_ok=True)
    m1, m2, m3, t3, e3, m4, t4 = load_all_models()
    c_df = pd.read_csv("crime_data.csv")
    if 'description' in c_df.columns:
        c_df = c_df.rename(columns={'description': 'text', 'category': 'cat_label', 'label': 'str_label'})
    x_tr_c, x_te_c, y_tr_c, y_te_c, y_tr_c_str, y_te_c_str = train_test_split(c_df['text'].tolist(), c_df['cat_label'].tolist(), c_df['str_label'].tolist(), test_size=0.2, random_state=42)
    d_df = pd.read_csv("deception_data.csv")
    if 'statement' in d_df.columns:
        d_df = d_df.rename(columns={'statement': 'text'})
    x_tr_d, x_te_d, y_tr_d, y_te_d = train_test_split(d_df['text'].tolist(), d_df['label'].tolist(), test_size=0.2, random_state=42)
    res = []
    def evaluate_sklearn(model, x, y, name, task):
        start = time.time()
        preds = model.predict(x)
        probs = model.predict_proba(x)
        inf_time = (time.time() - start) * 1000 / len(x)
        rpt = classification_report(y, preds, output_dict=True)
        classes = [str(c) for c in model.classes_]
        mac_auc = plot_metrics(y, preds, probs, name, classes)
        res.append({
            'Model': name, 'Task': task, 'Accuracy': rpt['accuracy'],
            'Macro_F1': rpt['macro avg']['f1-score'], 'Macro_AUC': mac_auc,
            'Inference_ms': inf_time, 'Architecture': 'sklearn'
        })
    def evaluate_bilstm():
        seqs = pad_sequences(t3.texts_to_sequences(x_te_c), maxlen=100, padding='post', truncating='post')
        y_encoded = e3.transform(y_te_c_str)
        start = time.time()
        probs = m3.predict(seqs)
        preds = np.argmax(probs, axis=1)
        inf_time = (time.time() - start) * 1000 / len(x_te_c)
        rpt = classification_report(y_encoded, preds, output_dict=True)
        mac_auc = plot_metrics(y_encoded, preds, probs, 'BiLSTM', list(e3.classes_))
        res.append({
            'Model': 'BiLSTM', 'Task': 'Evidence', 'Accuracy': rpt['accuracy'],
            'Macro_F1': rpt['macro avg']['f1-score'], 'Macro_AUC': mac_auc,
            'Inference_ms': inf_time, 'Architecture': 'TensorFlow'
        })
    def evaluate_distilbert():
        start = time.time()
        batch_size = 8
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(x_te_d), batch_size):
                batch_texts = x_te_d[i:i+batch_size]
                inputs = t4(batch_texts, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
                outputs = m4(**inputs)
                batch_probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()
                all_probs.append(batch_probs)
        probs = np.vstack(all_probs)
        preds = np.argmax(probs, axis=1)
        inf_time = (time.time() - start) * 1000 / len(x_te_d)
        rpt = classification_report(y_te_d, preds, output_dict=True)
        classes = ["Likely Truthful", "Evasive", "Defensive/Hostile", "Fabricated/Over-justified"]
        mac_auc = plot_metrics(y_te_d, preds, probs, 'DistilBERT', classes)
        res.append({
            'Model': 'DistilBERT', 'Task': 'Deception', 'Accuracy': rpt['accuracy'],
            'Macro_F1': rpt['macro avg']['f1-score'], 'Macro_AUC': mac_auc,
            'Inference_ms': inf_time, 'Architecture': 'Transformers'
        })
    evaluate_sklearn(m1, x_te_c, y_te_c, 'SVM', 'Evidence')
    evaluate_sklearn(m2, x_te_d, y_te_d, 'Random_Forest', 'Deception')
    evaluate_bilstm()
    evaluate_distilbert()
    df = pd.DataFrame(res)
    df.to_csv("assets/plots/model_comparison.csv", index=False)
    print(df.to_string())
    return df

def plot_all_confusion_matrices():
    pass

def plot_all_roc_curves():
    pass

def get_comparison_html():
    df = pd.read_csv("assets/plots/model_comparison.csv")
    return df.to_html()

if __name__ == "__main__":
    os.makedirs("assets/plots", exist_ok=True)
    run_full_evaluation()
