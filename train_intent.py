#!/usr/bin/python3
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import json
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset

def get_current_directory_path():
    return os.getcwd()

def load_data(file_path='models/intents.json'):
    with open(file_path, 'r') as f:
        intents = json.load(f)
    texts, labels, tag_to_idx = [], [], {}
    for intent in intents['intents']:
        tag = intent['tag']
        tag_to_idx[tag] = tag_to_idx.get(tag, len(tag_to_idx))
        for pattern in intent['patterns']:
            texts.append(pattern)
            labels.append(tag_to_idx[tag])
    return texts, labels, tag_to_idx

def prepare_dataset(texts, labels):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=64)
    dataset = Dataset.from_dict({"input_ids": encodings['input_ids'], "attention_mask": encodings['attention_mask'], "labels": labels})
    return dataset, tokenizer

def do_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    texts, labels, tag_to_idx = load_data(f"{get_current_directory_path()}/models/intents.json")
    dataset, tokenizer = prepare_dataset(texts, labels)
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    training_args = TrainingArguments(
        output_dir=f"{get_current_directory_path()}/models",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        logging_dir=f"{get_current_directory_path()}/models/logs",
    )

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(tag_to_idx)).to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    model_dir = f"{get_current_directory_path()}/models/intent_classification_model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    mapping_file = os.path.join(model_dir, "tag_to_idx.json")
    with open(mapping_file, 'w') as f:
        json.dump(tag_to_idx, f)

    print("Training complete. Model and mapping saved to", model_dir)

if __name__ == "__main__":
    do_training()
