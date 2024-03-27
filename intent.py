#!/usr/bin/python3.8
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import json
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, pipeline

# application imports
from helpers import *

class IntentClassifier:
    def __init__(self):
        model_dir = f"{get_current_directory_path()}/models/intent_classification_model"
        self.classify_intent = self.load_model_and_tokenizer(model_dir)
        self.idx_to_tag = self.load_tag_mappings(model_dir)
    
    def load_model_and_tokenizer(self, model_dir):
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        return pipeline("text-classification", model=model, tokenizer=tokenizer)

    def load_tag_mappings(self, model_dir):
        mapping_file = os.path.join(model_dir, "tag_to_idx.json")
        with open(mapping_file, 'r') as f:
            tag_to_idx = json.load(f)
        idx_to_tag = {v: k for k, v in tag_to_idx.items()}
        return idx_to_tag
    
    def classify_prompt(self, prompt):
        # Split the prompt into words
        words = prompt.split()

        # Take the first 100 words or less if the prompt is shorter than 100 words
        first_100_words = ' '.join(words[:100])

        # Perform intent classification on the first 100 words
        prediction = self.classify_intent(first_100_words)
        
        # Retrieve the intent tag based on the prediction
        intent_tag = self.idx_to_tag[int(prediction[0]['label'].split('_')[-1])]

        return intent_tag