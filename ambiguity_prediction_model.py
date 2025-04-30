import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
    EarlyStoppingCallback, IntervalStrategy,
)
from huggingface_hub import HfFolder, notebook_login
import json
from datasets import concatenate_datasets, load_dataset, Dataset
from scipy.stats import entropy
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from random import randrange 
import pandas as pd 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained", type=bool, default=False)
parser.add_argument("--only_disambiguation", type=bool, default=False)
parser.add_argument("--learning_rate", type=float, default=5e-6)
parser.add_argument("--disagreement", type=str, default="entropy")
parser.add_argument("--full_ambifc", type=bool, default=False)
args = parser.parse_args()

def compute_metrics(p): 
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(labels, pred)
    recall = recall_score(labels, pred, average="macro")
    precision = precision_score(labels, pred, average="macro")
    f1 = f1_score(labels, pred, average="macro")
    print("f1", f1)
    f1_per_class = f1_score(labels, pred, average=None)
    print("f1 per class", f1_per_class)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def reformat_data_ambifc(labels):
    r = [0,0,0]
    for label in labels:
        if label == "supporting": 
            r[0] += 1
        elif label == "refuting": 
            r[1] += 1
        elif label == "neutral":
            r[2] += 1
    return r

if args.pretrained:
    checkpoints = []
    for directory in os.listdir("/home/irs38/disambiguation/roberta-large-ambiguity_prediction"):
        if directory.startswith("checkpoint-"):
            checkpoints.append(directory)
    # get the latest checkpoint
    latest_checkpoint = max(checkpoints)
    model_id = "/home/irs38/disambiguation/roberta-large-ambiguity_prediction/"+latest_checkpoint
else: 
    model_id = "roberta-large"

if args.pretrained: 
    repository_id = "roberta-large-ambiguity_prediction_pretrained_and_onlydisambiguation"
elif args.only_disambiguation:
    repository_id = "roberta-large-ambiguity_prediction_onlydisambiguation"
else:
    repository_id = "roberta-large-ambiguity_prediction"

if not args.only_disambiguation and not args.pretrained:
    dev_certain = open("data/dev.certain.jsonl")
    dev_certain_items = [json.loads(item) for item in list(dev_certain)]
    dev_uncertain = open("data/dev.uncertain.jsonl")
    dev_uncertain_items = [json.loads(item) for item in list(dev_uncertain)]
    train_certain = open("data/train.certain.jsonl")
    train_certain_items = [json.loads(item) for item in list(train_certain)]
    train_uncertain = open("data/train.uncertain.jsonl")
    train_uncertain_items = [json.loads(item) for item in list(train_uncertain)]
    train_data = train_certain_items + train_uncertain_items
    dev_data = dev_certain_items + dev_uncertain_items

    entropy_values = []

    def gen_train():
        for item in train_data:
            evidence = " ".join(item["sentences"].values())
            labels = [passage["label"] for passage in item["passage_annotations"]]
            label_entropy = entropy(reformat_data_ambifc(labels))
            ordinal_labels = [-1 if label == "refuting" else 0 if label == "neutral" else 1 for label in labels]
            label_variance = np.var(ordinal_labels)
            entropy_values.append(label_entropy)
            if args.disagreement == "variance":
                ambiguity = 0 if label_variance >= 0.9 else 1 if max(set(labels), key=labels.count) == "supporting" else 2 if max(set(labels), key=labels.count) == "refuting" else 3
            elif args.disagreement == "entropy":
                ambiguity = 0 if label_entropy >= 0.9 else 1 if max(set(labels), key=labels.count) == "supporting" else 2 if max(set(labels), key=labels.count) == "refuting" else 3
            if label_entropy >= 0.9 or label_entropy == 0 or (ambiguity == 2 and label_entropy < 0.68):
                yield {"text": item["claim"]+"[SEP]"+evidence, "label": ambiguity}

    def gen_dev(): 
        for item in dev_data:
            evidence = " ".join(item["sentences"].values())
            labels = [passage["label"] for passage in item["passage_annotations"]]
            label_entropy = entropy(reformat_data_ambifc(labels))
            entropy_values.append(label_entropy)
            ambiguity = 0 if label_entropy >= 0.9 else 1 if max(set(labels), key=labels.count) == "supporting" else 2 if max(set(labels), key=labels.count) == "refuting" else 3
            if label_entropy >= 0.9 or label_entropy == 0 or (ambiguity == 2 and label_entropy < 0.68):
                yield {"text": item["claim"]+"[SEP]"+evidence, "label": ambiguity}

else: 
    train_path = "data/train_correction.jsonl"
    dev_path = "data/dev_correction.jsonl"
    test_path = "data/test_correction.jsonl"

    with open(train_path, "r") as f:
        disambiguation_train_data = [json.loads(line) for line in f]
    with open(dev_path, "r") as f:
        disambiguation_dev_data = [json.loads(line) for line in f]
    with open(test_path, "r") as f:
        disambiguation_test_data = [json.loads(line) for line in f]

    def gen_train():
        for train_item in disambiguation_train_data: 
            label = 0 if train_item["distance"] in [1,2,3] else 1 if train_item["distance"] == 0 else 2 if train_item["distance"] == 5 else 3
            yield {"text":train_item["source"]+"[SEP]"+train_item["evidence"], "label":label}

    def gen_test():
        for test_item in disambiguation_test_data: 
            label = 0 if test_item["distance"] in [1,2,3] else 1 if test_item["distance"] == 0 else 2 if test_item["distance"] == 5 else 3
            yield {"text":test_item["source"]+"[SEP]"+test_item["evidence"], "label":label}

    def gen_dev():
        for dev_item in disambiguation_dev_data: 
            label = 0 if dev_item["distance"] in [1,2,3] else 1 if dev_item["distance"] == 0 else 2 if dev_item["distance"] == 5 else 3
            yield {"text":dev_item["source"]+"[SEP]"+dev_item["evidence"], "label":label}

train_dataset = Dataset.from_generator(gen_train)
dev_dataset = Dataset.from_generator(gen_dev)

if not args.only_disambiguation and not args.pretrained and not args.full_ambifc:
    # subsample the Dataset based on the class distribution from the disambiguation set
    ambiguity_0_train_items = [item for item in train_dataset if item["label"] == 0]
    ambiguity_1_train_items = [item for item in train_dataset if item["label"] == 1]
    ambiguity_2_train_items = [item for item in train_dataset if item["label"] == 2]
    ambiguity_3_train_items = [item for item in train_dataset if item["label"] == 3][:len(ambiguity_2_train_items)]
    ambiguity_0_dev_items = [item for item in dev_dataset if item["label"] == 0]
    ambiguity_1_dev_items = [item for item in dev_dataset if item["label"] == 1]
    ambiguity_2_dev_items = [item for item in dev_dataset if item["label"] == 2]
    ambiguity_3_dev_items = [item for item in dev_dataset if item["label"] == 3][:len(ambiguity_2_dev_items)]
    print("label distributions:")
    print(len(ambiguity_0_train_items), len(ambiguity_1_train_items), len(ambiguity_2_train_items), len(ambiguity_3_train_items))
    print(len(ambiguity_0_dev_items), len(ambiguity_1_dev_items), len(ambiguity_2_dev_items), len(ambiguity_3_dev_items))
    train_dataset = Dataset.from_pandas(pd.DataFrame(data=ambiguity_0_train_items + ambiguity_1_train_items + ambiguity_2_train_items + ambiguity_3_train_items))
    dev_dataset = Dataset.from_pandas(pd.DataFrame(data=ambiguity_0_dev_items + ambiguity_1_dev_items + ambiguity_2_dev_items + ambiguity_3_dev_items))

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
dev_dataset = dev_dataset.map(tokenize, batched=True, batch_size=len(dev_dataset))

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
dev_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

num_labels = 4
class_names = ["ambiguous", "supported", "refuted", "neutral"]
print(f"number of labels: {num_labels}")
print(f"the labels: {class_names}")

# Create an id2label mapping
id2label = {i: label for i, label in enumerate(class_names)}

# Update the model's configuration with the id2label mapping
config = AutoConfig.from_pretrained("roberta-large")
config.update({"id2label": id2label})

# Model
model = RobertaForSequenceClassification.from_pretrained(model_id, config=config)

batch_size = 8

# TrainingArguments
training_args = TrainingArguments(
    output_dir=repository_id,
    num_train_epochs=30,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="steps",
    logging_dir=f"{repository_id}/logs",
    logging_strategy="steps",
    logging_steps=1000,
    learning_rate=args.learning_rate,
    weight_decay=0.01,
    warmup_steps=500,
    save_strategy="steps",
    eval_steps=len(train_dataset)//batch_size,
    save_steps=len(train_dataset)//batch_size,
    load_best_model_at_end=True,
    save_total_limit=1,
    report_to="tensorboard",
    push_to_hub=False,
    hub_strategy="every_save",
    hub_model_id=repository_id,
    hub_token=HfFolder.get_token(),
    greater_is_better=True,
    metric_for_best_model='f1',
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()
trainer.evaluate()

tokenizer.save_pretrained(repository_id)
