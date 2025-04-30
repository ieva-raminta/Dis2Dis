from random import randrange
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoModelForSequenceClassification,
)
import json
from datasets import concatenate_datasets, load_dataset, Dataset
import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from transformers import DataCollatorForSeq2Seq, pipeline
from huggingface_hub import HfFolder
import os
from scipy.stats import entropy
import torch 

model_id = "roberta-large"
repository_id = "/home/irs38/disambiguation/checkpoint-5299_roberta_4way_5e-6_f168"

# read a list of items from data/items_in_evaluation.txt
with open("data/generations_with_evidence_for_human_eval.txt") as f: 
    generation_data = f.readlines()
with open("data/inputs_in_evaluation.txt") as f: 
    input_data = f.readlines()

def gen_dev_generation(): 
    for item in generation_data:
        claim = item.split("Evidence: ")[0]
        evidence = "Evidence: "+item.split("Evidence: ")[1]
        yield {"text": claim+"[SEP]"+evidence, "label": None}
def gen_dev_input(): 
    for item in input_data:
        claim = item.split("Evidence: ")[0]
        evidence = "Evidence: "+item.split("Evidence: ")[1]
        yield {"text": claim+"[SEP]"+evidence, "label": None}

generation_dataset = Dataset.from_generator(gen_dev_generation)
input_dataset = Dataset.from_generator(gen_dev_input)

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForSequenceClassification.from_pretrained(repository_id).cuda()

ambiguous_scores = []
supported_scores = []
refuted_scores = []
neutral_scores = []
for item in generation_dataset: 
    output = model(tokenizer(item["text"], return_tensors="pt", max_length=512)["input_ids"].cuda())
    output.logits = torch.nn.functional.softmax(output.logits, dim=-1)
    ambiguous_scores.append(float(output.logits.detach()[0][0]))
    supported_scores.append(float(output.logits.detach()[0][1]))
    refuted_scores.append(float(output.logits.detach()[0][2]))
    neutral_scores.append(float(output.logits.detach()[0][3]))

input_ambiguous_scores = []
input_supported_scores = []
input_refuted_scores = []
input_neutral_scores = []
for item in input_dataset: 
    output = model(tokenizer(item["text"], return_tensors="pt", max_length=512)["input_ids"].cuda())
    output.logits = torch.nn.functional.softmax(output.logits, dim=-1)
    input_ambiguous_scores.append(float(output.logits.detach()[0][0]))
    input_supported_scores.append(float(output.logits.detach()[0][1]))
    input_refuted_scores.append(float(output.logits.detach()[0][2]))
    input_neutral_scores.append(float(output.logits.detach()[0][3]))

# write the scores to a csv file
with open("data/classifier_scores_for_human_eval.csv", "w") as f: 
    f.write("ambiguous,supported,refuted,neutral\n")
    for i in range(len(ambiguous_scores)): 
        f.write(str(ambiguous_scores[i])+","+str(supported_scores[i])+","+str(refuted_scores[i])+","+str(neutral_scores[i])+"\n")
with open("data/classifier_scores_for_human_eval_sources.csv", "w") as f: 
    f.write("ambiguous,supported,refuted,neutral\n")
    for i in range(len(input_ambiguous_scores)): 
        f.write(str(input_ambiguous_scores[i])+","+str(input_supported_scores[i])+","+str(input_refuted_scores[i])+","+str(input_neutral_scores[i])+"\n")


