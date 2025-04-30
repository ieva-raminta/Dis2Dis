from random import randrange
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
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
import torch

model_id = "google/flan-t5-base"
repository_id = f"{model_id.split('/')[1]}-disambiguation"

with open("data/dev_correction_ambiguous.jsonl", "r") as f:
    disambiguation_dev_data = [json.loads(line) for line in f]

def gen_dev():
    for dev_item in disambiguation_dev_data: 
        yield {"id": dev_item["id"], "source":dev_item["source"] + "</s>" + dev_item["evidence"], "target":dev_item["target"][0]}

dis_dev_dataset = Dataset.from_generator(gen_dev)


with open("data/test_correction.jsonl", "r") as f:
    disambiguation_test_data = [json.loads(line) for line in f]

def gen_test():
    for test_item in disambiguation_test_data: 
        yield {"id": test_item["id"], "source":test_item["source"] + " </s> " + test_item["evidence"], "target":test_item["target"][0]}

dis_test_dataset = Dataset.from_generator(gen_test)

# list the directory names in the repository_id directory in a list
files = os.listdir(repository_id)
# order the files list and get the last one
best_checkpoint = sorted([f for f in files if "checkpoint" in f])[0]
# load tokenizer from the repository_id directory
tokenizer = AutoTokenizer.from_pretrained(repository_id)

# load model and tokenizer from flan-t5-base-disambiguation directory
model = AutoModelForSeq2SeqLM.from_pretrained(repository_id+"/"+best_checkpoint).cuda()

test_predictions = []
test_predictions_for_human_eval = []
human_eval_ids = [2, 4, 9, 15, 31, 32, 34, 37, 39, 42, 43, 45, 47, 49, 54, 55, 57, 59, 62, 67, 70, 72, 74, 80, 85, 91, 95, 98, 102, 106, 108, 109, 114, 115, 117, 119, 121, 125, 129, 138, 140, 146, 147, 148, 151, 152, 153, 156, 163, 167]
human_eval_ids = [i-1 for i in human_eval_ids]

for sample_id, sample in enumerate(dis_test_dataset):
    # generate the output from the sample with the model
    output = model.generate(tokenizer(sample["source"], return_tensors="pt")["input_ids"].cuda(), max_new_tokens = 200)
    # convert the output token ids to text
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    test_predictions.append({"prediction": output_text, "reference": sample["target"], "id": sample["id"]})
    print(sample_id)
    if sample_id in human_eval_ids: 
        test_predictions_for_human_eval.append(output_text)

# save test_predictions to a file with newlines separating the json objects
with open("test_predictions_redo.jsonl", "w") as f:
    for prediction in test_predictions:
        f.write(json.dumps(prediction) + "\n")

with open("test_predictions_redo_human.txt", "w") as f:
    for prediction in test_predictions_for_human_eval:
        f.write(prediction + "\n")

