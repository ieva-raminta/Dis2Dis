import evaluate 
import numpy as np
import os
import json

bertscore_metrics = evaluate.load("bertscore")

# read the data from data/dev_correction.jsonl and data/dev_correction_ambiguous.jsonl
with open("data/dev_correction.jsonl", "r") as f:
    dev_data = [json.loads(line) for line in f]
with open("data/dev_correction_ambiguous.jsonl", "r") as f:
    dev_ambiguous_data = [json.loads(line) for line in f]

targets = []
ambiguous_targets = []
for dev_item in dev_data:
    targets.append(dev_item["target"][0])
for dev_item in dev_ambiguous_data:
    ambiguous_targets.append(dev_item["target"][0])
ambiguous_ids = [n for n,dev_item in enumerate(dev_data) if 4>dev_item["distance"]>0]

# iterate through files in llm_generations
for file in os.listdir("llm_generations"):
    # read the text file as a list separated by newlines
    with open("llm_generations/"+file, "r") as f:
        lines = f.readlines()

    lines = [l.strip() for l in lines]

    if "ambiguous" in file: 
        decoded_labels = ambiguous_targets
        lines = [lines[n] for n in ambiguous_ids]
    else: 
        decoded_labels = targets

    result = bertscore_metrics.compute(predictions=lines, references=decoded_labels, lang="en")
    result = {"f1": (2 * np.mean(result["precision"]) * np.mean(result["recall"])) / (np.mean(result["precision"]) + np.mean(result["recall"])), "precision": np.mean(result["precision"]), "recall": np.mean(result["recall"])}
    print(file)
    print(result)

    if "zero" in file: 
        decoded_labels = ambiguous_targets
        lines = [lines[n] for n in ambiguous_ids]
        result = bertscore_metrics.compute(predictions=lines, references=decoded_labels, lang="en")
        result = {"f1": (2 * np.mean(result["precision"]) * np.mean(result["recall"])) / (np.mean(result["precision"]) + np.mean(result["recall"])), "precision": np.mean(result["precision"]), "recall": np.mean(result["recall"])}
        print("ambiguous"+file)
        print(result)
