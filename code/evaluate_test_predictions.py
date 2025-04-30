import json
import os
import argparse
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset
from sari_hook import get_sari, get_sari_score
import evaluate
from nltk.tokenize import sent_tokenize
from scipy.stats import ttest_ind

model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

parser = argparse.ArgumentParser()
parser.add_argument("--ambiguous", type=bool, default=False)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--metric", type=str, default="keep")
parser.add_argument("--pretrained", type=bool, default=False)
parser.add_argument("--force", type=bool, default=False)
parser.add_argument("--num_beams", type=int, default=2)
parser.add_argument("--contrastive", type=bool, default=False)
parser.add_argument("--curriculum", type=bool, default=False)
parser.add_argument("--levenshtein", type=bool, default=False)
parser.add_argument("--difficulty", type=bool, default=False)
parser.add_argument("--train", type=bool, default=True)
parser.add_argument("--length_penalty", type=float, default=1.0)
parser.add_argument("--pipeline", type=bool, default=False)
parser.add_argument("--mbr", type=bool, default=False)
parser.add_argument("--num_mbr_samples", type=int, default=10)
parser.add_argument("--num_mbr_references", type=int, default=1)
parser.add_argument("--highlights", type=bool, default=False)
parser.add_argument("--top", type=str, default="topk")
parser.add_argument("--topk", type=int, default=50)
parser.add_argument("--epsilon_cutoff", type=float, default=0.02)
parser.add_argument("--topp", type=float, default=0.9)
parser.add_argument("--eval", type=str, default="test")
args = parser.parse_args()

# get prediction files from test_generations_disambiguation_model directory
files = os.listdir('test_generations_disambiguation_model/')
# group the ones that have the same name until "epoch"
grouped_files = {}
for file in files:
    name = file.split("_epoch")[0]
    #read the jsonl file with each line as json dict
    with open('test_generations_disambiguation_model/' + file) as f:
        lines = f.readlines()
        lines = [json.loads(line) for line in lines]
    if "8th grade" in lines[0]["reference"]:
        if name not in grouped_files:
            grouped_files[name] = []
        grouped_files[name].append(file)

print(grouped_files.keys())
print(len(grouped_files))

epoch_dict = {
    "predictions_mbr128128topepsilon_single_shuffle_crossentropy_novocabspec_numbeams1_all_bertscore_f1_flan-t5-base_lr5e-05_ootb_length_penalty1.0":7,
    #"predictions_mbr128128toptopp_single_shuffle_crossentropy_novocabspec_numbeams1_ambiguous_bertscore_precision_flan-t5-base_lr5e-05_ootb_length_penalty1.0":9,
    "predictions_nosampling_single_shuffle_crossentropy_novocabspec_numbeams2_all_bertscore_f1_flan-t5-base_lr5e-05_ootb_length_penalty1.0":7,
    #"predictions_nosampling_single_shuffle_crossentropy_novocabspec_numbeams2_ambiguous_bertscore_precision_flan-t5-base_lr5e-05_ootb_length_penalty1.0":5,
    }
#there are some more

get_sari_score.name = "sari"

def postprocess_text(preds, labels, sources):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    sources = [source.strip() for source in sources]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]
    sources = ["\n".join(sent_tokenize(source)) for source in sources]

    return preds, labels, sources

all_f1s = []
baseline_f1s = []

def compute_metrics(decoded_preds, label):
    
    dis_test_dataset = ambiguous_dis_test_dataset if "ambiguous" in full_name else all_dis_test_dataset

    if label != "all":
        idxs = label_idxs[label]
        decoded_preds = [decoded_preds[i] for i in idxs]
        dis_test_dataset = dis_test_dataset.select(idxs)

    pred_ids = [tokenizer(pred, padding=False, truncation=False)["input_ids"] for pred in decoded_preds]

    if args.eval == "test":
        decoded_multi_labels = dis_test_dataset["multi_target"]
        sources = [s for s in dis_test_dataset["source"]]
        sources_with_evidence = [s + "</s>" + e for s, e in zip(dis_test_dataset["source"], dis_test_dataset["evidence"])]
        all_references_ids = [[tokenizer(ref, padding=False, truncation=False)["input_ids"] for ref in item] for item in dis_test_dataset["multi_target"]]

    # Some simple post-processing
    decoded_preds, [], sources = postprocess_text(decoded_preds, [], sources)

    tokenized_input_ids = [tokenizer(item, padding=False, truncation=True)["input_ids"] for item in sources_with_evidence]
    source_ids = [tokenizer(item, padding=False, truncation=False)["input_ids"] for item in sources]

    f1s = []
    baseline_f1 = []

    results = {}
    baseline_results = {}
    metrics =  [evaluate.load("bertscore")]
    for metric in metrics: 
        if metric.name == "bert_score":
            result = {}
            baseline_result = {}
            result_items = []
            baseline_result_items = []
            if args.eval == "test":
                for item_id in range(len(decoded_preds)):
                    result_item = {}
                    baseline_result_item = {}
                    result_ = []
                    baseline_result_ = []
                    pred_id = decoded_preds[item_id]
                    all_references_id = decoded_multi_labels[item_id]
                    source_id = sources[item_id]
                    tokenized_input_id = tokenized_input_ids[item_id]
                    for ref_id, ref in enumerate(all_references_id):
                        result_.append(metric.compute(predictions=[pred_id], references=[ref], lang="en"))
                        baseline_result_.append(metric.compute(predictions=[source_id], references=[ref], lang="en"))
                    result_item["precision"] = np.mean([r["precision"] for r in result_])
                    result_item["recall"] = np.mean([r["recall"] for r in result_])
                    result_item["f1"] = (2 * result_item["precision"] * result_item["recall"]) / (result_item["precision"] + result_item["recall"])
                    f1s.append(result_item["f1"])
                    baseline_result_item["precision"] = np.mean([r["precision"] for r in baseline_result_])
                    baseline_result_item["recall"] = np.mean([r["recall"] for r in baseline_result_])
                    baseline_result_item["f1"] = (2 * baseline_result_item["precision"] * baseline_result_item["recall"]) / (baseline_result_item["precision"] + baseline_result_item["recall"])
                    baseline_f1.append(baseline_result_item["f1"])
                    result_items.append(result_item)
                    baseline_result_items.append(baseline_result_item)
                result["precision"] = np.mean([r["precision"] for r in result_items])
                result["recall"] = np.mean([r["recall"] for r in result_items])
                baseline_result["precision"] = np.mean([r["precision"] for r in baseline_result_items])
                baseline_result["recall"] = np.mean([r["recall"] for r in baseline_result_items])
                result = {"f1": (2 * np.mean(result["precision"]) * np.mean(result["recall"])) / (np.mean(result["precision"]) + np.mean(result["recall"])), "precision": np.mean(result["precision"]), "recall": np.mean(result["recall"])}
                baseline_result = {"f1": (2 * np.mean(baseline_result["precision"]) * np.mean(baseline_result["recall"])) / (np.mean(baseline_result["precision"]) + np.mean(baseline_result["recall"])), "precision": np.mean(baseline_result["precision"]), "recall": np.mean(baseline_result["recall"])}      
        elif metric.name == "rouge":
            if args.eval == "test":
                result = {}
                baseline_result = {}
                result_items = []
                baseline_result_items = []
                for item_id in range(len(decoded_preds)):
                    result_item = {}
                    baseline_result_item = {}
                    result_ = []
                    baseline_result_ = []
                    pred_id = decoded_preds[item_id]
                    all_references_id = decoded_multi_labels[item_id]
                    source_id = sources[item_id]
                    tokenized_input_id = tokenized_input_ids[item_id]
                    for ref_id, ref in enumerate(all_references_id):
                        result_.append(metric.compute(predictions=[pred_id], references=[ref], use_stemmer=True))
                        baseline_result_.append(metric.compute(predictions=[source_id], references=[ref], use_stemmer=True))
                    for key in result_[0]:
                        result_item[key] = np.mean([r[key] for r in result_])
                        baseline_result_item[key] = np.mean([r[key] for r in baseline_result_])
                    result_items.append(result_item)
                    baseline_result_items.append(baseline_result_item)
                for key in result_item: 
                    result[key] = np.mean([ri[key] for ri in result_items])
                    baseline_result[key] = np.mean([ri[key] for ri in baseline_result_items])
        elif metric.name == "comet":
            if args.eval == "test":
                result = {}
                baseline_result = {}
                result_items = []
                baseline_result_items = []
                for item_id in range(len(decoded_preds)):
                    result_item = {}
                    baseline_result_item = {}
                    result_ = []
                    baseline_result_ = []
                    pred_id = decoded_preds[item_id]
                    all_references_id = decoded_multi_labels[item_id]
                    source_id = sources[item_id]
                    source_ev_id = sources_with_evidence[item_id]
                    tokenized_input_id = tokenized_input_ids[item_id]
                    for ref_id, ref in enumerate(all_references_id):
                        result_.append(metric.compute(predictions=[pred_id], references=[ref], sources=[source_ev_id]))
                        baseline_result_.append(metric.compute(predictions=[source_id], references=[ref], sources=[source_ev_id]))
                    for key in result_[0]:
                        result_item[key] = np.mean([r[key] for r in result_])
                        baseline_result_item[key] = np.mean([r[key] for r in baseline_result_])
                    result_items.append(result_item)
                    baseline_result_items.append(baseline_result_item)
                for key in result_item: 
                    result[key] = np.mean([ri[key] for ri in result_items])
                    baseline_result[key] = np.mean([ri[key] for ri in baseline_result_items])
                del result["scores"]
                del baseline_result["scores"]
        elif metric.name == "sari":
            result = []
            baseline_result = []
            for item_id in range(len(pred_ids)):
                pred_id = pred_ids[item_id]
                all_references_id = all_references_ids[item_id]
                source_id = source_ids[item_id]
                tokenized_input_id = tokenized_input_ids[item_id]

                result.append(metric(prediction_ids=pred_id, list_of_targets=all_references_id, source_ids=tokenized_input_id))
                baseline_result.append(metric(prediction_ids=source_id, list_of_targets=all_references_id, source_ids=tokenized_input_id))
            result = {"sari": np.mean([r[0] for r in result]), "keep": np.mean([r[1] for r in result]), "add": np.mean([r[2] for r in result]), "delete": np.mean([r[3] for r in result])}
            result["keepandadd"] = result["keep"] + result["add"]
            baseline_result = {"sari": np.mean([r[0] for r in baseline_result]), "keep": np.mean([r[1] for r in baseline_result]), "add": np.mean([r[2] for r in baseline_result]), "delete": np.mean([r[3] for r in baseline_result])}
            baseline_result["keepandadd"] = baseline_result["keep"] + baseline_result["add"]
        result = {k: round(v * 100, 4) for k, v in result.items()}
        baseline_result = {k: round(v * 100, 4) for k, v in baseline_result.items()}
        for key in result: 
            results[key] = result[key]
        for key in baseline_result:
            baseline_results[key] = baseline_result[key]
        print(metric.name)
        print("model:")
        print(result)
        print("baseline:")
        print(baseline_result)
        if label == "all":
            all_f1s.append(f1s)
            baseline_f1s.append(baseline_f1)

ambiguous_test_path = "data/test_correction_ambiguous.jsonl"
all_test_path = "data/test_correction.jsonl"

with open(all_test_path, "r") as f:
    all_disambiguation_test_data = [json.loads(line) for line in f]

with open(ambiguous_test_path, "r") as f:
    ambiguous_disambiguation_test_data = [json.loads(line) for line in f]

label_idxs = {"supported": [], "refuted": [], "ambiguous": [], "insufficient": []}
for i in range(len(all_disambiguation_test_data)):
    if all_disambiguation_test_data[i]["mutation_type"] == "supported":
        label_idxs["supported"].append(i)
    elif all_disambiguation_test_data[i]["mutation_type"] == "refuted":
        label_idxs["refuted"].append(i)
    elif all_disambiguation_test_data[i]["mutation_type"] == "ambiguous":
        label_idxs["ambiguous"].append(i)
    elif all_disambiguation_test_data[i]["mutation_type"] == "insufficient":
        label_idxs["insufficient"].append(i)

def gen_all_test():
    for test_item in all_disambiguation_test_data: 
        yield {"id": test_item["id"], "source":test_item["source"], "evidence": test_item["evidence"], "target":test_item["target"][0], "multi_target":test_item["target"]}

def gen_ambiguous_test():
    for test_item in ambiguous_disambiguation_test_data: 
        yield {"id": test_item["id"], "source":test_item["source"], "evidence": test_item["evidence"], "target":test_item["target"][0], "multi_target":test_item["target"]}

all_dis_test_dataset = Dataset.from_generator(gen_all_test)
ambiguous_dis_test_dataset = Dataset.from_generator(gen_ambiguous_test)


for name in epoch_dict: 
    epoch = epoch_dict[name]
    full_name = name + "_epoch" + str(epoch) + ".jsonl"
    print("Q"*100)
    print(full_name)
    with open('test_generations_disambiguation_model/' + full_name) as f:
        lines = f.readlines()
        lines = [json.loads(line) for line in lines]
    decoded_preds = [line["prediction"] for line in lines]
    compute_metrics(decoded_preds, "supported")
    compute_metrics(decoded_preds, "refuted")
    compute_metrics(decoded_preds, "ambiguous")
    compute_metrics(decoded_preds, "insufficient")
    compute_metrics(decoded_preds, "all")

full_name = "llm_generations_test/eight_shot_generations.txt"
with open(full_name) as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    print("Q"*100)
    print(full_name)
    decoded_preds = lines
    compute_metrics(decoded_preds, "supported")
    compute_metrics(decoded_preds, "refuted")
    compute_metrics(decoded_preds, "ambiguous")
    compute_metrics(decoded_preds, "insufficient")
    compute_metrics(decoded_preds, "all")


print("Q"*100)
full_name = "human"
print(full_name)
lines = [i["target"][0] for i in all_disambiguation_test_data]
compute_metrics(lines, "supported")
compute_metrics(lines, "refuted")
compute_metrics(lines, "ambiguous")
compute_metrics(lines, "insufficient")
compute_metrics(lines, "all")


ambiguous_test_path = "data/test_correction_ambiguous.jsonl"
with open(ambiguous_test_path, "r") as f:
    disambiguation_test_data_ambiguous = [json.loads(line) for line in f]

all_f1s.append(baseline_f1s[0])
# compute significance scores between lists in all_f1s
for i in range(len(all_f1s)):
    for j in range(i+1, len(all_f1s)):
        print(i,j)
        print(ttest_ind(all_f1s[i], all_f1s[j]))
import pdb; pdb.set_trace()

print("Q"*100)
full_name = "human_ambiguous"
print(full_name)
lines = [i["target"][0] for i in disambiguation_test_data_ambiguous]
compute_metrics(lines, "all")

ambiguous_idxs = [i for i in range(len(all_disambiguation_test_data)) if all_disambiguation_test_data[i]["distance"] in [1,2,3]]

#todo do the ambiguous later too
full_name = "llm_generations_test/ambiguous_eight_shot_generations.txt"
with open(full_name) as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    # select the lines by ambiguous_idxs
    lines = [lines[i] for i in ambiguous_idxs]
    print("Q"*100)
    print(full_name)
    decoded_preds = lines
    compute_metrics(decoded_preds, "all")

