from random import randrange
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback, 
    IntervalStrategy,
    GenerationConfig,
    T5ForConditionalGeneration,
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    AutoConfig,
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
from sari_hook import get_sari, get_sari_score
from mbr import MBR, MBRConfig, MBRGenerationMixin
#from training_args_seq2seq_ import Seq2SeqTrainingArguments
import argparse

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
parser.add_argument("--eval", type=str, default="dev")
args = parser.parse_args()

if args.force: 
    from seq2seqtrainer_force import Seq2SeqTrainer
    batch_size = 1
elif args.mbr: 
    args.num_beams = 1
    if args.num_mbr_references == 1: 
        args.num_mbr_references = args.num_mbr_samples
    from seq2seqtrainer_mbr import Seq2SeqTrainer
    batch_size = 8
elif args.contrastive: 
    from seq2seqtrainer_ordered import Seq2SeqTrainer
    batch_size = 3
    args.ambiguous = False
elif args.levenshtein or args.curriculum or args.difficulty:
    from seq2seqtrainer_ordered import Seq2SeqTrainer
    args.ambiguous = False
    batch_size = 8
else: 
    from transformers import Seq2SeqTrainer
    batch_size = 8

nltk.download("punkt")

model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

if args.pipeline: 
    classifier_checkpoints = []
    for classifier_directory in os.listdir("/home/irs38/disambiguation/roberta-large-ambiguity_prediction"):
        if classifier_directory.startswith("checkpoint-"):
            classifier_checkpoints.append(classifier_directory)
    # get the latest checkpoint
    latest_classifier_checkpoint = max(classifier_checkpoints)
    classifier_model_id = "/home/irs38/disambiguation/roberta-large-ambiguity_prediction/"+latest_classifier_checkpoint
    classifier_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
    class_names = ["ambiguous", "supported", "refuted", "neutral"]
    id2label = {i: label for i, label in enumerate(class_names)}
    classifier_model_id = "roberta-large"
    classifier_config = AutoConfig.from_pretrained(classifier_model_id)
    classifier_config.update({"id2label": id2label})
    classifier = RobertaForSequenceClassification.from_pretrained(classifier_model_id, config=classifier_config)

if args.ambiguous:
    train_path = "data/train_correction_ambiguous.jsonl"
    dev_path = "data/dev_correction_ambiguous.jsonl"
    test_path = "data/test_correction_ambiguous.jsonl"
    args.metric = "bertscore_precision"
else: 
    args.metric = "bertscore_f1"
    if args.levenshtein: 
        train_path = "data/train_correction_levenshtein.jsonl"
        dev_path = "data/dev_correction.jsonl"
        test_path = "data/test_correction.jsonl"
    elif args.difficulty: 
        train_path = "data/train_correction_difficulty.jsonl"
        dev_path = "data/dev_correction.jsonl"
        test_path = "data/test_correction.jsonl"
    elif args.curriculum:
        train_path = "data/train_correction_distance.jsonl"
        dev_path = "data/dev_correction.jsonl"
        test_path = "data/test_correction.jsonl" 
    elif args.contrastive: 
        train_path = "data/train_correction_triples_label.jsonl" # or triples
        dev_path = "data/dev_correction.jsonl"
        test_path = "data/test_correction.jsonl"
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
        if "contrastive" not in train_item: 
            train_item["contrastive"] = False
        if args.pipeline: 
            classifier_tokenized_inputs = classifier_tokenizer(train_item["source"]+"[SEP]"+train_item["evidence"], truncation=True, padding="max_length", return_tensors="pt")
            classification = classifier(**classifier_tokenized_inputs)
            class_ = np.argmax(classification.logits.detach())
            if class_ == 0: 
                yield {"id": train_item["id"], "contrastive":train_item["contrastive"], "source":train_item["source"], "evidence": train_item["evidence"], "target":train_item["target"][0], "multi_target":train_item["target"]}
        else: 
            yield {"id": train_item["id"], "contrastive":train_item["contrastive"], "source":train_item["source"], "evidence": train_item["evidence"], "target":train_item["target"][0], "multi_target":train_item["target"]}

def gen_test():
    for test_item in disambiguation_test_data: 
        if "contrastive" not in test_item: 
            test_item["contrastive"] = False
        if args.pipeline: 
            classifier_tokenized_inputs = classifier_tokenizer(test_item["source"]+"[SEP]"+test_item["evidence"], truncation=True, padding="max_length", return_tensors="pt")
            classification = classifier(**classifier_tokenized_inputs)
            class_ = np.argmax(classification.logits.detach())
            if class_ == 0:
                yield {"id": test_item["id"], "contrastive":test_item["contrastive"], "source":test_item["source"], "evidence": test_item["evidence"], "rule_based_generation":None, "target":test_item["target"][0], "multi_target":test_item["target"]}
            elif class_ == 1:
                yield {"id": test_item["id"], "contrastive":test_item["contrastive"], "source":test_item["source"], "evidence": test_item["evidence"], "rule_based_generation":test_item["source"], "target":test_item["target"][0], "multi_target":test_item["target"]}
            elif class_ == 2:
                refuted_target = "It is not true that "+test_item["source"][0].lower()+test_item["source"][1:]
                yield {"id": test_item["id"], "contrastive":test_item["contrastive"], "source":test_item["source"], "evidence": test_item["evidence"], "rule_based_generation":refuted_target, "target":test_item["target"][0], "multi_target":test_item["target"]}
            else:
                neutral_target = "It is not clear from the evidence whether "+test_item["source"][0].lower()+test_item["source"][1:]
                yield {"id": test_item["id"], "contrastive":test_item["contrastive"], "source":test_item["source"], "evidence": test_item["evidence"], "rule_based_generation":neutral_target, "target":test_item["target"][0], "multi_target":test_item["target"]}
        else:
            yield {"id": test_item["id"], "contrastive":test_item["contrastive"], "source":test_item["source"], "evidence": test_item["evidence"], "target":test_item["target"][0], "multi_target":test_item["target"]}

dev_unambiguous_ids = []
def gen_dev():
    for dev_id, dev_item in enumerate(disambiguation_dev_data): 
        if "contrastive" not in dev_item: 
            dev_item["contrastive"] = False
        if args.pipeline: 
            classifier_tokenized_inputs = classifier_tokenizer(dev_item["source"]+"[SEP]"+dev_item["evidence"], truncation=True, padding="max_length", return_tensors="pt")
            classification = classifier(**classifier_tokenized_inputs)
            class_ = np.argmax(classification.logits.detach())
            if class_ == 0:
                yield {"id": dev_item["id"], "contrastive":dev_item["contrastive"], "source":dev_item["source"], "evidence": dev_item["evidence"], "rule_based_generation":None, "target":dev_item["target"][0], "multi_target":dev_item["target"]}
            elif class_ == 1:
                dev_unambiguous_ids.append(dev_id)
                yield {"id": dev_item["id"], "contrastive":dev_item["contrastive"], "source":dev_item["source"], "evidence": dev_item["evidence"], "rule_based_generation":dev_item["source"], "target":dev_item["target"][0], "multi_target":dev_item["target"]}
            elif class_ == 2:
                dev_unambiguous_ids.append(dev_id)
                refuted_target = "It is not true that "+dev_item["source"][0].lower()+dev_item["source"][1:]
                yield {"id": dev_item["id"], "contrastive":dev_item["contrastive"], "source":dev_item["source"], "evidence": dev_item["evidence"], "rule_based_generation":refuted_target, "target":dev_item["target"][0], "multi_target":dev_item["target"]}
            else:
                dev_unambiguous_ids.append(dev_id)
                neutral_target = "It is not clear from the evidence whether "+dev_item["source"][0].lower()+dev_item["source"][1:]
                yield {"id": dev_item["id"], "contrastive":dev_item["contrastive"], "source":dev_item["source"], "evidence": dev_item["evidence"], "rule_based_generation":neutral_target, "target":dev_item["target"][0], "multi_target":dev_item["target"]}
        else:
            yield {"id": dev_item["id"], "contrastive":dev_item["contrastive"], "source":dev_item["source"], "evidence": dev_item["evidence"], "target":dev_item["target"][0], "multi_target":dev_item["target"]}


dis_train_dataset = Dataset.from_generator(gen_train)
dis_dev_dataset = Dataset.from_generator(gen_dev)
dis_test_dataset = Dataset.from_generator(gen_test)

# subset datasets with randrange
#dis_train_dataset = dis_train_dataset.select(range(100))
#dis_dev_dataset = dis_dev_dataset.select(range(100))

tokenized_inputs = concatenate_datasets([dis_train_dataset, dis_dev_dataset, dis_test_dataset]).map(
    lambda x: tokenizer(x["source"], x["evidence"], truncation="only_second"),
    batched=True,
    remove_columns=["source", "evidence", "target"],
)
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
tokenized_targets = concatenate_datasets([dis_train_dataset, dis_dev_dataset, dis_test_dataset]).map(
    lambda x: tokenizer(x["target"], truncation=True),
    batched=True,
    remove_columns=["source", "evidence", "target"],
)
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])

def preprocess_function(sample,padding="max_length"):

    instruction = "Please make the following claim less ambiguous with regard to the following evidence: "

    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    inputs = [instruction + item for item in sample["source"]]
    input_evidence = [item for item in sample["evidence"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, input_evidence, max_length=max_source_length, padding=padding, truncation="only_second")

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["target"], max_length=max_source_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["contrastive"] = [[1]*len(labels["input_ids"][0]) if s==True else [0]*len(labels["input_ids"][0]) for s in sample["contrastive"]]
    return model_inputs

tokenized_dataset_train = dis_train_dataset.map(
    preprocess_function, batched=True, remove_columns=["source", "evidence", "target", "id", "multi_target"]
)
tokenized_dataset_dev = dis_dev_dataset.map(
    preprocess_function, batched=True, remove_columns=["source", "evidence", "target", "id", "multi_target"]
)
tokenized_dataset_test = dis_test_dataset.map(
    preprocess_function, batched=True, remove_columns=["source", "evidence", "target", "id", "multi_target"]
)
print(f"Keys of tokenized dataset: {list(tokenized_dataset_train.features)}")

if args.pretrained:
    checkpoints = []
    model_location = "/home/irs38/disambiguation/flan-t5-base-pretraining-disambiguation/"
    for directory in os.listdir(model_location):
        if directory.startswith("checkpoint-"):
            checkpoints.append(directory)
    # get the latest checkpoint
    latest_checkpoint = max(checkpoints)
    model = T5ForConditionalGeneration.from_pretrained(model_location+latest_checkpoint)
elif not args.train: 
    checkpoints = []
    model_location = "/home/irs38/disambiguation/flan-t5-base-pretraining-disambiguation/"
    for directory in os.listdir(model_location):
        if directory.startswith("checkpoint-"):
            checkpoints.append(directory)
    # get the latest checkpoint
    latest_checkpoint = max(checkpoints)
    model = T5ForConditionalGeneration.from_pretrained(model_location+latest_checkpoint)

if args.mbr: 
    mbr_config = MBRConfig(
        num_samples=args.num_mbr_samples,
        num_references=args.num_mbr_references,
        metric=evaluate.load("bertscore") if "bert" in args.metric else args.metric,
        metric_output_field="precision" if args.ambiguous else "f1",
        )
    if args.pretrained: 
        model = MBR(T5ForConditionalGeneration).from_pretrained(model_location+latest_checkpoint)
    else: 
        model = MBR(T5ForConditionalGeneration).from_pretrained(model_id)
    model.mbr_config = mbr_config
else: 
    model = T5ForConditionalGeneration.from_pretrained(model_id)

if (args.length_penalty != 1.0 or args.force == True) and args.num_beams == 1: 
    args.num_beams = 2

if args.mbr: 
    if args.top == "epsilon": 
        generation_config = GenerationConfig(do_sample=True, num_beams=args.num_beams, epsilon_cutoff=args.epsilon_cutoff, decoder_start_token_id=0, eos_token_id=1, pad_token_id=0, max_new_tokens=1000, length_penalty=args.length_penalty)
    elif args.top == "topk":
        generation_config = GenerationConfig(do_sample=True, num_beams=args.num_beams, top_k=args.topk, decoder_start_token_id=0, eos_token_id=1, pad_token_id=0, max_new_tokens=1000, length_penalty=args.length_penalty)
    else: 
        generation_config = GenerationConfig(do_sample=True, num_beams=args.num_beams, top_p=args.topp, decoder_start_token_id=0, eos_token_id=1, pad_token_id=0, max_new_tokens=1000, length_penalty=args.length_penalty)
else:
    generation_config = GenerationConfig(num_beams=args.num_beams, decoder_start_token_id=0, eos_token_id=1, pad_token_id=0, max_new_tokens=1000, length_penalty=args.length_penalty)
model.generation_config = generation_config
tokenizer.generation_config = generation_config

# Metric
get_sari_score.name = "sari"

# helper function to postprocess text
def postprocess_text(preds, labels, sources):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    sources = [source.strip() for source in sources]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]
    sources = ["\n".join(sent_tokenize(source)) for source in sources]

    return preds, labels, sources

def compute_metrics(eval_preds):
    list_for_saving = []

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # preds remove the -100 items
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    pred_ids = [[tok for tok in pred if tok != tokenizer.pad_token_id] for pred in preds]

    if args.eval == "test":
        decoded_multi_labels = dis_test_dataset["multi_target"]
        sources = [s for s in dis_test_dataset["source"]]
        sources_with_evidence = [s + "</s>" + e for s, e in zip(dis_test_dataset["source"], dis_test_dataset["evidence"])]
        all_references_ids = [[tokenizer(ref, padding=False, truncation=False)["input_ids"] for ref in item] for item in dis_test_dataset["multi_target"]]
    else: 
        decoded_multi_labels = dis_test_dataset["multi_target"]
        sources = [s for s in dis_dev_dataset["source"]] #!!! this is hard coded for now
        sources_with_evidence = [s + "</s>" + e for s, e in zip(dis_dev_dataset["source"], dis_dev_dataset["evidence"])]
        all_references_ids = [[tokenizer(ref, padding=False, truncation=False)["input_ids"] for ref in item] for item in dis_dev_dataset["multi_target"]]
        
    # Some simple post-processing
    decoded_preds, decoded_labels, sources = postprocess_text(decoded_preds, decoded_labels, sources)

    tokenized_input_ids = [tokenizer(item, padding=False, truncation=True)["input_ids"] for item in sources_with_evidence]
    source_ids = [tokenizer(item, padding=False, truncation=False)["input_ids"] for item in sources]

    if args.pipeline: #this was WIP
        for prediction_id, prediction in enumerate(decoded_preds): 
            if prediction_id in dev_unambiguous_ids: 
                decoded_preds[prediction_id] = dis_test_dataset["rule_based_generation"][prediction_id]
                maxlen = len(preds[0])
                preds[prediction_id] = tokenizer(dis_test_dataset["rule_based_generation"][prediction_id], return_tensors="pt", padding="max_length",max_length=maxlen)["input_ids"]

    results = {}
    baseline_results = {}
    metrics = [get_sari_score] if args.eval == "test" else [evaluate.load("rouge"), evaluate.load("bertscore"), evaluate.load("comet"), get_sari_score]
    for metric in metrics: 
        if metric.name == "bert_score":
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
                    baseline_result_item["precision"] = np.mean([r["precision"] for r in baseline_result_])
                    baseline_result_item["recall"] = np.mean([r["recall"] for r in baseline_result_])
                    result_items.append(result_item)
                    baseline_result_items.append(baseline_result_item)
                result["precision"] = np.mean([r["precision"] for r in result_items])
                result["recall"] = np.mean([r["recall"] for r in result_items])
                baseline_result["precision"] = np.mean([r["precision"] for r in baseline_result_items])
                baseline_result["recall"] = np.mean([r["recall"] for r in baseline_result_items])
                result = {"f1": (2 * np.mean(result["precision"]) * np.mean(result["recall"])) / (np.mean(result["precision"]) + np.mean(result["recall"])), "precision": np.mean(result["precision"]), "recall": np.mean(result["recall"])}
                baseline_result = {"f1": (2 * np.mean(baseline_result["precision"]) * np.mean(baseline_result["recall"])) / (np.mean(baseline_result["precision"]) + np.mean(baseline_result["recall"])), "precision": np.mean(baseline_result["precision"]), "recall": np.mean(baseline_result["recall"])}      
            else: 
                result = metric.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
                result = {"f1": (2 * np.mean(result["precision"]) * np.mean(result["recall"])) / (np.mean(result["precision"]) + np.mean(result["recall"])), "precision": np.mean(result["precision"]), "recall": np.mean(result["recall"])}
                baseline_result = metric.compute(predictions=sources, references=decoded_labels, lang="en")
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
            else:
                result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
                baseline_result = metric.compute(predictions=sources, references=decoded_labels, use_stemmer=True) 
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
            else:
                result = metric.compute(predictions=decoded_preds, references=decoded_labels, sources=sources_with_evidence)
                del result["scores"]
                baseline_result = metric.compute(predictions=sources, references=decoded_labels, sources=sources_with_evidence)
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
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        target_lens = [np.count_nonzero(target != tokenizer.pad_token_id) for target in labels]
        result["gen_len"] = np.mean(prediction_lens)
        baseline_result["gen_len"] = np.mean(target_lens)
        for key in result: 
            results[key] = result[key]
        for key in baseline_result:
            baseline_results[key] = baseline_result[key]
        print(metric.name)
        print("model:")
        print(result)
        print("baseline:")
        print(baseline_result)
    for count, pred in enumerate(decoded_preds):
        list_for_saving.append({"prediction": pred, "reference": decoded_labels[count], "source": sources[count], "id": dis_test_dataset["id"][count]})
    model_name = model_id.split("/")[1]
    epoch = str(trainer.state.epoch).split(".")[0] if str(trainer.state.epoch).split(".")[1]=="0" else int(str(trainer.state.epoch).split(".")[0])+1
    ambiguous = "ambiguous" if args.ambiguous else "all"
    metric = args.metric
    pipeline = "pipeline" if args.pipeline else "single"
    learning_rate = args.learning_rate
    pretrained = "pretrained" if args.pretrained else "ootb"
    force = "force" if args.force else "novocabspec"
    num_beams = str(args.num_beams)
    curriculum = "levenshtein" if args.levenshtein else "difficulty" if args.difficulty else "curriculum" if args.curriculum else "shuffle"
    contrastive = "contrastive" if args.contrastive else "crossentropy"
    length_penalty = "length_penalty"+str(args.length_penalty)
    mbr = "mbr"+str(args.num_mbr_samples)+str(args.num_mbr_references)+"top"+str(args.top) if args.mbr else "nosampling"
    folder = "test_generations_disambiguation_model" if args.eval == "test" else "generations_disambiguation_model"
    savename = f"{folder}/predictions_{mbr}_{pipeline}_{curriculum}_{contrastive}_{force}_numbeams{num_beams}_{ambiguous}_{metric}_{model_name}_lr{learning_rate}_{pretrained}_{length_penalty}_epoch{epoch}.jsonl"
    print(savename)
    with open(savename, "w") as f:
        for item in list_for_saving:
            f.write(json.dumps(item) + "\n")
    return results


# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8
)

# Hugging Face repository id
repository_id = f"{model_id.split('/')[1]}-disambiguation"

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=repository_id,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    fp16=False,  # Overflows with fp16
    learning_rate=args.learning_rate,
    num_train_epochs=30,
    # logging & evaluation strategies
    logging_dir=f"{repository_id}/logs",
    logging_strategy="steps",
    logging_steps=1000,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=len(dis_train_dataset)//batch_size,
    eval_steps=len(dis_train_dataset)//batch_size,
    save_total_limit=1,
    load_best_model_at_end=True,
    greater_is_better=True,
    report_to="tensorboard",
    push_to_hub=False,
    hub_strategy="every_save",
    hub_model_id=repository_id,
    hub_token=HfFolder.get_token(),
    metric_for_best_model= 'mean_score' if args.metric=="comet" else "f1" if "bertscore_f1" else "precision" if "bertscore_precision" else "recall" if "bertscore_recall" else args.metric,
    generation_num_beams=args.num_beams,
    #hightlights=args.highlights, 
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_dev if args.eval == "dev" else tokenized_dataset_test,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
)

if args.train:
    trainer.train()
trainer.evaluate()

# Save our tokenizer and create model card
tokenizer.save_pretrained(repository_id)
trainer.create_model_card()