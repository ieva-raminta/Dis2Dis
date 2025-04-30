from random import randrange
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback, 
    IntervalStrategy,
    GenerationConfig,
    T5ForConditionalGeneration,
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
import tqdm 
from supervised_correction_reader import SupervisedCorrectionReader

# get arguments from the command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--metric", type=str, default="keep")
parser.add_argument("--resume_from_checkpoint", type=bool, default=False)
args = parser.parse_args()

nltk.download("punkt")

model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def flatten_list(l):
    return [item for sublist in l for item in sublist]

train_file = "../acl2021_factual_error_correction/resources/mutation/train.jsonl"
dev_file = "../acl2021_factual_error_correction/resources/mutation/dev.jsonl"  
test_file = "../acl2021_factual_error_correction/resources/mutation/test.jsonl"

reader = SupervisedCorrectionReader(set(["SUPPORTS", "REFUTES"]), True) # not sure what test does (the True value)
train_instance_generator = reader.read(train_file)
dev_instance_generator = reader.read(dev_file)
test_instance_generator = reader.read(test_file)

train_instances = list(filter(lambda i: i is not None, train_instance_generator))
dev_instances = list(filter(lambda i: i is not None, dev_instance_generator))
test_instances = list(filter(lambda i: i is not None, test_instance_generator))

def gen_train():
    for train_item in train_instances: 
        yield {
               "source":train_item["source"], 
               "evidence": train_item["evidence"], 
               "target":train_item["target"], 
            }
def gen_dev():
    for dev_item in dev_instances: 
        yield {
               "source":dev_item["source"], 
               "evidence": dev_item["evidence"], 
               "target":dev_item["target"], 
            }
def gen_test():
    for test_item in test_instances: 
        yield {
               "source":test_item["source"], 
               "evidence": test_item["evidence"], 
               "target":test_item["target"], 
            }

pre_train_dataset = Dataset.from_generator(gen_train)
pre_dev_dataset = Dataset.from_generator(gen_dev)
pre_test_dataset = Dataset.from_generator(gen_test)

# subset datasets with randrange
#dis_train_dataset = dis_train_dataset.select(range(100))
#dis_dev_dataset = dis_dev_dataset.select(range(100))

tokenized_inputs = concatenate_datasets([pre_train_dataset, pre_dev_dataset]).map(
    lambda x: tokenizer(x["source"], x["evidence"], truncation="only_second"),
    batched=True,
    remove_columns=["source", "evidence", "target"],
)
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
tokenized_targets = concatenate_datasets([pre_train_dataset, pre_dev_dataset]).map(
    lambda x: tokenizer(x["target"], truncation=True),
    batched=True,
    remove_columns=["source", "evidence", "target"],
)
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])

def preprocess_function(sample,padding="max_length"):

    instruction = "Please correct the following claim to be supported by the following evidence: "

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
    return model_inputs

tokenized_dataset_train = pre_train_dataset.map(
    preprocess_function, batched=True, remove_columns=["source", "evidence", "target"]
)
tokenized_dataset_dev = pre_dev_dataset.map(
    preprocess_function, batched=True, remove_columns=["source", "evidence", "target"]
)

print(f"Keys of tokenized dataset: {list(tokenized_dataset_train.features)}")

if args.resume_from_checkpoint: 
    # iterate through directory names in /home/irs38/disambiguation/flan-t5-base-pretraining-disambiguation/
    checkpoints = []
    for directory in os.listdir("/home/irs38/disambiguation/flan-t5-base-pretraining-disambiguation/"):
        if directory.startswith("checkpoint-"):
            checkpoints.append(directory)
    # get the latest checkpoint
    latest_checkpoint = max(checkpoints)
    model = T5ForConditionalGeneration.from_pretrained("/home/irs38/disambiguation/flan-t5-base-pretraining-disambiguation/"+latest_checkpoint)
else: 
    model = T5ForConditionalGeneration.from_pretrained(model_id)
generation_config = GenerationConfig(decoder_start_token_id=0, eos_token_id=1, pad_token_id=0, max_new_tokens=1000)
model.generation_config = generation_config
tokenizer.generation_config = generation_config

# Metric
get_sari_score.name = "sari"
metrics = [evaluate.load("rouge"), evaluate.load("bertscore"), evaluate.load("comet"), get_sari_score]

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
    sources = [s for s in pre_dev_dataset["source"]]
    sources_with_evidence = [s + "</s>" + e for s, e in zip(pre_dev_dataset["source"], pre_dev_dataset["evidence"])]

    tokenized_input_ids = [tokenizer(item, padding=False, truncation=True)["input_ids"] for item in sources_with_evidence]
    source_ids = [tokenizer(item, padding=False, truncation=False)["input_ids"] for item in sources]
    pred_ids = [[tok for tok in pred if tok != tokenizer.pad_token_id] for pred in preds]
    target_ids = [[tok for tok in target if tok != tokenizer.pad_token_id] for target in labels]

    # Some simple post-processing
    decoded_preds, decoded_labels, sources = postprocess_text(decoded_preds, decoded_labels, sources)

    results = {}
    baseline_results = {}
    for metric in metrics: 
        if metric.name == "bert_score":
            result = metric.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
            result = {"f1": (2 * np.mean(result["precision"]) * np.mean(result["recall"])) / (np.mean(result["precision"]) + np.mean(result["recall"]))}
            baseline_result = metric.compute(predictions=sources, references=decoded_labels, lang="en")
            baseline_result = {"f1": (2 * np.mean(baseline_result["precision"]) * np.mean(baseline_result["recall"])) / (np.mean(baseline_result["precision"]) + np.mean(baseline_result["recall"]))}
        elif metric.name == "rouge":
            result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            baseline_result = metric.compute(predictions=sources, references=decoded_labels, use_stemmer=True) 
        elif metric.name == "comet":
            result = metric.compute(predictions=decoded_preds, references=decoded_labels, sources=sources_with_evidence)
            del result["scores"]
            baseline_result = metric.compute(predictions=sources, references=decoded_labels, sources=sources_with_evidence)
            del baseline_result["scores"]
        elif metric.name == "sari":
            result = []
            baseline_result = []
            for item_id in range(len(pred_ids)):
                pred_id = pred_ids[item_id]
                source_id = source_ids[item_id]
                tokenized_input_id = tokenized_input_ids[item_id]
                target_id = target_ids[item_id]

                result.append(metric(prediction_ids=pred_id, list_of_targets=[target_id], source_ids=tokenized_input_id))
                baseline_result.append(metric(prediction_ids=source_id, list_of_targets=[target_id], source_ids=tokenized_input_id))
            result = {"sari": np.mean([r[0] for r in result]), "keep": np.mean([r[1] for r in result]), "add": np.mean([r[2] for r in result]), "delete": np.mean([r[3] for r in result])}
            result["keepandadd"] = result["keep"] + result["add"]
            baseline_result = {"sari": np.mean([r[0] for r in baseline_result]), "keep": np.mean([r[1] for r in baseline_result]), "add": np.mean([r[2] for r in baseline_result]), "delete": np.mean([r[3] for r in baseline_result])}
            baseline_result["keepandadd"] = baseline_result["keep"] + baseline_result["add"]

        result = {k: round(v * 100, 4) for k, v in result.items()}
        baseline_result = {k: round(v * 100, 4) for k, v in baseline_result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
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
        list_for_saving.append({"prediction": pred, "reference": decoded_labels[count], "source": sources[count]})
    # save the list_for_saving into jsonl line by line
    model_name = model_id.split("/")[1]
    epoch = str(trainer.state.epoch).split(".")[0] if str(trainer.state.epoch).split(".")[1]=="0" else int(str(trainer.state.epoch).split(".")[0])+1
    metric = args.metric
    learning_rate = training_args.learning_rate
    with open(f"generations_pretrained_model/predictions_{metric}_{model_name}_lr{learning_rate}_epoch{epoch}.jsonl", "w") as f:
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
repository_id = f"{model_id.split('/')[1]}-pretraining-disambiguation"

batch_size = 8

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
    save_steps=len(pre_train_dataset)//batch_size,
    eval_steps=len(pre_train_dataset)//batch_size,
    save_total_limit=1,
    load_best_model_at_end=True,
    greater_is_better=True,
    report_to="tensorboard",
    push_to_hub=False,
    hub_strategy="every_save",
    hub_model_id=repository_id,
    hub_token=HfFolder.get_token(),
    metric_for_best_model = args.metric,
)


# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_dev,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
)

if args.resume_from_checkpoint:
    trainer.train(resume_from_checkpoint=True)
else: 
    trainer.train()
trainer.evaluate()

# Save our tokenizer and create model card
tokenizer.save_pretrained(repository_id)
trainer.create_model_card()
