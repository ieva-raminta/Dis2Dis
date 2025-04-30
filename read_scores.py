import os
import pandas as pd 
import json 

# create an empty pandas dataframe
df_all = pd.DataFrame(columns=["length", 'sari', 'sari_keep', 'sari_add', 'sari_delete', 'comet', 'bertscore_f1', 'bertscore_precision', 'bertscore_recall', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum'])
df_ambiguous = pd.DataFrame(columns=["length", 'sari', 'sari_keep', 'sari_add', 'sari_delete', 'comet', 'bertscore_f1', 'bertscore_precision', 'bertscore_recall', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum'])
df_baseline = pd.DataFrame(columns=["length", 'sari', 'sari_keep', 'sari_add', 'sari_delete', 'comet', 'bertscore_f1', 'bertscore_precision', 'bertscore_recall', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum'])

from transformers import AutoTokenizer
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
with open("data/dev_correction.jsonl") as f:
    dev_correction = [json.loads(line) for line in f]
with open("data/dev_correction_ambiguous.jsonl") as f:
    dev_correction_ambiguous = [json.loads(line) for line in f]
ambiguous_ids = [i for i in range(len(dev_correction)) if dev_correction[i]["distance"] in [1,2,3]]
mean_len_all = sum([len(tokenizer.tokenize(x["target"][0])) for x in dev_correction])/len(dev_correction)
mean_len_ambiguous = sum([len(tokenizer.tokenize(x["target"][0])) for x in dev_correction_ambiguous])/len(dev_correction_ambiguous)
# read the files from llm_generations directory: ambiguous_eight_shot_generations.txt
with open("llm_generations/ambiguous_eight_shot_generations.txt") as f:
    ambiguous_eight_shot_generations = f.readlines()
    ambiguous_eight_shot_generations = [ambiguous_eight_shot_generations[i] for i in ambiguous_ids]
with open("llm_generations/ambiguous_four_shot_generations.txt") as f:
    ambiguous_four_shot_generations = f.readlines()
    ambiguous_four_shot_generations = [ambiguous_four_shot_generations[i] for i in ambiguous_ids]
with open("llm_generations/zero_shot_generations.txt") as f:
    zero_shot_generations = f.readlines()
    ambiguous_zero_shot_generations = [zero_shot_generations[i] for i in ambiguous_ids]
with open("llm_generations/eight_shot_generations.txt") as f:
    eight_shot_generations = f.readlines()
with open("llm_generations/four_shot_generations.txt") as f:
    four_shot_generations = f.readlines()
ambiguous_eight_shot_generations_length = sum([len(tokenizer.tokenize(x)) for x in ambiguous_eight_shot_generations])/len(ambiguous_eight_shot_generations)
ambiguous_four_shot_generations_length = sum([len(tokenizer.tokenize(x)) for x in ambiguous_four_shot_generations])/len(ambiguous_four_shot_generations)
ambiguous_zero_shot_generations_length = sum([len(tokenizer.tokenize(x)) for x in ambiguous_zero_shot_generations])/len(ambiguous_zero_shot_generations)
zero_shot_generations_length = sum([len(tokenizer.tokenize(x)) for x in zero_shot_generations])/len(zero_shot_generations)
eight_shot_generations_length = sum([len(tokenizer.tokenize(x)) for x in eight_shot_generations])/len(eight_shot_generations)
four_shot_generations_length = sum([len(tokenizer.tokenize(x)) for x in four_shot_generations])/len(four_shot_generations)

# iterate through the files in the current directory
for filename in os.listdir('disambiguation_scores/'):
    # if the file endswith .output
    if filename.endswith('.output') and not filename.endswith("test.output"):
        lines = []
        last_line = ""
        # read each line as a json dict
        with open("disambiguation_scores/"+filename) as f:
            for line in f:
                if line.startswith('{') and "eval_loss" not in line and "train_runtime" not in line and "learning_rate" not in line:
                    # replace ' with " in line
                    line = line.replace("'", "\"").strip()
                    lines.append(json.loads(line))
                if "generations_disambiguation_model" in line:
                    last_line = line
        if last_line: 
            last_line_end = last_line.split("epoch")[-1]
            epoch = int(last_line_end.split(".jsonl")[0].strip())
        else: 
            epoch = int(len(lines)/8)
        model_index = 8*(epoch-1)
        baseline_index = 8*(epoch-1) + 1
        if len(lines)> 8 and model_index+6 >= len(lines): 
            import pdb; pdb.set_trace()
        if lines and len(lines) > 8 and "sari" in lines[model_index+6] and "sari" in lines[baseline_index+6]:
            sari = lines[model_index+6]["sari"]
            sari_keep = lines[model_index+6]["keep"]
            sari_add = lines[model_index+6]["add"]
            sari_delete = lines[model_index+6]["delete"]
            comet = lines[model_index+4]["mean_score"]
            bertscore_f1 = lines[model_index+2]["f1"]
            if "precision" in lines[model_index+2]:
                bertscore_precision = lines[model_index+2]["precision"]
                bertscore_recall = lines[model_index+2]["recall"]
            else: 
                bertscore_precision = None
                bertscore_recall = None
            gen_len = lines[model_index+2]["gen_len"]
            rouge1 = lines[model_index]["rouge1"]
            rouge2 = lines[model_index]["rouge2"]
            rougeL = lines[model_index]["rougeL"]
            rougeLsum = lines[model_index]["rougeLsum"]
            sari_baseline = lines[baseline_index+6]["sari"]
            sari_keep_baseline = lines[baseline_index+6]["keep"]
            sari_add_baseline = lines[baseline_index+6]["add"]
            sari_delete_baseline = lines[baseline_index+6]["delete"]
            comet_baseline = lines[baseline_index+4]["mean_score"]
            bertscore_f1_baseline = lines[baseline_index+2]["f1"]
            if "precision" in lines[baseline_index+2]:
                bertscore_precision_baseline = lines[baseline_index+2]["precision"]
                bertscore_recall_baseline = lines[baseline_index+2]["recall"]
            gen_len_baseline = lines[baseline_index+2]["gen_len"]
            rouge1_baseline = lines[baseline_index]["rouge1"]
            rouge2_baseline = lines[baseline_index]["rouge2"]
            rougeL_baseline = lines[baseline_index]["rougeL"]
            rougeLsum_baseline = lines[baseline_index]["rougeLsum"]
            # save results to the dataframe with filename as the index
            if "ambiguous" in filename:
                df_ambiguous.loc[filename.replace(".output", "")] = [gen_len, sari, sari_keep, sari_add, sari_delete, comet, bertscore_f1, bertscore_precision, bertscore_recall, rouge1, rouge2, rougeL, rougeLsum]
            else: 
                df_all.loc[filename.replace(".output", "")] = [gen_len, sari, sari_keep, sari_add, sari_delete, comet, bertscore_f1, bertscore_precision, bertscore_recall, rouge1, rouge2, rougeL, rougeLsum]
            df_baseline.loc[filename.replace(".output", "") + "_baseline"] = [gen_len_baseline, sari_baseline, sari_keep_baseline, sari_add_baseline, sari_delete_baseline, comet_baseline, bertscore_f1_baseline, bertscore_precision_baseline, bertscore_recall_baseline, rouge1_baseline, rouge2_baseline, rougeL_baseline, rougeLsum_baseline]
            
import pdb; pdb.set_trace()
