import evaluate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import click
import datasets as dts
import json
import torch
import numpy as np 
import os
# 1-gram individual BLEU
from nltk.translate.bleu_score import sentence_bleu


@click.command()
@click.option("-m", "model_name", type=str)
@click.option("-d", "dataset_name", type=str, default="tiagoblima/du-qg-squadv1_pt")
@click.option("-i","--input_names", type=str, default="paragraph,answer")
@click.option("-o","--output_dir", type=str, default="validation")
@click.option("-t","--target_name", type=str, default="question")
@click.option("--metrics", type=str, default="sacrebleu")
@click.option("--split_name", type=str, default="test")
@click.option("-bs", "--batch_size", type=int, default=16)
@click.option("-ml", "--max_new_tokens", type=int, default=96)
@click.option("--num_beams", type=int, default=5)
@click.option("--num_proc", type=int, default=1)
@click.option("--lang", type=str, default='pt')
def main(model_name,
         dataset_name,
         metrics, 
         output_dir,
         input_names,
         target_name,
         split_name,
         batch_size,
         max_new_tokens,
         num_beams,
         num_proc,
         lang
         ):
    input_names = input_names.split(",")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    eval_ds = dts.load_dataset(dataset_name, split=split_name)

    def generate_input(tup_example):
          
            return f"{tokenizer.eos_token}".join(tup_example)

    def predict(examples):
        
        text_inputs = [examples[input_name] 
                            for input_name in input_names]
        text_inputs = [generate_input(example) for example in zip(*text_inputs)]
        
        model_inputs = tokenizer(text_inputs,
                                  max_length=model.config.max_length, 
                                  padding=True, truncation=False, return_tensors="pt")
        
        # Tokenize targets with text_target=...
        for inps in model_inputs:
            model_inputs[inps] =  model_inputs[inps].to(device)
        outputs_ids = model.generate(**model_inputs, num_beams=num_beams, max_new_tokens=max_new_tokens)

        examples.update({
            "predicted": tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
        })

        return examples

    predict_ds = eval_ds.map(predict,
                            
                            batch_size=batch_size,
                            batched= batch_size > 1, 
                            num_proc=num_proc if device == "cpu" and num_proc > 1 else None)

    hypothesis = np.array(predict_ds["predicted"])
    references = np.expand_dims(np.array(predict_ds[target_name]), axis=1)
    
    result_dict = {}
    for metric_name in metrics.split(","):
        metric = evaluate.load(metric_name)

        if metric_name == "bertscore":
            bert_scores = metric.compute(predictions=hypothesis, 
                                        references=references.squeeze(), 
                                        lang=lang)
            bert_scores.pop('hashcode') 
            for key in bert_scores:
                result_dict[f"avg_{key}"] = np.array(bert_scores[key]).mean()
            continue
          
        metric_dict = metric.compute(predictions=hypothesis,
                                      references=references)

        if "score" in metric_dict:
            result_dict[metric_name] = metric_dict["score"]
        else:
            result_dict.update(metric_dict)
    output_dir = os.path.join(output_dir, model_name) 
    os.makedirs(output_dir, exist_ok=True)
    json.dump(result_dict, open(f'{output_dir}/scores.json', "w"), indent=4)
    open(f'{output_dir}/hypothesis.txt', "w").writelines([hyp + "\n" for hyp in hypothesis])
if __name__ == "__main__":
    main()