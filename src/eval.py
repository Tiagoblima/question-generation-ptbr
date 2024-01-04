import evaluate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import click
import datasets as dts
import json
import torch
import numpy as np 


@click.command()
@click.option("-m", "model_name", type=str)
@click.option("-d", "dataset_name", type=str, default="tiagoblima/qg_squad_v1_pt")
@click.option("--metrics", type=str, default="sacrebleu")
@click.option("-i","--input_names", type=str, default="answer,paragraph")
@click.option("-t","--target_name", type=str, default="question")
@click.option("--split_name", type=str, default="validation")
@click.option("-bs", "--batch_size", type=int, default=16)
@click.option("-ml", "--max_new_tokens", type=int, default=96)
@click.option("--num_beams", type=int, default=5)
@click.option("--bs_model_type", type=str, default='neuralmind/bert-base-portuguese-cased')
def main(model_name,
         dataset_name,
         metrics, 
         input_names,
         target_name,
         split_name,
         batch_size,
         max_new_tokens,
         num_beams,
         bs_model_type
         ):
    input_names = input_names.split(",")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    eval_ds = dts.load_dataset("tiagoblima/qg_squad_v1_pt", split=split_name)

    def generate_input(tup_example):
          
            return f"{tokenizer.eos_token}".join(tup_example)

    def predict(*text_inputs):
        
        text_inputs = [generate_input(example) for example in zip(*text_inputs)]
        
        model_inputs = tokenizer(text_inputs,
                                  max_length=model.config.max_length, 
                                  padding=True, truncation=True, return_tensors="pt")
        # Tokenize targets with text_target=...
        for inps in model_inputs:
            model_inputs[inps] =  model_inputs[inps].to(device)
        outputs_ids = model.generate(**model_inputs, num_beams=num_beams, max_new_tokens=max_new_tokens)

        batch.update({
            "predicted": tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
        })
        return batch

    predict_ds = eval_ds.map(predict,
                            input_columns=input_names,
                            batch_size=batch_size,
                            batched= batch_size > 1)

    hypothesis = np.array(predict_ds["predicted"])
    references = np.expand_dims(np.array(predict_ds[target_name]), axis=1)
    
    result_dict = {}
    for metric_name in metrics.split(","):
        metric = evaluate.load(metric_name)

        if metric_name == "bertscore":
            results_scores = metric.compute(predictions=hypothesis, 
                                        references=references.squeeze(), 
                                        model_type=bs_model_type)
            for key in results_scores:
                results_scores[f"avg_{key}"] = np.array(results_scores.pop(key)).mean()

            result_dict.update(results_scores)

        metric_dict = metric.compute(predictions=hypothesis,
                                      references=references)

        if "score" in metric_dict:
            result_dict[metric_name] = metric_dict["score"]
        else:
            result_dict.update(metric_dict)
    print(result_dict)
    json.dump(result_dict, open('results.json', "w"), indent=4)
    json.dump(dict(zip(hypothesis.tolist(), references.tolist())), open('predictions.json', "w"), indent=4)
if __name__ == "__main__":
    main()