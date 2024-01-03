import evaluate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import click
import datasets as dts
import json
import torch
import numpy as np 


@click.command()
@click.option("-m", "model_name", type=str)
@click.option("-d", "dataset_name", type=str)
@click.option("--metrics", type=str, default="sacrebleu")
@click.option("-i","--input_name", type=str, default="paragraph_answer")
@click.option("-t","--target_name", type=str, default="question")
@click.option("--split_name", type=str, default="validation")
@click.option("-bs", "--batch_size", type=int, default=16)
@click.option("--bs_model_type", type=str, default='neuralmind/bert-base-portuguese-cased')
def main(model_name,
         dataset_name,
         metric_list, 
         input_name,
         target_name,
         split_name,
         batch_size,
         bs_model_type
         ):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    eval_ds = dts.load_dataset(dataset_name, split=split_name)

    def predict(batch):
        
        text_inputs = batch[input_name]
        

        model_inputs = tokenizer(text_inputs,
                                  max_length=model.config.max_length, 
                                  padding=True, truncation=True, return_tensors="pt")
        # Tokenize targets with text_target=...
        for inps in model_inputs:
            model_inputs[inps] =  model_inputs[inps].to(device)
        outputs_ids = model.generate(**model_inputs)

        batch.update({
            "predicted": tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
        })
        return batch

    predict_ds = eval_ds.map(predict, batch_size=batch_size, batched= batch_size > 1)

    hypothesis = np.array(predict_ds["predicted"])
    references = np.expand_dims(np.array(predict_ds[target_name]), axis=1)
    
    result_dict = {}
    for metric_name in metric_list.split(","):
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