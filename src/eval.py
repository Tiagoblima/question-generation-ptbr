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
@click.option("--answer_column", type=str, default="answer")
@click.option("--context_column", type=str, default="paragraph")
@click.option("--question_column", type=str, default="question")
@click.option("--split_name", type=str, default="validation")
@click.option("-bs", "--batch_size", type=int, default=32)
def main(model_name,
         dataset_name,
         metrics, 
         answer_column,
         context_column,
         question_column,
         split_name,
         batch_size
         ):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    eval_ds = dts.load_dataset(dataset_name, split=split_name)

    metric_list = metrics.split(",")

    def generate_input(_answer, _context):
        return " ".join(["answer:", _answer.lstrip(), "context:", _context.lstrip(), ])
       
        
    def predict(batch):
        
        text_inputs = [generate_input(answer, context) 
                    for answer, context in zip(batch[answer_column], batch[context_column])]
        

        model_inputs = tokenizer(text_inputs,
                                  max_length=model.config.max_length, 
                                  padding=True, truncation=True, return_tensors="pt")
        # Tokenize targets with text_target=...
        for inps in model_inputs:
            model_inputs[inps] =  model_inputs[inps].to(device)
        outputs_ids = model.generate(**model_inputs)

        batch.update({
            "predicted": tokenizer.batch_decode(outputs_ids)
        })
        return batch

    predict_ds = eval_ds.map(predict, batch_size=batch_size, batched= batch_size > 1)

    hypothesis = np.array(predict_ds["predicted"])
    references = np.expand_dims(np.array(predict_ds[question_column]), axis=0)
    
    result_dict = {}
    for metric_name in metric_list:
        metric = dts.load_metric(metric_name)

        metric_dict = metric.compute(predictions=,
                                      references=[: None])

        if "score" in metric_dict:
            result_dict[metric_name] = metric_dict["score"]
        else:
            result_dict.update(metric_dict)
    print(result_dict)
    json.dump(result_dict, open('results.json', "w"), indent=4)

if __name__ == "__main__":
    main()