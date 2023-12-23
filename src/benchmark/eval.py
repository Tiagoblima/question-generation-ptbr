import evaluate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import click
import datasets as dts



@click.command()
@click.option("-m", "model_name", type=str)
@click.option("-ds", "dataset_name", type=str)
@click.option("--metrics", type=str, default="sacrebleu")
@click.option("--answer_column", type=str, default="answer")
@click.option("--context_column", type=str, default="paragraph")
@click.option("--question_column", type=str, default="question")
@click.option("--eval_split", type=str, default="evaluation")
def main(model_name,
         dataset_name,
         metrics, 
         answer_column,
         context_column,
         question_column,
         eval_split
         ):

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    eval_ds = dts.load_dataset(dataset_name, split=eval_split)

    metric_list = metrics.split(",")

    def generate_input(_answer, _context):
        return " ".join(["answer:", _answer.lstrip(), "context:", _context.lstrip(), ])
       
        
    def predict(batch):
        
        text_inputs = [generate_input(answer, context) 
                    for answer, context in zip(batch[answer_column], batch[context_column])]
        

        model_inputs = tokenizer(text_inputs, max_length=model.config.max_length, padding=True, truncation=True)
        # Tokenize targets with text_target=...

        outputs_ids = model.generate(**model_inputs)

        batch.update({
            "predicted": tokenizer.batch_decode(outputs_ids)
        })
        return batch

    predict_ds = eval_ds.map(predict)

    result_dict = {}
    for metric_name in metric_list:
        metric = evaluate.load(metric_name)

        metric_dict = metric.compute(predict_ds["predicted"],
                                      predict_ds[question_column])

        if "score" in metric_dict:
            result_dict[metric_name] = metric_dict["score"]
        else:
            result_dict.update(metric_dict)


if __name__ == "__main__":
    main()