from lmqg import evaluate
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
@click.option("-d", "dataset_name", type=str, default="tiagoblima/qg_squad_v1_pt")
@click.option("-i","--input_names", type=str, default="paragraph_answer")
@click.option("-hyp","--hyp_test", type=str, default=None)
@click.option("-o","--output_dir", type=str, default="validation")
@click.option("-t","--target_name", type=str, default="question")
@click.option("--metrics", type=str, default="sacrebleu")
@click.option("--split_name", type=str, default="test")
@click.option("-bs", "--batch_size", type=int, default=16)
@click.option("-ml", "--max_new_tokens", type=int, default=96)
@click.option("--num_beams", type=int, default=5)
@click.option("--prediction_level", type=int, default="paragraph")
@click.option("--lang", type=str, default='pt')
def main(model_name,
         dataset_name,
         hyp_test, 
         output_dir,
         input_names,
         target_name,
         split_name,
         batch_size,
         max_new_tokens,
         num_beams,
         prediction_level,
         lang
         ):
    
    metric = evaluate(
        export_dir=output_dir,
        batch_size=batch_size,
        n_beams=num_beams,
        hypothesis_file_dev=None,
        hypothesis_file_test=hyp_test,
        model=model_name,
        max_length=max_new_tokens,
        max_length_output=max_new_tokens,
        dataset_path=None,
        dataset_name=dataset_name,
        input_type=input_names,
        output_type=target_name,
        prediction_aggregation="first",
        prediction_level=prediction_level,
        overwrite=True,
        language=lang,
        bleu_only=False,
        use_auth_token=False,
        test_split=split_name,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        low_cpu_mem_usage=False
    )
    print(json.dumps(metric, indent=4, sort_keys=True))

if __name__ == "__main__":
    main()