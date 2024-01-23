import click
import datasets as dts
import numpy as np 
import os
from metrics import bleu
import json 

langdict = {
    "pt": "portuguese"
}

@click.command()
@click.option("-m", "--model_name", type=str)
@click.option("-d", "--dataset_name", type=str, default="tiagoblima/qg_squad_v1_pt")
@click.option("-r", "--ref_file", type=str, default=None)
@click.option("-p", "--pred_file", type=str, default="hypothesis.txt")
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
         ref_file,
         pred_file,
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
    
    if not ref_file:
        refs = dts.load_dataset(dataset_name)[split_name][target_name]
    else:
        refs = open(ref_file).readlines()

    
    candidates = open(pred_file).readlines()
    print(len(candidates), len(refs))
    bleu_scores = bleu.get_corpus_bleu(refs, candidates, language=langdict[lang])
    print(bleu_scores)
    os.makedirs(output_dir, exist_ok=True)
    full_outpath = os.path.join(output_dir, "metrics.json")
    json.dump(bleu_scores, open(full_outpath, "w"), indent=4)

    
    

if __name__ == "__main__":
    main()