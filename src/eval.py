import click
import datasets as dts
import os
from metrics import bleu
import json 
from metrics.rouge import Rouge
import evaluate

langdict = {
    "pt": "portuguese"
}

@click.command()
@click.option("-d", "--dataset_name", type=str, default="tiagoblima/du-qg-squadv1_pt")
@click.option("-r", "--ref_file", type=str, default=None)
@click.option("-p", "--pred_file", type=str, default="hypothesis.txt")
@click.option("-i","--input_names", type=str, default="paragraph,answer")
@click.option("-o","--output_dir", type=str, default="results")
@click.option("-t","--target_name", type=str, default="question")
@click.option("--split_name", type=str, default="test")
@click.option("--lang", type=str, default='pt')
def main(
         dataset_name,
         ref_file,
         pred_file,
         output_dir,
         target_name,
         split_name,
         lang
         ):
    
    if not ref_file:
        refs = dts.load_dataset(dataset_name)[split_name][target_name]
    else:
        refs = open(ref_file).readlines()

    
    candidates = open(pred_file).readlines()
    
    scores = bleu.get_corpus_bleu(refs, candidates, language=langdict[lang])
    
    os.makedirs(output_dir, exist_ok=True)
    full_outpath = os.path.join(output_dir, "metrics.json")
   
    rouge = evaluate.load('rouge').compute(predictions=candidates,
                         references=refs)
    
    scores.update(rouge)
    scores.update({
        "rougeL": Rouge().calc_score(candidates, refs)
    })
    print(scores)
    json.dump(scores, open(full_outpath, "w"), indent=4)

    
    

if __name__ == "__main__":
    main()