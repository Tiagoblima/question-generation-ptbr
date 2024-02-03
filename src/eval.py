import click
import datasets as dts
from metrics import bleu
from metrics.rouge import Rouge
from nltk.tokenize import word_tokenize
import pandas as pd 

langdict = {
    "pt": "portuguese"
}

@click.command()
@click.option("-d", "--dataset_name", type=str, default="tiagoblima/du-qg-squadv1_pt")
@click.option("-r", "--ref_file", type=str, default=None)
@click.option("-p", "--pred_file", type=str, default="hypothesis.txt")
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
        refs = [ref.strip("\n") for ref in open(ref_file).readlines()]


    candidates = [cand.strip("\n") for cand in open(pred_file).readlines()]
    
    scores = bleu.get_corpus_bleu(refs, candidates, language=langdict[lang])
    
    
    full_outpath = output_dir + ".csv"
    rouge = Rouge()
    cands = dict(zip(list(range(len(candidates))), [[" ".join(word_tokenize(cand))] for cand in candidates]))
    refs = dict(zip(list(range(len(candidates))), [[" ".join(word_tokenize(ref))] for ref in refs]))
    rouge_score, _ = rouge.compute_score(refs,cands)
    scores.update({
        "rougeL": rouge_score
    })
    scores["model"] = full_outpath.split("/")[-1]
    metrics_df = pd.DataFrame.from_dict(scores, orient="index").T.loc[:, ["model", "Bleu_4","rougeL"]]
    print(metrics_df.to_csv(header=False))
    metrics_df.to_csv(full_outpath,index=False)
    

if __name__ == "__main__":
    main()