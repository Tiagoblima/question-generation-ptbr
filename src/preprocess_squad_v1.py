import os 
import pandas as pd 
import datasets as dts
import nltk 
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from metrics import rouge, bleu
import evaluate
from tqdm import tqdm
BASE_DIR="/home/tiagoblima/repos/question-generation-ptbr/reports/tiagoblima"
sent_bleu = []

test_set = dts.load_dataset("tiagoblima/preprocessed-du-qg-squadv1_pt", split="test")

rougel = evaluate.load("rouge")
rouge = rouge.Rouge()
for model_dir in os.listdir(BASE_DIR):

    fpath = os.path.join(BASE_DIR, model_dir)
    if os.path.isfile(fpath): continue
    lines = open(os.path.join(fpath, "hypothesis.txt")).readlines()
    for example, hyp in tqdm(zip(test_set, lines), total=len(lines), desc=model_dir):
        bleu_score = bleu.get_corpus_bleu([example["question"]],
                       [hyp])['Bleu_4']
        #rouge_score = rougel.compute(predictions=[hyp], references=[example["question"]])
        # bleu_score = sentence_bleu([word_tokenize(example["question"])],
        #                            word_tokenize(hyp))
        cands = dict(zip([1], [[" ".join(word_tokenize(hyp.strip("\n")))]]))
        refs = dict(zip([1], [[" ".join(word_tokenize(example["question"]))]]))
        avg_rouge, rouge_score = rouge.compute_score(refs,cands)

        sent_bleu.append({
            "ref":example["question"],
            "paragraph": example["paragraph_answer"],
            "hyp": hyp,
            "bleu": bleu_score,
            "rouge": avg_rouge,
            "model": model_dir
        })

dataset = pd.DataFrame(sent_bleu).sort_values(by="bleu", ascending=False )

dataset.to_csv("sent_bleu.csv", index=False, sep="\t")