import os 
import pandas as pd 
import datasets as dts
from qgevalcap import eval
BASE_DIR="/home/tiagoblima/repos/question-generation-ptbr/data/du-squadv1"

test_set = dts.load_dataset("tiagoblima/preprocessed-du-qg-squadv1_pt", split="test")

open("src-sent.txt", "w").writelines([ex["paragraph"] + "\n" for ex in test_set ])

for filename in os.listdir(os.path.join(BASE_DIR, "preds")):

    fpath = os.path.join(BASE_DIR, "preds", filename)
    if os.path.isfile(fpath): 
        lines = open(os.path.join(fpath)).readlines()
   

dataset = pd.DataFrame(sent_bleu).sort_values(by="bleu", ascending=False )

dataset.to_csv("sent_bleu.csv", index=False, sep="\t")