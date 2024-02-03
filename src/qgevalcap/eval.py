#!/usr/bin/env python
__author__ = 'xinya'

from bleu.bleu import Bleu
from rouge.rouge import Rouge
from collections import defaultdict
from argparse import ArgumentParser
from text_normalization import text_normalization
import pandas as pd
import sys
# reload(sys)

# import importlib,sys
# importlib.reload(sys)

# sys.setdefaultencoding('utf-8')

class QGEvalCap:
    def __init__(self, gts, res):
        self.gts = gts
        self.res = res

    def evaluate(self):
        output = []
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
          
            (Rouge(), "ROUGE_L"),
           
        ]

        # =================================================
        # Compute scores
        # =================================================
        report = {}
        for scorer, method in scorers:
            # print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(self.gts, self.res)
            if type(method) == list:
                report.update(dict(zip( method, score)))
                for sc, scs, m in zip(score, scores, method):
                    print ("%s: %0.5f"%(m, sc))
                    output.append(sc)
            else:
                print ("%s: %0.5f"%(method, score))
                report.update({
                    method: score
                })
                output.append(score)

        return report

def eval(out_file, src_file, tgt_file, isDIn = False, num_pairs = 500):
    """
        Given a filename, calculate the metric scores for that prediction file

        isDin: boolean value to check whether input file is DirectIn.txt
    """

    pairs = []
    with open(src_file, 'r') as infile:
        for line in infile:
            pair = {}
            pair['tokenized_sentence'] = line[:-1]
            pairs.append(pair)

    with open(tgt_file, "r") as infile:
        cnt = 0
        for line in infile:
            pairs[cnt]['tokenized_question'] = line[:-1]
            cnt += 1

    output = []
    with open(out_file, 'r') as infile:
        for line in infile:
            line = line[:-1]
            output.append(line)


    for idx, pair in enumerate(pairs):
        pair['prediction'] = output[idx]


    ## eval
    from eval import QGEvalCap
    import json
    from json import encoder
    encoder.FLOAT_REPR = lambda o: format(o, '.4f')

    res = defaultdict(lambda: [])
    gts = defaultdict(lambda: [])
    for pair in pairs[:]:
        key = pair['tokenized_sentence']
      
        res[key] = [text_normalization(pair['prediction']).encode('utf-8')]

        ## gts 
        gts[key].append(text_normalization(pair['tokenized_question']).encode('utf-8'))

    QGEval = QGEvalCap(gts, res)
    return QGEval.evaluate()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-out", "--out_file", dest="out_file", default="../../reports/tiagoblima/t5_base-qg-aap-oficial/hypothesis.txt", help="output file to compare")
    parser.add_argument("-src", "--src_file", dest="src_file", default="../../data/du-squadv1/src-test.txt", help="src file")
    parser.add_argument("-tgt", "--tgt_file", dest="tgt_file", default="../../data/du-squadv1/tgt-test.txt", help="target file")
    args = parser.parse_args()

    print ("scores: \n")
    report = eval(args.out_file, args.src_file, args.tgt_file)

    report.update({
        "model": args.out_file.split("/")[-2]
    })
    report_df = pd.DataFrame.from_dict(report, orient='index').T
    print(report_df)


