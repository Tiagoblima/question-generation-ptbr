import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def get_corpus_bleu(references, predictions, lower_case=True, language="portuguese"):
    list_of_references = []
    hypotheses = []

    for ref in references:
        ref_processed = word_tokenize(ref, language=language) # tokenize
        if lower_case:
            ref_processed = [each_string.lower() for each_string in ref_processed] # lowercase
        list_of_references.append([ref_processed])

    for pred in predictions:
        pred_processed = word_tokenize(pred, language=language) # tokenize
        if lower_case:
            pred_processed = [each_string.lower() for each_string in pred_processed] # lowercase
        hypotheses.append(pred_processed)

    bleu_1 = corpus_bleu(list_of_references, hypotheses, weights = [1,0,0,0])
    bleu_2 = corpus_bleu(list_of_references, hypotheses, weights = [0.5,0.5,0,0])
    bleu_3 = corpus_bleu(list_of_references, hypotheses, weights = [1/3,1/3,1/3,0])
    bleu_4 = corpus_bleu(list_of_references, hypotheses, weights = [0.25,0.25,0.25,0.25])

    return {"Bleu_1": bleu_1, "Bleu_2": bleu_2, "Bleu_3": bleu_3, "Bleu_4": bleu_4}
