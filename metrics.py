from bert_score import score
import nltk
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

nltk.download(['punkt_tab', 'wordnet'])


def calculate_bleu_score(predictions, references):
    bleu_scores = []
    smoothing_function = SmoothingFunction().method1
    for prediction, reference in zip(predictions, references):
        reference_tokens = reference.split()
        prediction_tokens = prediction.split()
        bleu = sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=smoothing_function)
        bleu_scores.append(bleu)
    return bleu_scores

def calculate_meteor_score(predictions, references):
    meteor_scores = []
    for prediction, reference in zip(predictions, references):
        reference_tokens = word_tokenize(reference)
        
        prediction_tokens = word_tokenize(prediction)
        meteor = meteor_score([reference_tokens], prediction_tokens)
        meteor_scores.append(meteor)
    return meteor_scores