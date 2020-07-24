import pandas as pd
import numpy as np

MNLI_PATH = "/home/ndg/users/jkurre/mnli/utils/multinli_1.0_train.jsonl"
GLOVE_PATH = "/home/ndg/users/jkurre/mnli/utils/embeddings/glove.6B.50d.txt"
LABEL_TO_INT = {'contradiction':1, 'entailment':2, 'neutral':3}

def load_mnli():
    # read data to pandas dataframe 
    mnli_data = pd.read_json(MNLI_PATH, lines=True)
    # combine pairs and map labels to ids
    mnli_data["sentence"] = mnli_data["sentence1"] + "<END_OF_PAIR>" + mnli_data["sentence2"]
    mnli_data["gold_label"] = mnli_data["gold_label"].apply(lambda label: LABEL_TO_INT[label])
    # split data into train, validation, and test set
    train, validate, test = np.split(
        mnli_data.sample(frac=1), [int(.6*len(mnli_data)),int(.8*len(mnli_data))]
    )
    # export to csv and return train, validation, and test set
    train.to_csv("train.csv")
    validate.to_csv("val.csv")
    test.to_csv("test.csv")
    return train, validate, test

def load_glove(vocabulary):
    """
    Wikipedia 2014 + Gigaword 5 vectors
    https://nlp.stanford.edu/projects/glove/
    """
    embeddings = {}
  
    with open(GLOVE_PATH, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word, vector = values[0], np.asarray(values[1:], "float32")
            if word in vocabulary:
                embeddings[word] = vector
    return embeddings

if __name__ == "__main__":
    load_mnli()