import re
from pymystem3 import Mystem


class TextsTokenizer:
    """Tokenizer"""
    def __init__(self):
        self.m = Mystem()

    def texts2tokens(self, texts: [str]) -> [str]:
        """Lemmatization for texts in list. It returns list with lemmatized texts."""
        text_ = "\n".join(texts)
        text_ = re.sub(r"[^\w\n\s]", " ", text_)
        lm_texts = "".join(self.m.lemmatize(text_))
        return [lm_q.split() for lm_q in lm_texts.split("\n")][:-1]

    def __call__(self, texts: [str]):
        return self.texts2tokens(texts)

