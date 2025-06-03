from llm_agent_x.constants import LANGUAGE


from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.utils import get_stop_words


import math


def summarize(text, num_sentences=5):
    sentences = num_sentences
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    summarizer = Summarizer(Stemmer(LANGUAGE))
    summarizer.stop_words = get_stop_words(LANGUAGE)
    summary = " ".join(str(s) for s in summarizer(parser.document, sentences))
    return summary
