from dostoevsky.models import FastTextSocialNetworkModel
from dostoevsky.tokenization import RegexTokenizer

from typing import List
import pandas as pd


class VocabSentimentAnalyzer:
    def __init__(self, dict_path:str):
        self.emo_dict = pd.read_csv(dict_path, sep=";")

    def estimate_sentiment_by_vocab(self, noun_tokens: List[str]):
        emotion = 0
        for w in noun_tokens:
            if "NGTV" in self.emo_dict[self.emo_dict.term == w].tag.to_list():
                emotion -= 1
            elif "PSTV" in self.emo_dict[self.emo_dict.term == w].tag.to_list():
                emotion += 1
            else:
                continue
        return emotion

    def aplly_df(self, df, adj_extractor):
        filtered_tokens = adj_extractor(df.text.apply)
        df["vocab_sentimnet"] = [self.estimate_sentiment_by_vocab(x) for x in filtered_tokens]
        return df

    # sns.displot(data=dev, x="dict_emo", hue="label", kind="kde")

class DostoevskyAnalyzer:
    def __init__(self):
        tokenizer = RegexTokenizer()
        self.model = FastTextSocialNetworkModel(tokenizer=tokenizer)

    def get_sentiment(self, text):
        results = self.model.predict(text, k=1)
        label = []
        score = []
        for x, y in list([list(x.items())[0] for x in results]):
            label.append(x)
            score.append(y)
        return label, score

    def apply_df(self, df, write_scores=False):
        labels, scores = self.get_sentiment(df.text)
        df["predicted_sentiment"] = labels
        if write_scores:
            df["sentiment_score"] = scores
        return df