import nltk
import re
from sentence_graph import TextParser
from pymystem3 import Mystem


class NLTKTagger:
    def __init__(self) -> None:
        self.repr = []

    def init(self, texts):
        tokens = nltk.word_tokenize(texts)
        self.repr = nltk.pos_tag(tokens, lang="rus")

    def get_postags(self,):
        return " ".join([x[1] for x in self.repr])
   
    def add_pos_stats(self, df_, normilize=True):
        df_["adjective_count"] = df_.pos.apply(lambda x: len(re.findall(r"A[ $]|A=[\w]+", x)))
        df_["noun_count"] = df_.pos.apply(lambda x: x.count("S"))
        df_["verb_count"] = df_.pos.apply(lambda x: len(re.findall(r"V[$ ]", x)))
        if normilize:
            df_["adjective_count"] = df_["adjective_count"] / df_["token_len"]
            df_["noun_count"] = df_["noun_count"] / df_["token_len"]
            df_["verb_count"] = df_["verb_count"] / df_["token_len"]
        return df_

    def apply_df(self, df):
        df['pos'] = df.text.apply(lambda x: self.get_postags(x))
        return self.add_pos_stats(df)
   
    # dev = pos_count_add(dev)
    # sns.displot(data=dev, x="noun_count", hue="label", kind="kde")

class UDPipeTagger:

    def __init__(self, udpipe_model_path):
        self.parser = TextParser(udpipe_model_path)
        self.repr = []
        
    def init(self, texts):
        for t in texts:
            self.repr.append(self.parser.parse(t))
    
    def get_postags(self,):
        pos_tag_text = []
        for t in self.repr:
            pos_tags = ' '.join([ ' '.join([w.pos for w in sent])
                                for sent in t])
            pos_tag_text.append(pos_tags)

        #pos_tag_stats = [pd.DataFrame(x, index=[0]) for x in pos_tag_stats]
        #df_pos_tag_stats = pd.concat(pos_tag_stats,axis=0).fillna(0).reset_index(drop=True)
        #dev_ = pd.concat([df, df_pos_tag_stats], axis=1)
        return pos_tag_text

    def add_pos_stats(self, df_, normilize=True):
        df_["adjective_count"] = df_.pos.apply(lambda x: x.count("ADJ"))
        df_["noun_count"] = df_.pos.apply(lambda x: x.count("NOUN"))
        df_["verb_count"] = df_.pos.apply(lambda x: x.count("VERB"))
        if normilize:
            df_["adjective_count"] = df_["adjective_count"] / df_["token_len"]
            df_["noun_count"] = df_["noun_count"] / df_["token_len"]
            df_["verb_count"] = df_["verb_count"] / df_["token_len"]
        return df_

    def apply_df(self, df):
        df["pos"] = self.get_postags()
        return self.add_pos_stats(df)

        """ from matplotlib import pyplot as plt
        for tag in pos.columns:
        plt.xlim(0, dev_[tag].max())
        #plt.hist(dev_[dev_.label == 0][tag], density=True, histtype="barstacked")
        #plt.hist(dev_[dev_.label == 1][tag], density=True, histtype="barstacked")
        sns.histplot(data=dev_, x=tag, hue="label", kde=True)
        plt.show() """

class MystemTagger:

    def __init__(self) -> None:
        self.parser = Mystem()
        self.repr = []

    def init(self, texts: list):
        for t in texts:
            self.repr.append(self.parser.analyze(t))

    def get_adverbs(self,): 
        filtered_sent = []
        for analized_text in self.repr:
            for word in analized_text:
                if "analysis" not in word.keys():
                    continue
                if len(word["analysis"]) == 0:
                    continue
                word = word["analysis"][0]
                gr = word["gr"]
                if "PART=" in gr or len(re.findall(r"A[ $]|A=[\w]+", gr)) > 0 or "ADV" in gr:
                    filtered_sent.append(word["lex"])
                else:
                    continue
        return filtered_sent