import numpy as np
import pandas as pd

from random import shuffle, choice
from scipy.stats import dirichlet
from scipy.special import logsumexp
from nltk.tree import Tree
from typing import List

# credits: https://gist.github.com/aaronstevenwhite/bb5452f232832400bd8c9a29e1bac0a9

class UnsupervisedPCFG(object):
    '''unsupervised pcfg induction with EM

    Parameters
    ----------
    n_nonterminals : int
        the number of nonterminals in the induced grammar
    smoothing : float
        the pseudocount to add to each probability
    '''
    
    def __init__(self, n_nonterminals=2, smoothing=1e-10):
        self._n_nonterminals = n_nonterminals
        self._smoothing = smoothing
        
    def _initialize(self, corpus, vocabulary):
        self._initialize_vocabulary(corpus, vocabulary)
        self._initialize_logprobs()
        self._initialize_counts()

    def _hash_sentence(self, sentence):
        return [self._vocab_hash[w] for w in sentence]
        
    def _initialize_vocabulary(self, corpus, vocabulary):
        self._corpus = [[w.lower() for w in sent] for sent in corpus]

        self._n_sents = len(self._corpus)
        self._n_words = [len(s) for s in self._corpus]

        self._vocabulary = vocabulary
        
        if self._vocabulary is None:
            self._vocabulary = np.unique([w
                                          for sent in self._corpus
                                          for w in sent])

        else:
            corpus_vocab = set([w
                                for sent in self._corpus
                                for w in sent])

            try:
                assert not corpus_vocab.difference(set(self._vocabulary))
            except AssertionError:
                msg = 'If a vocabulary is passed, it must be a superset'+\
                      'of the unique words in the corpus'
                raise ValueError(msg)
            
        self._vocab_hash = {w: i for i, w in enumerate(self._vocabulary)}
        self._n_vocab = self._vocabulary.shape[0]

        self._sents_hashed = [self._hash_sentence(s)
                              for s in self._corpus]

    def _initialize_logprobs(self):
        self._unary_logprobs = np.log(dirichlet.rvs(np.ones(self._n_vocab),
                                                    self._n_nonterminals)).T
        self._binary_logprobs = np.zeros([self._n_nonterminals]*3)-\
                                2*np.log(self._n_nonterminals)

    def _initialize_counts(self):
        self._unary_counts = np.zeros([self._n_vocab,
                                       self._n_nonterminals])
        self._binary_counts = np.zeros([self._n_nonterminals]*3)

        self._unary_counts_by_sent = np.zeros([self._n_sents,
                                               self._n_vocab,
                                               self._n_nonterminals])
        self._binary_counts_by_sent = np.zeros([self._n_sents]+\
                                               [self._n_nonterminals]*3)        
                
    def fit(self, X: List[List[str]], vocabulary=None, epochs=5, verbose=True):
        self._initialize(X, vocabulary)

        indices = list(range(self._n_sents))
        
        for e in range(epochs):
            shuffle(indices)

            logprob_trace = []
            
            for i, sid in enumerate(indices):
                beta, prob_symbs, prob_rules = self._compute_posteriors(sid)
                self._update_parameters(sid, prob_symbs, prob_rules)

                logprob_curr = beta[0,0,self._n_words[sid]]/self._n_words[sid]
                logprob_trace.append(logprob_curr)
                
                if verbose:
                    print('\n\nepoch', e, ', sentence', i, ', sentence id', sid)
                    print('normalized log-probability:\t', logprob_curr)
                    print('mean log-probability for epoch:\t', np.mean(logprob_trace))

    def _compute_posteriors(self, sid):
        sent_hashed = self._sents_hashed[sid]
        
        logbeta = self._compute_insides(sent_hashed)
        logalpha = self._compute_outsides(sent_hashed, logbeta)
        
        logmu = logalpha+logbeta

        posteriors_symbs = np.exp(logmu-logsumexp(logmu,
                                                  axis=0)[None,:])

        logpsi = logsumexp([logalpha[:,i,j][:,None,None]+\
                            logbeta[:,i,k][None,:,None]+\
                            logbeta[:,k,j][None,None,:]+\
                            self._binary_logprobs
                            for j in range(1,logalpha.shape[1]+1)
                            for i in range(j-1, -1, -1)
                            for k in range(i+1, j)], axis=0)
        
        posteriors_rules = np.exp(logpsi-logsumexp(logpsi,
                                                   axis=(1,2))[:,None,None])

        return logbeta, posteriors_symbs, posteriors_rules

    def _update_parameters(self, idx, prob_symbs, prob_rules):
        self._update_unary_counts(idx, prob_symbs)
        self._update_binary_counts(idx, prob_rules)
        self._update_logprobs()
             
    def _update_unary_counts(self, idx, prob_symbs):
        sent_hashed = self._sents_hashed[idx]
        n = self._n_words[idx]
        
        self._unary_counts -= self._unary_counts_by_sent[idx]
        self._unary_counts_by_sent[idx,sent_hashed] =  prob_symbs[:,
                                                                  np.arange(n),
                                                                  np.arange(1,n+1)].T
        self._unary_counts += self._unary_counts_by_sent[idx] + self._smoothing
        
    def _update_binary_counts(self, idx, prob_rules):
        self._binary_counts -= self._binary_counts_by_sent[idx]
        self._binary_counts_by_sent[idx] = prob_rules
        self._binary_counts += self._binary_counts_by_sent[idx] + self._smoothing        

    def _update_logprobs(self):
        self._unary_logprobs = np.log(self._unary_counts)-\
                               np.log(self._unary_counts.sum(axis=0)[None,:])
        self._binary_logprobs = np.log(self._binary_counts)-\
                                np.log(np.sum(self._binary_counts,
                                              axis=(1,2))[:,None,None])

    def _compute_insides(self, sent_hashed, mode='inference'):
        n = len(sent_hashed)

        logbeta = np.zeros([self._n_nonterminals, n, n+1])-np.inf
        logbeta[:,np.arange(n),np.arange(1,n+1)] = self._unary_logprobs[sent_hashed].T

        if mode == 'prediction':
            backpointer = np.zeros([self._n_nonterminals, n, n+1, 4])

        for j in range(1,n+1):
            for i in range(j-2, -1, -1):                    
                joint_logprobs = [logbeta[:,i,k][None,:,None]+\
                                  logbeta[:,k,j][None,None,:]+\
                                  self._binary_logprobs
                                  for k in range(i+1, j)]

                if mode == 'inference':
                    logbeta[:,i,j] = logsumexp(joint_logprobs, axis=(0,2,3))

                elif mode == 'prediction':
                    logbeta[:,i,j] = np.max(joint_logprobs, axis=(0,2,3))

                    location = np.where(joint_logprobs == logbeta[:,i,j][None,:,None,None])
                    location = [choice([idx
                                        for idx in zip(*location)
                                        if idx[1] == nt])
                                for nt in range(self._n_nonterminals)]
                    backpointer[:,i,j] = np.array(location)
                    backpointer[:,i,j,0] += i + 1

        if mode == 'inference':
            return logbeta
        else:
            return backpointer
        
    def _compute_outsides(self, sent_hashed, logbeta):
        n = len(sent_hashed)
        logalpha = np.zeros([self._n_nonterminals, n, n+1])-np.inf

        for j in range(n, 0, -1):
            for i in range(0, j):
                if (i,j) == (0,n):
                    logalpha[:,i,j] = np.array([0.]+\
                                               [-np.inf]*\
                                               (self._n_nonterminals-1))
                    continue

                joint_logprobs_l = [logalpha[:,i,l][:,None,None]+\
                                    logbeta[:,j,l][None,None,:]+\
                                    self._binary_logprobs
                                    for l in range(j+1,n+1)]
                joint_logprobs_r = [logalpha[:,l,j][:,None,None]+\
                                    logbeta[:,l,i][None,:,None]+\
                                    self._binary_logprobs
                                    for l in range(i)]

                if joint_logprobs_l and joint_logprobs_r:
                    marginal_logprobs_l = logsumexp(joint_logprobs_l,
                                                    axis=(0,1,3))
                    marginal_logprobs_r = logsumexp(joint_logprobs_r,
                                                    axis=(0,1,2))

                    logalpha[:,i,j] = np.logaddexp(marginal_logprobs_l,
                                                   marginal_logprobs_r)

                elif joint_logprobs_l:
                    logalpha[:,i,j] = logsumexp(joint_logprobs_l,
                                                axis=(0,1,3))

                elif joint_logprobs_r:
                    logalpha[:,i,j] = logsumexp(joint_logprobs_r,
                                                axis=(0,1,2))

        return logalpha

    def predict(self, X):
        parses = []
        
        for sent in X:
            sent_hashed = [self._vocab_hash[w] for w in sent]
            backpointer = self._compute_insides(sent_hashed, mode='prediction')
            parse = self._construct_parse(sent, backpointer)
            parses.append(parse)

        return parses

    def _construct_parse(self, sent, backpointer, lhs=0, idx=None):
        idx = (0, len(sent)) if idx is None else idx
        
        k, _, rhs1, rhs2 = backpointer[lhs,idx[0],idx[1]]
        
        if idx[0] < k-1:
            subtree1 = self._construct_parse(sent, backpointer, int(rhs1), (idx[0],int(k)))
        else:
            subtree1 = Tree('N'+str(int(rhs1)), [sent[idx[0]]])

        if idx[1] > k+1:
            subtree2 = self._construct_parse(sent, backpointer, int(rhs2), (int(k),idx[1]))
        else:
            subtree2 = Tree('N'+str(int(rhs2)), [sent[int(k)]])
        
        return Tree('N'+str(int(lhs)), [subtree1, subtree2])
    
    def score(self, X, length_normalize=True):
        total_score = 0.
        all_scores = []
        
        for sent in X:
            sent_hashed = self._hash_sentence(sent)
            logbeta = self._compute_insides(sent_hashed)

            score = logbeta[0,0,len(sent)]
            
            if length_normalize:
                score /= len(sent)

            total_score += score
            all_scores.append(score)

        mean_score = total_score/len(X)
                
        return total_score, mean_score, all_scores

    @property
    def unary_probabilities(self):
        nonterminals = ['N'+str(i) for i in range(self._n_nonterminals)]
        
        return pd.DataFrame(np.exp(self._unary_logprobs),
                            index=self._vocabulary,
                            columns=nonterminals)

    @property
    def binary_probabilities(self):
        nonterminals = ['N'+str(i) for i in range(self._n_nonterminals)]
        
        return pd.Panel(np.exp(self._binary_logprobs),
                        items=nonterminals,
                        major_axis=nonterminals,
                        minor_axis=nonterminals)

    
if __name__ == "__main__":
    len_min = 2
    len_max = 11
    
    from nltk.downloader import Downloader
    from nltk.corpus.reader import CHILDESCorpusReader

    nltk_data_dir = Downloader().default_download_dir()
    brown_adam_dir = os.path.join(nltk_data_dir, '/corpora/CHILDES/Brown/Adam/')
    
    print('Loading Adam corpus from Brown...')
    
    childes = CHILDESCorpusReader(brown_adam_dir, '.+xml')
    
    #X = [[w[1] for w in s] for s in treebank.tagged_sents() if len(s) > 2]
    corpus = [[w.lower() for w in s if '*' not in w]
              for s in childes.sents(speaker='MOT')]
    corpus = [s for s in corpus if len(s) > len_min and len(s) < len_max]

    print('Loading complete. '+str(len(corpus))+' sentences loaded.')
    
    m = UnsupervisedPCFG(2)
    m.fit(corpus[:10000])

