from collections import defaultdict,Counter
import re
class BPE:
    def split(self,line):
        tokens = list()
        for word in line.split():
            tokens.append(" ".join(word) + " </w>")
        words_N = Counter(tokens)
        return words_N

    def get_N_pairs(self,vocab):
        pairs = defaultdict(int)
        for words, number in vocab.items():
            word = words.split()
            for i in range(len(word) - 1):
                pairs[word[i], word[i + 1]] += number
        return pairs

    def merge_vocab(self,pair, words):
        output = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in words:
            w_out = p.sub(''.join(pair), word)
            output[w_out] = words[word]
        return output

    def use(self,sentence):
        words = self.split(sentence)

        num_merges = 100
        for i in range(num_merges):
            pairs = self.get_N_pairs(words)
            if not pairs:
                break
            for key, value in pairs.items():
                if value == max(pairs.values()):
                    best = key
            words = self.merge_vocab(best, words)

        tokens = list()
        for key in words.keys():
            tokens.append(key)
        return tokens



