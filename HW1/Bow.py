import numpy as np

class Bow:
    def __init__(self):
        # Here is where we keep all words. A set for faster lookup
        self.all_words = set()
        # Two dictionaries to map word to index and back. This helps 'encoding' and 'decoding' a BoW
        self.word_to_idx= {}
        self.idx_to_word= {}
        # The total number of words is just kept to aid starting the numpy array size, but can be inferred from all_words set.
        self.total_words = 0

    def fit(self, data):
        """
        Fits the BoW using the data. This is used to help the BoW learn the vocabulary and word indexes.
        """
        print(len(data))
        for i in range(len(data)):
            for word in data[i]:
                self.all_words.add(word)
        for idx, word in enumerate(self.all_words):
            # Set the mapping indexes.
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        # Set the vocab size.
        self.total_words = len(self.all_words)
        transformed = np.empty((len(data), self.total_words))
        # Iterate over all sentences - this can be parallelized.
        print(self.word_to_idx)
        for row, sentence in enumerate(data):
            print(row)
            print(sentence)
            # Substitute each row by the sentence BoW.
            transformed_1 = np.zeros(self.total_words)
            for word in sentence:
                # Iterate over sentence words checking if they are in the vocabulary.
                if word in self.all_words:
                    word_idx = self.word_to_idx[word]
                    # Change the value of that specific index, by increasing the value.
                    transformed_1[word_idx] += 1
            transformed[row] = transformed_1
        return transformed



if __name__ == "__main__":
    sentences = [['this','is','a','list','of','sentences'], ['second', 'sentence', 'in' ,'list', 'of' ,'sentences'], ['a', 'word', 'for', 'complexity']]
    #sentences = ['this is a list of sentences', 'second sentence in list of sentences', 'a word for complexity']
    bow = Bow()
    x = bow.fit(sentences)

    print(x)