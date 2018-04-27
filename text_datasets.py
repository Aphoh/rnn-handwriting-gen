from nltk.corpus import gutenberg, reuters
import numpy as np
import tensorflow as tf

##########################################################
# Collection of text datasets that use nltk and the tf.data
# APIs. Make sure to download the gutenberg and reuters datasets
# before using (nltk.download('gutenberg'), nltk.download('reuters'))
# Example:
#   sess = tf.Session()
#   g = Gutenberg()
#   print(g.tf_iterator().get_next().eval(session=sess))
#   r = Reuters()
#   print(r.tf_iterator().get_next().eval(session=sess))
#
# python text_datasets.py will run this example.

class NltkDataset():
    def __init__(self, nltk_module, sentence_size, batch_size):
        print("Making Gutenberg dataset...")
        self.sentence_size=sentence_size
        self.batch_size=batch_size
        self.fileids = nltk_module.fileids()
        chars = []
        for f in self.fileids:
            chars.extend(map(lambda x: ord(x), nltk_module.raw(f)))
        chars = np.array(chars)
        # Split into batches
        n = chars.size // sentence_size
        chars = chars[:n * sentence_size]
        self.sentences = np.array(np.split(chars, n)) # Splits into lists of size sentence_size
        self.batches = self.sentences.T
        self.dset = tf.data.Dataset.from_tensor_slices(self.batches)

    def tf_iterator(self):
        return self.dset.batch(self.batch_size).make_one_shot_iterator()

class Gutenberg(NltkDataset):
    def __init__(self, sentence_size=100, batch_size=100):
        NltkDataset.__init__(self, gutenberg, sentence_size, batch_size)

class Reuters():
    def __init__(self, sentence_size=100, batch_size=100):
        NltkDataset.__init__(self, reuters, sentence_size, batch_size)


if __name__ == "__main__":
    sess = tf.Session()
    g = Gutenberg()
    print(g.tf_iterator().get_next().eval(session=sess))
    r = Reuters()
    print(r.tf_iterator().get_next().eval(session=sess))



