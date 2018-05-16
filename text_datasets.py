from nltk.corpus import gutenberg, reuters
import numpy as np
import tensorflow as tf

##########################################################
# Collection of text datasets that use nltk and the tf.data
# APIs. Make sure to download the gutenberg and reuters datasets
# before using (nltk.download('gutenberg'), nltk.download('reuters'))
# python text_datasets.py will run an example

class NltkDataset():
    def __init__(self, nltk_module, timesteps, batch_size, train_size=0.8):
        self.timesteps=timesteps
        self.batch_size=batch_size
        self.fileids = nltk_module.fileids()
        chars = []
        for f in self.fileids:
            chars.extend(map(lambda x: ord(x), nltk_module.raw(f)))
        chars = np.array(chars)
        # Split into batches
        numb_of_timesteps = (chars.size // timesteps)
        size = numb_of_timesteps * timesteps
        chars = chars[:size]
        timesteps = np.array(np.split(chars, numb_of_timesteps)) # Splits into lists of size timesteps

        n_train = int(train_size * timesteps.shape[0])

        self.train_x = timesteps[:n_train - 1].T
        self.train_y = timesteps[1:n_train].T
        self.test_x = timesteps[n_train:-1].T
        self.test_y = timesteps[n_train + 1:].T

    def get_data(self):
        return (self.train_x, self.train_y), (self.test_x, self.test_y)

class Gutenberg(NltkDataset):
    def __init__(self, timesteps=100, batch_size=100):
        print("Making Gutenberg dataset...")
        NltkDataset.__init__(self, gutenberg, timesteps, batch_size)

class Reuters(NltkDataset):
    def __init__(self, timesteps=100, batch_size=100):
        print("Making Reuters dataset...")
        NltkDataset.__init__(self, reuters, timesteps, batch_size)


def get_input_fn(x, y):
    dset = tf.data.Dataset.from_tensor_slices(({'x': x}, y))
    return dset.shuffle(1000) \
               .repeat() \
               .batch(self.batch_size) \
               .map(lambda x, y: (tf.one_hot(x, 256), tf.one_hot(y, 256)))

if __name__ == "__main__":
    g = Gutenberg()
    r = Reuters()
    print(r.get_data()[0][0][:4])



