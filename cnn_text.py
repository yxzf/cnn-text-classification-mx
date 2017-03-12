import data_helpers
import numpy as np
import models
import mxnet as mx
import logging


def preprocess():
    x, y, vocab, vocab_inv = data_helpers.load_data()
    vocab_size = len(vocab)

    # randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # split train/dev set
    x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
    y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]
    print('Train/Dev split: %d/%d' % (len(y_train), len(y_dev)))
    print('train shape:', x_train.shape)
    print('dev shape:', x_dev.shape)
    print('vocab_size', vocab_size)


    sentence_size = x_train.shape[1]

    print('batch size', batch_size)
    print('sentence max words', sentence_size)
    print('embedding size', num_embed)

    train_iter = mx.io.NDArrayIter(
        data=x_train,
        label=y_train,
        batch_size=batch_size
    )
    dev_iter = mx.io.NDArrayIter(
        data=x_dev,
        label=y_dev,
        batch_size=batch_size
    )
    return vocab_size, sentence_size, train_iter, dev_iter

if __name__ == '__main__':
    batch_size = 500
    num_embed = 300
    epoch = 500
    vocab_size, sentence_size, train_iter, dev_iter = preprocess()
    logging.basicConfig(level=logging.DEBUG)
    model = models.cnn_for_sentence_classification(batch_size=batch_size, vocab_size=vocab_size, embed_size=num_embed,
                                                   num_label=2, with_embedding=False, sentence_size=sentence_size)
    model = mx.mod.Module(model)
    model.fit(train_iter, initializer=mx.init.Xavier(factor_type="in", magnitude=2.34), eval_data=dev_iter, optimizer='rmsprop', optimizer_params={'learning_rate': 0.0001},
              num_epoch=epoch)
