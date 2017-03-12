#!/usr/bin/env python
#encoding=utf-8
import mxnet as mx
import numpy as np

def cnn_for_sentence_classification(batch_size, vocab_size, embed_size, num_label, with_embedding,
                                    sentence_size, dropout=0.5, filter_list = [2, 3, 4], num_filter=100):
    input_x = mx.sym.Variable(name='data')
    input_y = mx.sym.Variable(name='softmax_label')
    # mxnet 中conv的输入格式是四维(batch_size, channel_size, height, width) ()

    if not with_embedding:
        embed_layer = mx.sym.Embedding(data=input_x, input_dim=vocab_size, output_dim=embed_size)
        conv_input = mx.sym.Reshape(data=embed_layer, target_shape=[batch_size, 1, sentence_size, embed_size])
    else:
        conv_input = input_x

    conv_list = []
    for i, filter_size in enumerate(filter_list):
        conv = mx.sym.Convolution(conv_input, kernel=(filter_size, embed_size),
                                  stride=(1, embed_size), num_filter=num_filter)
        relu = mx.sym.Activation(conv, act_type='relu')
        pool = mx.sym.Pooling(relu, pool_type='max', kernel=(sentence_size-filter_size+1, 1), stride=(1, 1))
        conv_list.append(pool)
    filter_total = num_filter * len(filter_list)
    concat = mx.sym.Concat(*conv_list, dim=1)
    concat_reshape = mx.sym.Reshape(concat, target_shape=(batch_size, filter_total))
    concat_dropout = mx.sym.Dropout(concat_reshape, p=dropout)
    fc1 = mx.sym.FullyConnected(concat_dropout, num_hidden=100)
    relu1 = mx.sym.Activation(fc1, act_type='relu')
    dropout1 = mx.sym.Dropout(relu1, p=dropout)
    fc2 = mx.sym.FullyConnected(dropout1, num_hidden=num_label)
    sm = mx.sym.SoftmaxOutput(data=fc2, label=input_y)
    return sm






