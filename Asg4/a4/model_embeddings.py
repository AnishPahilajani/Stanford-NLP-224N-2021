#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2020-21: Homework 4
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Vera Lin <veralin@stanford.edu>
"""

import torch.nn as nn

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layers.

        @param embed_size (int): Embedding size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size

        # default values
        self.source = None
        self.target = None

        src_pad_token_idx = vocab.src['<pad>']
        tgt_pad_token_idx = vocab.tgt['<pad>']

        ### YOUR CODE HERE (~2 Lines)
        ### TODO - Initialize the following variables:
        ###     self.source (Embedding Layer for source language)
        ###     self.target (Embedding Layer for target langauge)
        ###
        ### Note:
        ###     1. `vocab` object contains two vocabularies:
        ###            `vocab.src` for source
        ###            `vocab.tgt` for target
        ###     2. You can get the length of a specific vocabulary by running:
        ###             `len(vocab.<specific_vocabulary>)`
        ###     3. Remember to include the padding token for the specific vocabulary
        ###        when creating your Embedding.
        ###
        ### Use the following docs to properly initialize these variables:
        ###     Embedding Layer:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding
        
        # embedding = nn.Embedding(num_embeddings=10, embedding_dim=3)
        # then it means that you have 10 words and represent each of those words by an embedding of size 3, for example.
        # hello -> [0.01 0.2 0.5]
        # world -> [0.04 0.6 0.7]
        # list(embedding.parameters())
        # [Parameter containing:
        #  tensor([[ 0.9227,  0.6492, -1.1440],
        #          [ 1.5318, -0.2873, -0.7290],
        #          [-0.4234, -1.7012, -0.9684],
        #          [-0.2859,  1.4677, -1.4499],
        #          [-1.8966, -1.4591,  0.5218],
        #          [ 2.4023, -1.5395, -0.7947],
        #          [-0.0464,  0.7174, -0.7452],
        #          [ 0.9500, -0.4633,  0.5398],
        #          [ 0.3458, -0.7997,  0.8895],
        #          [-0.3303, -0.5663, -0.2300]], requires_grad=True)]
        self.source = nn.Embedding(num_embeddings = len(vocab.src), embedding_dim = self.embed_size, padding_idx = src_pad_token_idx)
        self.target = nn.Embedding(num_embeddings = len(vocab.tgt), embedding_dim = self.embed_size, padding_idx = tgt_pad_token_idx)

        ### END YOUR CODE


