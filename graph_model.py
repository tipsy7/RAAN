
# -------------------------------------------------------------------------------------
"""Visual Graph and Textual Graph"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
import pdb


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def inter_relations(K, Q, xlambda):
    """
    Q: (batch, queryL, d)
    K: (batch, sourceL, d)
    return (batch, queryL, sourceL)
    """
    batch_size, queryL = Q.size(0), Q.size(1)
    batch_size, sourceL = K.size(0), K.size(1)


    # --> (batch, sourceL, queryL)
    queryT = torch.transpose(Q, 1, 2)

    attn = torch.bmm(K, queryT)
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn * xlambda)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)

    return attn

def inter_relations_rare(K,  xlambda):
    """
    Q: (batch, queryL, d)
    K: (batch, sourceL, d)
    return (batch, queryL, sourceL)
    """

    batch_size, sourceL,queryL = K.size(0), K.size(1), K.size(2)


    attn = nn.LeakyReLU(0.1)(K)
    attn = l2norm(attn, 2)


    attn = torch.transpose(attn, 1, 2).contiguous()

    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn * xlambda)

    attn = attn.view(batch_size, queryL, sourceL)

    return attn


def intra_relations(K, Q, xlambda):
    """
    Q: (n_context, sourceL, d)
    K: (n_context, sourceL, d)
    return (n_context, sourceL, sourceL)
    """
    batch_size, KL = K.size(0), K.size(1)
    K = torch.transpose(K, 1, 2).contiguous()
    attn = torch.bmm(Q, K)

    attn = attn.view(batch_size * KL, KL)

    attn = nn.Softmax(dim=1)(attn * xlambda)
    attn = attn.view(batch_size, KL, KL)
    return attn


class VisualGraph(nn.Module):

    def __init__(self,
                 feat_dim,
                 hid_dim,
                 out_dim,
                 dropout,
                 n_kernels=8):
        '''
        ## Variables:
        - feat_dim: dimensionality of input image features
        - out_dim: dimensionality of the output
        - dropout: dropout probability
        - n_kernels : number of Gaussian kernels for convolutions
        - bias: whether to add a bias to Gaussian kernels
        '''

        super(VisualGraph, self).__init__()

        # Set parameters
        self.feat_dim = feat_dim
        self.out_dim = out_dim


        self.out_1 = nn.utils.weight_norm(nn.Linear(feat_dim, 8))
        self.out_2 = nn.utils.weight_norm(nn.Linear(8, out_dim))



        self.out_9 = nn.utils.weight_norm(nn.Linear(1024, 16))
        self.out_10 = nn.utils.weight_norm(nn.Linear(16, 1))



    def node_level_matching(self, tnodes, vnodes, n_block,xlambda,text_word_frequency_i_expand_new):
        text_word_frequency_i_expand_new = nn.Softmax(dim=1)( 0.02*text_word_frequency_i_expand_new)
        tnodes = tnodes + 0.02 * tnodes * text_word_frequency_i_expand_new


        inter_relation = inter_relations(tnodes, vnodes, xlambda)


        attnT = torch.transpose(inter_relation, 1, 2)




        contextT = torch.transpose(tnodes, 1, 2)
        weightedContext = torch.bmm(contextT, attnT)
        weightedContextT = torch.transpose(
            weightedContext, 1, 2)

        # Multi-block similarity

        qry_set = torch.split(vnodes, n_block, dim=2)
        ctx_set = torch.split(weightedContextT, n_block, dim=2)

        qry_set = torch.stack(qry_set, dim=2)
        ctx_set = torch.stack(ctx_set, dim=2)


        vnode_mvector = cosine_similarity(
            qry_set, ctx_set, dim=-1)

        return vnode_mvector,weightedContextT


    def structure_level_matching(self, vnode_mvector):



        sim = self.out_2(self.out_1(vnode_mvector).tanh())


        return sim




    def few_detect(self,  text):

        rare_vector = self.out_9(text)
        text_few = self.out_10(rare_vector.tanh())

        return text_few,rare_vector


    def forward(self, images, captions, bbox, cap_lens, word_freqs, opt):
        similarities = []
        n_block = opt.embed_size // opt.num_block
        n_image, n_caption = images.size(0), captions.size(0)

        text_word_frequency  = word_freqs
        text_few,rare_vector = self.few_detect(captions)
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()


            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            text_word_frequency_i = text_word_frequency[i, :n_word,:].unsqueeze(0).contiguous()

            text_word_frequency_i_expand = text_word_frequency_i.repeat(n_image, 1, 1).contiguous()


            vnode_mvector,weightedContextT_ori = self.node_level_matching(
                cap_i_expand, images, n_block,opt.lambda_softmax,text_word_frequency_i_expand)


            v2t_similarity = self.structure_level_matching(
                vnode_mvector)


            v2t_similarity1 = v2t_similarity.view(n_image, -1).mean(dim=1, keepdim=True)
            similarities.append(v2t_similarity1)

        similarities = torch.cat(similarities, 1)
        return similarities,text_few,rare_vector

    def _compute_pseudo(self, bb_centre):
        '''

        Computes pseudo-coordinates from bounding box centre coordinates

        ## Inputs:
        - bb_centre (batch_size, K, coord_dim)
        - polar (bool: polar or euclidean coordinates)
        ## Returns:
        - pseudo_coord (batch_size, K, K, coord_dim)
        '''
        K = bb_centre.size(1)

        # Compute cartesian coordinates (batch_size, K, K, 2)
        pseudo_coord = bb_centre.view(-1, K, 1, 2) - \
            bb_centre.view(-1, 1, K, 2)

        # Conver to polar coordinates
        rho = torch.sqrt(
            pseudo_coord[:, :, :, 0]**2 + pseudo_coord[:, :, :, 1]**2)
        theta = torch.atan2(
            pseudo_coord[:, :, :, 0], pseudo_coord[:, :, :, 1])
        pseudo_coord = torch.cat(
            (torch.unsqueeze(rho, -1), torch.unsqueeze(theta, -1)), dim=-1)

        return pseudo_coord


class TextualGraph(nn.Module):

    def __init__(self,
                 feat_dim,
                 hid_dim,
                 out_dim,
                 dropout,
                 n_kernels=8):
        '''
        ## Variables:
        - feat_dim: dimensionality of input image features
        - out_dim: dimensionality of the output
        - dropout: dropout probability
        - n_kernels : number of Gaussian kernels for convolutions
        - bias: whether to add a bias to Gaussian kernels
        '''

        super(TextualGraph, self).__init__()

        # Set parameters
        self.feat_dim = feat_dim
        self.out_dim = out_dim



        self.out_1 = nn.utils.weight_norm(nn.Linear(feat_dim, 8))
        self.out_2 = nn.utils.weight_norm(nn.Linear(8, out_dim))



    def build_sparse_graph(self, dep, lens):
        adj = np.zeros((lens, lens), dtype=np.int)
        for i, pair in enumerate(dep):
            if i == 0 or pair[0] >= lens or pair[1] >= lens:
                continue
            adj[pair[0], pair[1]] = 1
            adj[pair[1], pair[0]] = 1
        adj = adj + np.eye(lens)
        return torch.from_numpy(adj).cuda().float()

    def node_level_matching(self, vnodes, tnodes, n_block, xlambda):


        inter_relation = inter_relations(vnodes, tnodes, xlambda)


        attnT = torch.transpose(inter_relation, 1, 2)
        contextT = torch.transpose(vnodes, 1, 2)
        weightedContext = torch.bmm(contextT, attnT)
        weightedContextT = torch.transpose(
            weightedContext, 1, 2)

        qry_set = torch.split(tnodes, n_block, dim=2)
        ctx_set = torch.split(weightedContextT, n_block, dim=2)

        qry_set = torch.stack(qry_set, dim=2)
        ctx_set = torch.stack(ctx_set, dim=2)

        tnode_mvector = cosine_similarity(
            qry_set, ctx_set, dim=-1)
        return tnode_mvector,weightedContextT

    def structure_level_matching(self, tnode_mvector, intra_relation, depends, rare_vector,opt):

        sim = self.out_2(self.out_1(tnode_mvector + 0.4 * rare_vector).tanh())

        return sim



    def forward(self, images, captions, depends, cap_lens,rare_vector,opt):
        n_image = images.size(0)
        n_caption = captions.size(0)
        similarities = []
        n_block = opt.embed_size // opt.num_block
        words_sim = []

        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()

            cap_i_expand = cap_i.repeat(n_image, 1, 1).contiguous()

            rare_vector_i = rare_vector[i, :n_word, :].unsqueeze(0).contiguous()

            rare_vector_i= rare_vector_i.repeat(n_image, 1, 1).contiguous()


            nodes_sim,weightedContextT = self.node_level_matching(
                images, cap_i_expand, n_block,opt.lambda_softmax)

            phrase_sim = self.structure_level_matching(
                nodes_sim, words_sim, depends[i],rare_vector_i, opt)

            phrase_sim = phrase_sim.view(n_image, -1).mean(dim=1, keepdim=True)

            similarities.append(phrase_sim)


        similarities = torch.cat(similarities, 1)

        return similarities
