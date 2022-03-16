from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import reduce
import time

import torch
from torch.nn.modules.dropout import Dropout
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .AttModel import pack_wrapper
from .CaptionModel import CaptionModel
from .XRelation import PositionRelation, SemanticRelation, BoxRelationalEmbedding
from . import utils

TIMES = 0

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the', 'white', 'red',' blue']


class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = torch.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size
        # print(torch.any(torch.isnan(weight)))
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).to(weight)
            weight = weight / (weight.sum(1, keepdim=True)+1e-8) # normalize to 1
        # print(torch.any(torch.isnan(weight)))
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1)) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res, weight


class UpDownHierarchyAttentionCore(nn.Module):
    def __init__(self, opt):
        super(UpDownHierarchyAttentionCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.hierarchy_attention = opt.hierarchy_attention

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v

        self.attention_origin_att_feats = Attention(opt)
        self.attention_position_relation_att_feats = Attention(opt)
        self.attention_semantic_relation_att_feats = Attention(opt)

        # layernorm
        self.layer_norm1 = nn.LayerNorm(opt.rnn_size,  eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(opt.rnn_size,  eps=1e-6)

        #
        self.att23_mlp = nn.Linear(opt.rnn_size*2, opt.rnn_size, bias=False)

        self.gate1 = nn.Sequential(
            nn.Linear(opt.rnn_size + opt.input_encoding_size, opt.rnn_size),
            nn.Sigmoid(),
        ) 
        self.gate2 = nn.Sequential(
            nn.Linear(opt.rnn_size + opt.input_encoding_size, opt.rnn_size),
            nn.Sigmoid(),
        ) 
        self.fusion_layer_norm = nn.LayerNorm(opt.rnn_size,  eps=1e-6)


    def forward(
        self, xt, fc_feats, att_feats, p_att_feats, state, att_masks,
        position_relation_att_feats, p_position_relation_att_feats, position_att_masks,
        semantic_relation_att_feats, p_semantic_relation_att_feats, semantic_att_masks,
    ):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))


        # region
        if self.hierarchy_attention == 1:
            assert False

        else:
            att1,weight1 = self.attention_origin_att_feats(h_att, att_feats, p_att_feats, att_masks)
            att1_index = torch.max(weight1,dim=1)[1].long() 
            bbbatch = torch.arange(0,att_feats.size(0))
            att2,weight2 = self.attention_position_relation_att_feats(h_att, position_relation_att_feats[bbbatch, att1_index], 
                p_position_relation_att_feats[bbbatch, att1_index],position_att_masks[bbbatch, att1_index]
            )
            att3,weight3 = self.attention_semantic_relation_att_feats(h_att, semantic_relation_att_feats[bbbatch, att1_index], 
                p_semantic_relation_att_feats[bbbatch, att1_index],semantic_att_masks[bbbatch, att1_index]
            )
            gate1 = self.gate1(torch.cat([xt, h_att],dim=1))
            att23 = torch.cat([self.layer_norm1(gate1 * att2), self.layer_norm2((1-gate1) * att3)], dim= 1)
            att23 = self.att23_mlp(att23)

            att = self.fusion_layer_norm(att23 + att1)

            gate2 = self.gate2(torch.cat([xt, h_att],dim=1))
            att = gate2 * att1 + (1-gate2) * att23

        lang_lstm_input = torch.cat([att, h_att], 1)

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state



class DualRelationModelX2(CaptionModel):
    def __init__(self, opt):
        super(DualRelationModelX2, self).__init__()
        self.core = UpDownHierarchyAttentionCore(opt)

        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = getattr(opt, 'max_length', 175) or opt.seq_length # maximum sample length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.bboxes_embed_size = opt.bboxes_embed_size
        self.class_embed_size = opt.class_embed_size
        self.semantic_relation_size = opt.semantic_relation_size 
        self.semantic_relation_layers = opt.semantic_relation_layers 
        self.use_enhanced_feats = opt.use_enhanced_feats

        self.drop_prob_position = opt.drop_prob_position
        self.drop_prob_semantic = opt.drop_prob_semantic

        self.w2v_path = opt.w2v_path
        self.w2v = self.load_w2v(opt.w2v_path, opt.device)


        self.bos_idx = getattr(opt, 'bos_idx', 0)
        self.eos_idx = getattr(opt, 'eos_idx', 0)
        self.pad_idx = getattr(opt, 'pad_idx', 0)

        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0 # Schedule sampling probability

        # ===================== add ==========
        self.att_feats_embed_early = nn.Sequential(
            nn.Linear(self.att_feat_size, self.rnn_size),
            nn.LeakyReLU()
        )

        # position
        self.bboxes_embed = nn.Sequential(
            nn.Linear(4, self.bboxes_embed_size),
            nn.GELU()
        )
        self.position_encode = PositionRelation(
            self.rnn_size, self.bboxes_embed_size, self.rnn_size, self.rnn_size, dropout_prob = self.drop_prob_position
        )

        # semantic
        self.semantic_encode = SemanticRelation(
            self.rnn_size, self.class_embed_size, self.rnn_size, self.rnn_size, self.semantic_relation_size, 
            self.semantic_relation_layers, dropout_prob = self.drop_prob_semantic
        )
        self.class_embed_mlp = nn.Sequential(
            nn.Linear(self.class_embed_size*2, self.class_embed_size),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_prob_semantic)
        )
        if self.use_enhanced_feats:
            assert False
            self.transform_gate = nn.Sequential(
                nn.Linear(self.att_feat_size, self.att_feat_size),
                nn.Sigmoid()
            )
            self.fusion_enhanced_feats = nn.Sequential(
                nn.Dropout(self.drop_prob_semantic),
                nn.Linear(self.rnn_size, self.att_feat_size)
            )
        
        self.position_relation_att_embed = nn.Sequential(
            nn.Linear(self.rnn_size, self.rnn_size),
            nn.ReLU(),
            nn.Dropout(self.drop_prob_lm)
        )
        self.position_relation_ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.semantic_relation_att_embed = nn.Sequential(
            nn.Linear(self.rnn_size, self.rnn_size),
            nn.ReLU(),
            nn.Dropout(self.drop_prob_lm)
        )
        self.semantic_relation_ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)
        # ==================== end add ===========


        self.embed = nn.Sequential(nn.Embedding(self.vocab_size, self.input_encoding_size),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))

        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(*(reduce(lambda x,y:x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size)]))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

        # For remove bad endding
        self.vocab = opt.vocab
        self.bad_endings_ix = [int(k) for k,v in self.vocab.items() if v in bad_endings]

    def load_w2v(self, w2v_path, device):
        w2v = torch.load(w2v_path,map_location='cpu').to(device) 
        return w2v

    def _position_relation_encoding(self, att_feats, att_masks, bboxes, rel_iou_threshold=0):
        batch_size = att_feats.size(0)
        device = att_feats.device
        gt_batch_boxes_num = torch.sum(att_masks.data.long(),dim=1, keepdim=True).squeeze(1) 

        bboxes_embedding = BoxRelationalEmbedding(bboxes, dim_g=64, wave_len=1000, trignometric_embedding= False) 
        bboxes_embedding = self.bboxes_embed(bboxes_embedding) 

        postion_feats = att_feats.new_zeros(batch_size, att_feats.size(1), att_feats.size(1), att_feats.size(2))
        masks = att_feats.new_zeros(batch_size, att_feats.size(1), att_feats.size(1), dtype=torch.long)
        for b in range(batch_size):
            box_num = gt_batch_boxes_num[b]
            ious = torchvision.ops.box_iou(bboxes[b][:box_num], bboxes[b][:box_num])

            mask = (1 - torch.eye(ious.size(0))).to(device) 
            mask = ((ious >= rel_iou_threshold)*mask).long()
            masks[b,:box_num, :box_num] = mask

            batch_feats = att_feats[b, :box_num]
            postion_feat = self.position_encode(batch_feats, bboxes_embedding[b, :box_num, :box_num]) 
            postion_feats[b, :box_num, :box_num] = postion_feat

        return postion_feats, masks

    def _semantic_relation_encoding(self,  att_feats, att_masks, classes, w2v):
        batch_size = att_feats.size(0)
        device = att_feats.device
        gt_batch_classes_num = torch.sum(att_masks.data.long(),dim=1, keepdim=True).squeeze(1) 

        classes_embedding = w2v[classes] 
        pairs = torch.ones([att_feats.size(0), att_feats.size(1),att_feats.size(1)]) 
        pairs = torch.nonzero(pairs, as_tuple =False).long()
        classes_embedding_ = classes_embedding.new_zeros([batch_size, att_feats.size(1)**2, self.class_embed_size*2]) 
        for b in range(batch_size):
            xxxx = att_feats.size(1)**2
            temp = classes_embedding[b][pairs[b*xxxx:(b+1)*xxxx, 1:]]
            temp = torch.cat([temp[:,0],temp[:,1]],1) 
            classes_embedding_[b] = temp
        classes_embedding = self.class_embed_mlp(classes_embedding_).reshape([batch_size, att_feats.size(1), att_feats.size(1), -1])

        semantic_relation_att_feats_G = att_feats.new_zeros(batch_size, att_feats.size(1), att_feats.size(1), att_feats.size(2)) 
        semantic_relation_att_feats_enhanced = att_feats.new_zeros(batch_size, att_feats.size(1), att_feats.size(2))
        relation_cls = att_feats.new_zeros(batch_size, att_feats.size(1), att_feats.size(1), self.semantic_relation_size)
        masks = att_feats.new_zeros(batch_size, att_feats.size(1), att_feats.size(1), dtype=torch.long)
        for b in range(batch_size):
            class_num = gt_batch_classes_num[b]

            mask = (1 - torch.eye(class_num)).to(device) 
            masks[b,:class_num, :class_num] = mask.long()

            batch_feats = att_feats[b, :class_num]
            semantic_feat, enhanced_feats, relation_cls_ = self.semantic_encode(batch_feats, classes_embedding[b, :class_num, :class_num]) 
            semantic_relation_att_feats_G[b, :class_num, :class_num] = semantic_feat
            semantic_relation_att_feats_enhanced[b, :class_num] = enhanced_feats
            relation_cls[b, :class_num, :class_num] = relation_cls_


        return semantic_relation_att_feats_G, semantic_relation_att_feats_enhanced, relation_cls, masks

    def _forward(self, fc_feats, att_feats, seq, att_masks=None, *args, **kwargs):
        batch_size = fc_feats.size(0)
        device = att_feats.device
        if seq.ndim == 3:  
            seq = seq.reshape(-1, seq.shape[2])
        seq_per_img = seq.shape[0] // batch_size
        state = self.init_hidden(batch_size*seq_per_img)

        outputs = fc_feats.new_zeros(batch_size*seq_per_img, seq.size(1), self.vocab_size)

        assert att_masks != None,"mask needed"

        relation_att_feats = self.att_feats_embed_early(att_feats)
        
        # posotion
        bboxes = kwargs.get("bboxes", None)
        rel_iou_threshold = kwargs.get("rel_iou_threshold", 0)
        assert bboxes!=None, "boxes needed"
        position_relation_att_feats, position_feats_masks \
            = self._position_relation_encoding(relation_att_feats, att_masks, bboxes, rel_iou_threshold)

        # semantic
        bboxes_class = kwargs.get("bboxes_class", None)
        assert bboxes_class!=None,"object class needed"
        semantic_relation_att_feats, semantic_relation_att_feats_enhanced, relation_cls, semantic_feats_masks \
            = self._semantic_relation_encoding(relation_att_feats, att_masks, bboxes_class, self.w2v)
        self.relation_cls_out = relation_cls 
        self.relation_cls_mask = semantic_feats_masks 
        # semantic fusion
        if self.use_enhanced_feats:
            gate = self.transform_gate(att_feats) 
            semantic_relation_att_feats_enhanced= self.fusion_enhanced_feats(semantic_relation_att_feats_enhanced) 
            att_feats = gate * att_feats + (1 - gate) * semantic_relation_att_feats_enhanced

        # Prepare the features
        p_position_relation_att_feats, pp_position_relation_att_feats, p_position_att_masks = self._prepare_positon_realtion_feature(position_relation_att_feats, position_feats_masks, att_masks)
        p_semantic_relation_att_feats, pp_semantic_relation_att_feats, p_semantic_att_masks = self._prepare_semantic_realtion_feature(semantic_relation_att_feats, semantic_feats_masks, att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost
        # ===================== end ======================


        # Prepare the features
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost

        if seq_per_img > 1:
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = \
                utils.repeat_tensors(seq_per_img,
                [p_fc_feats, p_att_feats, pp_att_feats, p_att_masks]
            )
            p_position_relation_att_feats, pp_position_relation_att_feats, p_position_att_masks = \
                utils.repeat_tensors(seq_per_img,
                [p_position_relation_att_feats, pp_position_relation_att_feats, p_position_att_masks]
            )
            p_semantic_relation_att_feats, pp_semantic_relation_att_feats, p_semantic_att_masks = \
                utils.repeat_tensors(seq_per_img,
                [p_semantic_relation_att_feats, pp_semantic_relation_att_feats, p_semantic_att_masks]
            )

        for i in range(seq.size(1)):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size*seq_per_img).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(
                it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks,
                p_position_relation_att_feats, pp_position_relation_att_feats,p_position_att_masks,
                p_semantic_relation_att_feats, pp_semantic_relation_att_feats,p_semantic_att_masks,
                state
            )
            outputs[:, i] = output

        return outputs

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        assert False
    
    def _diverse_sample(self, fc_feats, att_feats, att_masks=None, opt={}):
        assert False

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}, on_info=None, *args, **kwargs):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        sample_n = int(opt.get('sample_n', 1))
        group_size = opt.get('group_size', 1)
        output_logsoftmax = opt.get('output_logsoftmax', 1)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)
        if group_size > 1:
            return self._diverse_sample(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size*sample_n)

        assert att_masks != None,"mask needed"

        relation_att_feats = self.att_feats_embed_early(att_feats)
        
        # posotion
        bboxes = kwargs.get("bboxes", None)
        rel_iou_threshold = kwargs.get("rel_iou_threshold", 0)
        assert bboxes!=None, "boxes needed"
        position_relation_att_feats, position_feats_masks \
            = self._position_relation_encoding(relation_att_feats, att_masks, bboxes, rel_iou_threshold)

        # semantic
        bboxes_class = kwargs.get("bboxes_class", None)
        assert bboxes_class!=None,"object class needed"
        semantic_relation_att_feats, semantic_relation_att_feats_enhanced, relation_cls, semantic_feats_masks \
            = self._semantic_relation_encoding(relation_att_feats, att_masks, bboxes_class, self.w2v)
        self.relation_cls_out = relation_cls 
        self.relation_cls_mask = semantic_feats_masks 

        if self.use_enhanced_feats:
            gate = self.transform_gate(att_feats)
            semantic_relation_att_feats_enhanced= self.fusion_enhanced_feats(semantic_relation_att_feats_enhanced)
            att_feats = gate * att_feats + (1 - gate) * semantic_relation_att_feats_enhanced

        # Prepare the features
        p_position_relation_att_feats, pp_position_relation_att_feats, p_position_att_masks = self._prepare_positon_realtion_feature(position_relation_att_feats, position_feats_masks, att_masks)
        p_semantic_relation_att_feats, pp_semantic_relation_att_feats, p_semantic_att_masks = self._prepare_semantic_realtion_feature(semantic_relation_att_feats, semantic_feats_masks, att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost
        # ===================== end ======================

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        if sample_n > 1:
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = \
                utils.repeat_tensors(sample_n,
                [p_fc_feats, p_att_feats, pp_att_feats, p_att_masks]
            )
            p_position_relation_att_feats, pp_position_relation_att_feats, p_position_att_masks = \
                utils.repeat_tensors(sample_n,
                [p_position_relation_att_feats, pp_position_relation_att_feats, p_position_att_masks]
            )
            p_semantic_relation_att_feats, pp_semantic_relation_att_feats, p_semantic_att_masks = \
                utils.repeat_tensors(sample_n,
                [p_semantic_relation_att_feats, pp_semantic_relation_att_feats, p_semantic_att_masks]
            )


        trigrams = [] # will be a list of batch_size dictionaries
        
        seq = fc_feats.new_full((batch_size*sample_n, self.seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size*sample_n, self.seq_length, self.vocab_size)
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.new_full([batch_size*sample_n], self.bos_idx, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(
                it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks,
                p_position_relation_att_feats, pp_position_relation_att_feats,p_position_att_masks,
                p_semantic_relation_att_feats, pp_semantic_relation_att_feats,p_semantic_att_masks,
                state, output_logsoftmax=output_logsoftmax, on_info=on_info
            )
            
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            if remove_bad_endings and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                prev_bad = np.isin(seq[:,t-1].data.cpu().numpy(), self.bad_endings_ix)
                # Make it impossible to generate bad_endings
                tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
                logprobs = logprobs + tmp

            # Mess with trigrams
            # Copy from https://github.com/lukemelas/image-paragraph-captioning
            if block_trigrams and t >= 3:
                # Store trigram generated at last step
                prev_two_batch = seq[:,t-3:t-1]
                for i in range(batch_size): # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current  = seq[i][t-1]
                    if t == 3: # initialize
                        trigrams.append({prev_two: [current]}) # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]: # add to list
                            trigrams[i][prev_two].append(current)
                        else: # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:,t-2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).to(logprobs.device) # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i,j] += 1
                # Apply mask to log probs
                #logprobs = logprobs - (mask * 1e9)
                alpha = 2.0 # = 4
                logprobs = logprobs + (mask * -0.693 * alpha) # ln(1/2) * alpha (alpha -> infty works best)

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            # stop when all finished
            if t == 0:
                unfinished = it != self.eos_idx
            else:
                it[~unfinished] = self.pad_idx # This allows eos_idx not being overwritten to 0
                logprobs = logprobs * unfinished.unsqueeze(1).to(logprobs)
                unfinished = unfinished & (it != self.eos_idx)
            seq[:,t] = it
            seqLogprobs[:,t] = logprobs
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs
    
    def init_hidden(self, bsz):
        weight = self.logit.weight \
                 if hasattr(self.logit, "weight") \
                 else self.logit[0].weight
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            # print("origin:",max_len)
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks
    
    def clip_relation_att(self, att_feats, masks, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            # print("relation:",max_len)
            att_feats = att_feats[:, :max_len, :max_len].contiguous()
            masks = masks[:, :max_len, :max_len].contiguous()
        return att_feats, masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats, att_masks

    def _prepare_positon_realtion_feature(self, att_feats, masks, att_masks):
        att_feats, att_masks = self.clip_relation_att(att_feats, masks, att_masks)
        # embed att feats
        att_feats = self.position_relation_att_embed(att_feats)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.position_relation_ctx2att(att_feats)

        return att_feats, p_att_feats, att_masks
    
    def _prepare_semantic_realtion_feature(self, att_feats, masks, att_masks):
        att_feats, att_masks = self.clip_relation_att(att_feats, masks, att_masks)
        # embed att feats
        att_feats = self.semantic_relation_att_embed(att_feats)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.semantic_relation_ctx2att(att_feats)

        return att_feats, p_att_feats, att_masks

    def get_logprobs_state(
        self, it, fc_feats, att_feats, p_att_feats, att_masks, 
        p_position_relation_att_feats, pp_position_relation_att_feats,p_position_att_masks,
        p_semantic_relation_att_feats, pp_semantic_relation_att_feats,p_semantic_att_masks,
        state, output_logsoftmax=1, on_info=None
    ):
        # 'it' contains a word index
        xt = self.embed(it)
        if isinstance(on_info, dict):  # for study on-lstm related methods
            output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks, on_info)
        else:
            output, state = self.core(
                xt, fc_feats, att_feats, p_att_feats, state, att_masks,
                p_position_relation_att_feats, pp_position_relation_att_feats,p_position_att_masks,
                p_semantic_relation_att_feats, pp_semantic_relation_att_feats,p_semantic_att_masks,
            )
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)

        return logprobs, state
