import torch
from . import losses
from ..utils.rewards import init_scorer, get_self_critical_reward
from captioning.utils.model_utils import idxs2captions

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = losses.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = losses.LanguageModelCriterion()
        self.rl_crit = losses.RewardCriterion()
        self.struc_crit = losses.StructureLosses(opt)

        # =================== add realtion loss ======
        self.gt_relation_path = opt.gt_relation_path
        self.gt_relation = torch.load(opt.gt_relation_path, map_location='cpu').to(opt.device)
        assert self.gt_relation.dtype==torch.int64
        # ================== end =====================
        self.ablation_study = opt.ablation_study if opt.ablation_study else '111'

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag, struc_flag, *args, **kwargs):
        opt = self.opt

        out = {}
        if struc_flag:
            if opt.structure_loss_weight < 1:
                lm_loss = self.crit(self.model(fc_feats, att_feats, labels[..., :-1], att_masks, *args, **kwargs), labels[..., 1:], masks[..., 1:])
            else:
                lm_loss = torch.tensor(0).type_as(fc_feats)
            if opt.structure_loss_weight > 0:
                gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                    opt={'sample_method':opt.train_sample_method,
                        'beam_size':opt.train_beam_size,
                        'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin'\
                            or not 'margin' in opt.structure_loss_type,
                        'sample_n': opt.train_sample_n},
                    mode='sample', *args, **kwargs)
                gts = [gts[_] for _ in gt_indices.tolist()]
                struc_loss = self.struc_crit(sample_logprobs, gen_result, gts)
            else:
                struc_loss = {'loss': torch.tensor(0).type_as(fc_feats),
                              'reward': torch.tensor(0).type_as(fc_feats)}
            loss = (1-opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * struc_loss['loss']
            out['lm_loss'] = lm_loss
            out['struc_loss'] = struc_loss['loss']
            out['reward'] = struc_loss['reward']
        elif not sc_flag:
            loss = self.crit(self.model(fc_feats, att_feats, labels[..., :-1], att_masks, *args, **kwargs), labels[..., 1:], masks[..., 1:])
            if self.ablation_study == '111' or self.ablation_study=='010' or self.ablation_study=='011' or self.ablation_study=='110':
                loss_relation = self._compute_realtion_loss(att_masks, *args, **kwargs)
                out['relation_loss'] = loss_relation

                loss = loss + self.opt.relation_loss_weight * loss_relation
            else:
                out['relation_loss'] = torch.tensor([0])
        else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks,
                    mode='sample',
                    opt={'sample_method': opt.sc_sample_method,
                         'beam_size': opt.sc_beam_size,
                         'block_trigrams': opt.block_trigrams,
                         'temperature': opt.temperature}, 
                    *args, **kwargs)
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                    opt={'sample_method':opt.train_sample_method,
                        'beam_size':opt.train_beam_size,
                        'sample_n': opt.train_sample_n},
                    mode='sample', 
                    *args, **kwargs)
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).to(sample_logprobs)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:,0].mean()
        out['loss'] = loss
        return out
    
    def _compute_realtion_loss(self, att_masks, *args, **kwargs):
        bboxes_class = kwargs.get("bboxes_class", None) 
        batch_size = bboxes_class.size(0)
        num_box = bboxes_class.size(1)

        self.gt_region_relation = att_masks.new_zeros([batch_size, num_box, num_box, self.opt.semantic_relation_size])

        for b in range(batch_size):
            pairs = torch.ones([num_box,num_box])
            pairs = torch.nonzero(pairs, as_tuple =False).long() 
            region_pairs = bboxes_class[b][pairs] 
            self.gt_region_relation[b] = self.gt_relation[region_pairs[:,0],region_pairs[:,1]].reshape([num_box, num_box,-1]) 

        loss_relation = self._multilabel_categorical_crossentropy(self.gt_region_relation, self.model.relation_cls_out)
        loss_relation = (loss_relation*self.model.relation_cls_mask).mean() # mask loss items
        return loss_relation


    def _multilabel_categorical_crossentropy(self, y_true, y_pred):
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss
