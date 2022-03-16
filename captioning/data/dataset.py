import json
import pickle
import os

import torch
import h5py
from torch.utils.data import Dataset
import numpy as np


class CaptionDataset(Dataset):

    def __init__(self,mapping_file_path='../data/imgs_train_path.txt',opt={}):
        super(CaptionDataset, self).__init__()
        self.mapping_file_path = mapping_file_path
        self.opt=opt

        self.encoded_paragraphs = json.load(open(self.opt.encoded_paragraphs_path, 'r'))
        self.mappings = self.get_mappings()

    def get_mappings(self):
        mappings = []
        with open(self.mapping_file_path, 'r') as f:
            for l in f:
                l = l.strip()
                if l:
                    l = os.path.basename(l).split('.')[0]
                    mappings.append(l)
        return mappings

    def __getitem__(self, i):

        gt_id = self.mappings[i]

        if 'train' in self.mapping_file_path:
            with h5py.File(self.opt.train_visual_features_path, 'r') as h:
                visual_feature = torch.tensor(h['feats'][i], dtype=torch.float)
                box_info = torch.tensor(h['boxes'][i], dtype=torch.float)
        elif 'val' in self.mapping_file_path:
            with h5py.File(self.opt.val_visual_features_path, 'r') as h:
                visual_feature = torch.tensor(h['feats'][i], dtype=torch.float)
                box_info = torch.tensor(h['boxes'][i], dtype=torch.float)
        else:
            with h5py.File(self.opt.test_visual_features_path, 'r') as h:
                visual_feature = torch.tensor(h['feats'][i], dtype=torch.float)
                box_info = torch.tensor(h['boxes'][i], dtype=torch.float)

        # con_stop_norm = torch.tensor(self.encoded_paragraphs[str(gt_id)][0], dtype=torch.long)
        encoded_paragraph = torch.tensor(self.encoded_paragraphs[str(gt_id)][0], dtype=torch.long)
        length = torch.tensor(self.encoded_paragraphs[str(gt_id)][1], dtype=torch.long)
        nonzero = (encoded_paragraph!=0).sum()+2
        mask = torch.zeros(self.opt.seq_length,dtype=torch.float)
        mask[:nonzero] = 1
        region_mask = 0

        return gt_id, 0, visual_feature, box_info, 0, encoded_paragraph, length, mask, region_mask

    def __len__(self):
        return len(self.mappings)


class CaptionDatasetFcClip(Dataset):

    def __init__(self,mapping_file_path='../data/imgs_train_path.txt',opt={}):
        super(CaptionDatasetFcClip, self).__init__()
        self.mapping_file_path = mapping_file_path
        self.opt=opt

        self.encoded_paragraphs = json.load(open(self.opt.encoded_paragraphs_path, 'r'))
        self.mappings = self.get_mappings()

    def get_mappings(self):
        mappings = []
        with open(self.mapping_file_path, 'r') as f:
            for l in f:
                l = l.strip()
                if l:
                    l = os.path.basename(l).split('.')[0]
                    mappings.append(l)
        return mappings

    def __getitem__(self, i):

        gt_id = self.mappings[i]

        if 'train' in self.mapping_file_path:
            with h5py.File(self.opt.clip_fc_features_path, 'r') as h:
                fc_feature = torch.tensor(h['train_fc_feats'][i], dtype=torch.float)
            with h5py.File(self.opt.train_visual_features_path, 'r') as h:
                visual_feature = torch.tensor(h['feats'][i], dtype=torch.float)
                box_info = torch.tensor(h['boxes'][i], dtype=torch.float)
        elif 'val' in self.mapping_file_path:
            with h5py.File(self.opt.clip_fc_features_path, 'r') as h:
                fc_feature = torch.tensor(h['val_fc_feats'][i], dtype=torch.float)
            with h5py.File(self.opt.val_visual_features_path, 'r') as h:
                visual_feature = torch.tensor(h['feats'][i], dtype=torch.float)
                box_info = torch.tensor(h['boxes'][i], dtype=torch.float)
        else:
            with h5py.File(self.opt.clip_fc_features_path, 'r') as h:
                fc_feature = torch.tensor(h['test_fc_feats'][i], dtype=torch.float)
            with h5py.File(self.opt.test_visual_features_path, 'r') as h:
                visual_feature = torch.tensor(h['feats'][i], dtype=torch.float)
                box_info = torch.tensor(h['boxes'][i], dtype=torch.float)

        # con_stop_norm = torch.tensor(self.encoded_paragraphs[str(gt_id)][0], dtype=torch.long)
        encoded_paragraph = torch.tensor(self.encoded_paragraphs[str(gt_id)][0], dtype=torch.long)
        length = torch.tensor(self.encoded_paragraphs[str(gt_id)][1], dtype=torch.long)
        nonzero = (encoded_paragraph!=0).sum()+2
        mask = torch.zeros(self.opt.seq_length,dtype=torch.float)
        mask[:nonzero] = 1
        region_mask = 0

        return gt_id, fc_feature, visual_feature, box_info, 0, encoded_paragraph, length, mask, region_mask

    def __len__(self):
        return len(self.mappings)


class CaptionDatasetClip(Dataset):

    def __init__(self,mapping_file_path='../data/imgs_train_path.txt',opt={}):
        super(CaptionDatasetClip, self).__init__()
        self.mapping_file_path = mapping_file_path
        self.opt=opt

        self.encoded_paragraphs = json.load(open(self.opt.encoded_paragraphs_path, 'r'))
        self.mappings = self.get_mappings()

    def get_mappings(self):
        mappings = []
        with open(self.mapping_file_path, 'r') as f:
            for l in f:
                l = l.strip()
                if l:
                    l = os.path.basename(l).split('.')[0]
                    mappings.append(l)
        return mappings

    def __getitem__(self, i):

        gt_id = self.mappings[i]

        if 'train' in self.mapping_file_path:
            with h5py.File(self.opt.clip_fc_features_path, 'r') as h:
                fc_feature = torch.tensor(h['train_fc_feats'][i], dtype=torch.float)
            with h5py.File(self.opt.clip_region_features_path, 'r') as h:
                visual_feature = torch.tensor(h['train_region_feats'][i], dtype=torch.float)
            with h5py.File(self.opt.train_visual_features_path, 'r') as h:
                box_info = torch.tensor(h['boxes'][i], dtype=torch.float)
        elif 'val' in self.mapping_file_path:
            with h5py.File(self.opt.clip_fc_features_path, 'r') as h:
                fc_feature = torch.tensor(h['val_fc_feats'][i], dtype=torch.float)
            with h5py.File(self.opt.clip_region_features_path, 'r') as h:
                visual_feature = torch.tensor(h['val_region_feats'][i], dtype=torch.float)
            with h5py.File(self.opt.val_visual_features_path, 'r') as h:
                box_info = torch.tensor(h['boxes'][i], dtype=torch.float)
        else:
            with h5py.File(self.opt.clip_fc_features_path, 'r') as h:
                fc_feature = torch.tensor(h['test_fc_feats'][i], dtype=torch.float)
            with h5py.File(self.opt.clip_region_features_path, 'r') as h:
                visual_feature = torch.tensor(h['test_region_feats'][i], dtype=torch.float)
            with h5py.File(self.opt.test_visual_features_path, 'r') as h:
                box_info = torch.tensor(h['boxes'][i], dtype=torch.float)

        # con_stop_norm = torch.tensor(self.encoded_paragraphs[str(gt_id)][0], dtype=torch.long)
        encoded_paragraph = torch.tensor(self.encoded_paragraphs[str(gt_id)][0], dtype=torch.long)
        length = torch.tensor(self.encoded_paragraphs[str(gt_id)][1], dtype=torch.long)
        nonzero = (encoded_paragraph!=0).sum()+2
        mask = torch.zeros(self.opt.seq_length,dtype=torch.float)
        mask[:nonzero] = 1
        region_mask = 0
        
        return gt_id, fc_feature, visual_feature, box_info, 0, encoded_paragraph, length, mask, region_mask

    def __len__(self):
        return len(self.mappings)


class CaptionDatasetBU(Dataset):

    def __init__(self,mapping_file_path='../data/imgs_train_path.txt',opt={}):
        super(CaptionDatasetBU, self).__init__()
        self.mapping_file_path = mapping_file_path
        self.opt=opt
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)

        self.encoded_paragraphs = json.load(open(self.opt.encoded_paragraphs_path, 'r'))
        self.mappings = self.get_mappings()

    def get_mappings(self):
        mappings = []
        with open(self.mapping_file_path, 'r') as f:
            for l in f:
                l = l.strip()
                if l:
                    l = os.path.basename(l).split('.')[0]
                    if str(l) in ["2346046", "2341671"]:
                        continue
                    mappings.append(l)
        return mappings

    def __getitem__(self, i):

        gt_id = self.mappings[i]

        att_feats = np.load(os.path.join(self.opt.input_att_dir, str(gt_id) + '.npz'))['feat']
        att_feats = att_feats.reshape(-1, att_feats.shape[-1])
        if self.norm_att_feat:
                att_feats = att_feats / np.linalg.norm(att_feats, 2, 1, keepdims=True)
    
        box_info = np.load(os.path.join(self.opt.input_box_dir, str(gt_id) + '.npy'))
        assert att_feats.shape[0]==box_info.shape[0]
        # merge att_feats
        max_att_len = self.opt.padding_region_length
        if att_feats.shape[0]>=max_att_len:
            att_feats_ = att_feats[:max_att_len,:]

            box_info_ = box_info[:max_att_len,:]

            att_masks = np.ones([max_att_len], dtype='float32')
        else:
            att_feats_ = np.zeros([max_att_len, att_feats.shape[1]], dtype = 'float32')
            att_feats_[:att_feats.shape[0],:] = att_feats
            
            box_info_ = np.zeros([max_att_len, 4], dtype = 'float32')
            box_info_[:att_feats.shape[0],:] = box_info

            att_masks = np.zeros([att_feats_.shape[0]], dtype='float32')
            att_masks[:att_feats.shape[0]] = 1

        # # devided by image width and height
        # x1,y1,x2,y2 = np.hsplit(box_feat, 4)
        # h,w = self.info['images'][ix]['height'], self.info['images'][ix]['width']
        # box_feat = np.hstack((x1/w, y1/h, x2/w, y2/h, (x2-x1)*(y2-y1)/(w*h))) # question? x2-x1+1??
        # if self.norm_box_feat:
        #     box_feat = box_feat / np.linalg.norm(box_feat, 2, 1, keepdims=True)
        # att_feat = np.hstack([att_feat, box_feat])
        # # sort the features by the size of boxes
        # att_feat = np.stack(sorted(att_feat, key=lambda x:x[-1], reverse=True))

        fc_feats = np.load(os.path.join(self.opt.input_fc_dir, str(gt_id) + '.npy'))

        fc_feature = torch.from_numpy(fc_feats)
        visual_feature = torch.from_numpy(att_feats_)
        region_mask = torch.from_numpy(att_masks)
        box_info = torch.from_numpy(box_info_)


        # con_stop_norm = torch.tensor(self.encoded_paragraphs[str(gt_id)][0], dtype=torch.long)
        encoded_paragraph = torch.tensor(self.encoded_paragraphs[str(gt_id)][0], dtype=torch.long)
        length = torch.tensor(self.encoded_paragraphs[str(gt_id)][1], dtype=torch.long)
        nonzero = (encoded_paragraph!=0).sum()+2
        mask = torch.zeros(self.opt.seq_length,dtype=torch.float)
        mask[:nonzero] = 1

        return gt_id, fc_feature, visual_feature, box_info, 0, encoded_paragraph, length, mask, region_mask

    def __len__(self):
        return len(self.mappings)


class CaptionDatasetBUWithOA(Dataset):

    def __init__(self,mapping_file_path='../data/imgs_train_path.txt',opt={}):
        super(CaptionDatasetBUWithOA, self).__init__()
        self.mapping_file_path = mapping_file_path
        self.opt=opt
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)

        self.encoded_paragraphs = json.load(open(self.opt.encoded_paragraphs_path, 'r'))
        self.mappings = self.get_mappings()

    def get_mappings(self):
        mappings = []
        with open(self.mapping_file_path, 'r') as f:
            for l in f:
                l = l.strip()
                if l:
                    l = os.path.basename(l).split('.')[0]
                    if str(l) in ["2346046", "2341671"]:
                        continue
                    mappings.append(l)
        return mappings

    def __getitem__(self, i):

        gt_id = self.mappings[i]

        att_feats = np.load(os.path.join(self.opt.input_att_dir, str(gt_id) + '.npz'))['feat']
        att_feats = att_feats.reshape(-1, att_feats.shape[-1])
        if self.norm_att_feat:
                att_feats = att_feats / np.linalg.norm(att_feats, 2, 1, keepdims=True)
    
        box_info = np.load(os.path.join(self.opt.input_box_dir, str(gt_id) + '.npy'))
        object_class = np.load(os.path.join(self.opt.input_object_dir, str(gt_id) + '.npy'))
        attr = np.load(os.path.join(self.opt.input_attr_dir, str(gt_id) + '.npy'))
        assert att_feats.shape[0] == box_info.shape[0] and att_feats.shape[0] == object_class.shape[0] \
            and att_feats.shape[0] == attr.shape[0]

        # merge att_feats
        max_att_len = self.opt.padding_region_length
        if att_feats.shape[0]>=max_att_len:
            att_feats_ = att_feats[:max_att_len,:]

            box_info_ = box_info[:max_att_len,:]

            object_class_ = object_class[:max_att_len]
            attr_ = attr[:max_att_len]

            att_masks = np.ones([max_att_len], dtype='float32')
        else:
            att_feats_ = np.zeros([max_att_len, att_feats.shape[1]], dtype = 'float32')
            att_feats_[:att_feats.shape[0],:] = att_feats
            
            box_info_ = np.zeros([max_att_len, 4], dtype = 'float32')
            box_info_[:att_feats.shape[0],:] = box_info

            object_class_ = np.zeros([max_att_len], dtype = 'int')
            attr_ = np.zeros([max_att_len], dtype = 'int')
            object_class_[:att_feats.shape[0]] = object_class
            attr_[:att_feats.shape[0]] = attr

            att_masks = np.zeros([att_feats_.shape[0]], dtype='float32')
            att_masks[:att_feats.shape[0]] = 1

        fc_feats = np.load(os.path.join(self.opt.input_fc_dir, str(gt_id) + '.npy'))

        fc_feature = torch.from_numpy(fc_feats)
        visual_feature = torch.from_numpy(att_feats_)
        region_mask = torch.from_numpy(att_masks)
        box_info = torch.from_numpy(box_info_)
        object_class = torch.from_numpy(object_class_)
        attr = torch.from_numpy(attr_)

        encoded_paragraph = torch.tensor(self.encoded_paragraphs[str(gt_id)][0], dtype=torch.long)
        length = torch.tensor(self.encoded_paragraphs[str(gt_id)][1], dtype=torch.long)
        nonzero = (encoded_paragraph!=0).sum()+2
        mask = torch.zeros(self.opt.seq_length,dtype=torch.float)
        mask[:nonzero] = 1

        data={
            "gt_ids":gt_id, 
            "fc_feats":fc_feature, 
            "att_feats":visual_feature, 
            "boxes":box_info, 
            "con_stop":0, 
            "labels":encoded_paragraph, 
            "lens":length, 
            "masks":mask, 
            "att_masks":region_mask, 
            "object_classes":object_class, 
            "attrs":attr
        }

        return data

    def __len__(self):
        return len(self.mappings)

def collate_bu_fn(batch):
    """
    batch: list of tensor
    """
    batch_size = len(batch)

    gt_ids = []
    fc_feature = []
    visual_feature = []
    box_info = []
    encoded_paragraph = []
    length = []
    mask = []
    region_mask = []
    object_class = []
    attr = []
    
    for  i in range(batch_size):
        gt_ids.append(batch[i]['gt_ids'])
        fc_feature.append(batch[i]['fc_feats'])
        visual_feature.append(batch[i]['att_feats'])
        box_info.append(batch[i]['boxes'])
        encoded_paragraph.append(batch[i]['labels'])
        length.append(batch[i]['lens'])
        mask.append(batch[i]['masks'])
        region_mask.append(batch[i]['att_masks'])
        object_class.append(batch[i]['object_classes'])
        attr.append(batch[i]['attrs'])

    fc_feature = torch.stack(fc_feature, dim=0)
    visual_feature = torch.stack(visual_feature, dim=0)
    box_info = torch.stack(box_info, dim=0)
    encoded_paragraph = torch.stack(encoded_paragraph, dim=0)
    length = torch.stack(length, dim=0)
    mask = torch.stack(mask, dim=0)
    region_mask = torch.stack(region_mask, dim=0)
    object_class = torch.stack(object_class, dim=0)
    attr = torch.stack(attr, dim=0)

    data={
            "gt_ids":gt_ids, 
            "fc_feats":fc_feature, 
            "att_feats":visual_feature, 
            "boxes":box_info, 
            "con_stop":0, 
            "labels":encoded_paragraph, 
            "lens":length, 
            "masks":mask, 
            "att_masks":region_mask, 
            "object_classes":object_class, 
            "attrs":attr
        }

    return data


class CaptionDatasetBUWithOA_trigram_vocab(Dataset):
    def __init__(self,mapping_file_path='../data/imgs_train_path.txt',opt={}):
        super(CaptionDatasetBUWithOA_trigram_vocab, self).__init__()
        self.mapping_file_path = mapping_file_path
        self.opt=opt
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)

        self.encoded_paragraphs = h5py.File(self.opt.encoded_paragraphs_path, 'r', driver='core')
        self.label_start_ix = self.encoded_paragraphs['label_start_ix'][:]
        self.label_end_ix = self.encoded_paragraphs['label_end_ix'][:]

        self.mappings = self.get_mappings()

        self.gtids2idx = {}
        paratalk_vocab = json.load(open(self.opt.paratalk_vocab_path,'r'))
        for i in range(len(paratalk_vocab["images"])):
            self.gtids2idx[str(paratalk_vocab["images"][i]["id"])] = i

    def get_mappings(self):
        mappings = []
        with open(self.mapping_file_path, 'r') as f:
            for l in f:
                l = l.strip()
                if l:
                    l = os.path.basename(l).split('.')[0]
                    if str(l) in ["2346046", "2341671"]:
                        continue
                    mappings.append(l)
        return mappings

    def __getitem__(self, i):

        gt_id = self.mappings[i]

        att_feats = np.load(os.path.join(self.opt.input_att_dir, str(gt_id) + '.npz'))['feat']
        att_feats = att_feats.reshape(-1, att_feats.shape[-1])
        if self.norm_att_feat:
                att_feats = att_feats / np.linalg.norm(att_feats, 2, 1, keepdims=True)
    
        box_info = np.load(os.path.join(self.opt.input_box_dir, str(gt_id) + '.npy'))
        object_class = np.load(os.path.join(self.opt.input_object_dir, str(gt_id) + '.npy'))
        attr = np.load(os.path.join(self.opt.input_attr_dir, str(gt_id) + '.npy'))
        assert att_feats.shape[0] == box_info.shape[0] and att_feats.shape[0] == object_class.shape[0] \
            and att_feats.shape[0] == attr.shape[0]

        # merge att_feats
        max_att_len = self.opt.padding_region_length
        if att_feats.shape[0]>=max_att_len:
            att_feats_ = att_feats[:max_att_len,:]

            box_info_ = box_info[:max_att_len,:]

            object_class_ = object_class[:max_att_len]
            attr_ = attr[:max_att_len]

            att_masks = np.ones([max_att_len], dtype='float32')
        else:
            att_feats_ = np.zeros([max_att_len, att_feats.shape[1]], dtype = 'float32')
            att_feats_[:att_feats.shape[0],:] = att_feats
            
            box_info_ = np.zeros([max_att_len, 4], dtype = 'float32')
            box_info_[:att_feats.shape[0],:] = box_info

            object_class_ = np.zeros([max_att_len], dtype = 'int')
            attr_ = np.zeros([max_att_len], dtype = 'int')
            object_class_[:att_feats.shape[0]] = object_class
            attr_[:att_feats.shape[0]] = attr

            att_masks = np.zeros([att_feats_.shape[0]], dtype='float32')
            att_masks[:att_feats.shape[0]] = 1

        # # devided by image width and height
        # x1,y1,x2,y2 = np.hsplit(box_feat, 4)
        # h,w = self.info['images'][ix]['height'], self.info['images'][ix]['width']
        # box_feat = np.hstack((x1/w, y1/h, x2/w, y2/h, (x2-x1)*(y2-y1)/(w*h))) # question? x2-x1+1??
        # if self.norm_box_feat:
        #     box_feat = box_feat / np.linalg.norm(box_feat, 2, 1, keepdims=True)
        # att_feat = np.hstack([att_feat, box_feat])
        # # sort the features by the size of boxes
        # att_feat = np.stack(sorted(att_feat, key=lambda x:x[-1], reverse=True))

        fc_feats = np.load(os.path.join(self.opt.input_fc_dir, str(gt_id) + '.npy'))

        fc_feature = torch.from_numpy(fc_feats)
        visual_feature = torch.from_numpy(att_feats_)
        region_mask = torch.from_numpy(att_masks)
        box_info = torch.from_numpy(box_info_)
        object_class = torch.from_numpy(object_class_)
        attr = torch.from_numpy(attr_)


        # con_stop_norm = torch.tensor(self.encoded_paragraphs[str(gt_id)][0], dtype=torch.long)
        ix = self.gtids2idx[str(gt_id)]
        encoded_paragraph = torch.tensor([0]+self.encoded_paragraphs['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]][0].tolist(), dtype=torch.long)
        length = torch.tensor(self.encoded_paragraphs['label_length'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]].tolist()[0], dtype=torch.long)
        nonzero = (encoded_paragraph!=0).sum()+2
        # print(encoded_paragraph)
        # print(length)
        assert self.opt.seq_length==(len(encoded_paragraph)-1)
        mask = torch.zeros(self.opt.seq_length+2,dtype=torch.float)
        mask[:nonzero] = 1

        data={
            "gt_ids":gt_id, 
            "fc_feats":fc_feature, 
            "att_feats":visual_feature, 
            "boxes":box_info, 
            "con_stop":0, 
            "labels":encoded_paragraph, 
            "lens":length, 
            "masks":mask, 
            "att_masks":region_mask, 
            "object_classes":object_class, 
            "attrs":attr
        }

        return data

    def __len__(self):
        return len(self.mappings)