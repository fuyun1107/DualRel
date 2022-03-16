import torch
import torch.nn as nn
import torch.nn.functional as F

def build_layers(layers_dim, dropout_rate=0):
    layers = []
    for i in range(len(layers_dim) - 1):
        input_dim = layers_dim[i]
        output_dim = layers_dim[i+1]
        layers.append(nn.Linear(input_dim, output_dim))
        #layers.append(nn.ReLU())
        
        if i != (len(layers_dim) - 2): # not last layer
            #layers.append(nn.BatchNorm1d(output_dim))
            layers.append(nn.ReLU())
            #layers.append(nn.LeakyReLU())
        
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
    return nn.Sequential(*layers)


def _init_weight(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


# Asymmetric
class PositionRelationEncodeUnit(nn.Module):
    # single layer
    def __init__(self, input_dim, output_dim, hidden_dim, gconv_pooling='avg'):
        super(PositionRelationEncodeUnit, self).__init__()
        self.gconv_pooling = gconv_pooling
        conv_layers = [sum(input_dim), hidden_dim, sum(output_dim)]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = build_layers(conv_layers)
        self._init_weights()
    
    def _init_weights(self):
        self.conv.apply(_init_weight)

    def forward(self, object_feats, bboxes_embedding, pairs):
        features = object_feats[pairs.long()]
        bboxes_need_embedding = bboxes_embedding[pairs.long()[:,0],pairs.long()[:,1]]

        features = torch.cat((features[:,0],features[:,1], bboxes_need_embedding),1)
        output = self.conv(features)
        feat_dict = {}
        new_feats = object_feats.clone()
        for i in range(pairs.max().item()+1):
            feat_dict[i] = [object_feats[i]]
        
        for feat, pair in zip(output, pairs):
            sub, obj = pair[0].item(), pair[1].item()
            feat_dict[sub].append(feat)

        for key in feat_dict:
            if len(feat_dict[key]) == 0:
                continue
            feats = torch.stack(feat_dict[key])

            new_feat = torch.mean(feats, dim=0)
            new_feats[key] = new_feat
        return new_feats, bboxes_embedding, pairs


class PositionRelationEncode(nn.Module):
    def __init__(self, feat_dim=2048, hidden_dim=2048, box_emb_dim=2048, gconv_layers=1):
        super(PositionRelationEncode, self).__init__()
        print('PositionRelationEncode_layers = %i'%gconv_layers)
        self.gconv_layers = gconv_layers
        if gconv_layers == 0:
            self.gconv = Identity()
            assert False
        elif gconv_layers > 0:
            self.gconv = PositionRelationEncodeUnit([feat_dim, feat_dim, box_emb_dim], [feat_dim], hidden_dim)
        
    def forward(self, obj_feats, bboxes_embedding, pairs):#, w2v):
        for i in range(self.gconv_layers):
            obj_feats, bboxes_embedding, pairs = self.gconv(obj_feats, bboxes_embedding, pairs)
        return obj_feats





# 使用矩阵乘法加速
class PositionRelationEncodeUnit_GCN(nn.Module):
    # single layer
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(PositionRelationEncodeUnit_GCN, self).__init__()
        conv_layers = [sum(input_dim), hidden_dim, sum(output_dim)]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = build_layers(conv_layers)

        self.weight_mlp = nn.Sequential(
            nn.Linear(sum(input_dim), hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(hidden_dim),
            # nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()
    
    def _init_weights(self):
        self.conv.apply(_init_weight)

    def forward(self, object_feats, bboxes_embedding, mask):
        pairs = torch.ones([object_feats.size(0),object_feats.size(0)])
        pairs = torch.nonzero(pairs, as_tuple =False).long() # 50*50

        features = object_feats[pairs.long()]
        bboxes_need_embedding = bboxes_embedding[pairs.long()[:,0],pairs.long()[:,1]]
        features = torch.cat((features[:,0],features[:,1], bboxes_need_embedding),1) # (2500,1024)

        # print(features.size())
        confidence = self.weight_mlp(features).squeeze(1) # (2500)
        confidence = torch.softmax(confidence.reshape([-1,object_feats.size(0)]), dim=1) # (50,50)
        # print(confidence.size())

        # mask
        # print(confidence)
        confidence = confidence * mask
        confidence =confidence / (confidence.sum(1, keepdim=True)+1e-8) # normalize to 1,要加个小的数，防止除以0
        # print(confidence.sum(1, keepdim=True))

        # mlp tofeatures
        features = features.reshape([object_feats.size(0),object_feats.size(0),-1]) # (50,50,1024)
        features = self.conv(features)

        new_feats = torch.sum(confidence.unsqueeze(2) * features, dim=1 ,keepdim=False)

        return new_feats, bboxes_embedding, mask

class PositionRelationEncode_GCN(nn.Module):
    def __init__(self, feat_dim=2048, hidden_dim=2048, box_emb_dim=2048, gconv_layers=1):
        super(PositionRelationEncode_GCN, self).__init__()
        print('PositionRelationEncode_layers = %i'%gconv_layers)
        self.gconv_layers = gconv_layers
        if gconv_layers == 0:
            self.gconv = Identity()
            assert False
        elif gconv_layers > 0:
            self.gconv = PositionRelationEncodeUnit_GCN([feat_dim, feat_dim, box_emb_dim], [feat_dim], hidden_dim)
        
    def forward(self, obj_feats, bboxes_embedding, mask):#, w2v):
        for i in range(self.gconv_layers):
            obj_feats, bboxes_embedding, mask = self.gconv(obj_feats, bboxes_embedding, mask)
        return obj_feats