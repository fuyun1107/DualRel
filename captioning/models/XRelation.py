import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weight(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)

class PositionRelation(nn.Module):
    def __init__(self, input_dim, box_emb_dim, hidden_dim, output_dim, dropout_prob = 0.0):
        super(PositionRelation, self).__init__()
        
        self.input_dim = input_dim
        self.box_emb_dim = box_emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        self.position_encoding = nn.Sequential(
            nn.Linear(input_dim*2 + box_emb_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout_prob)
        )
        self._init_weights()
    
    def _init_weights(self):
        self.position_encoding.apply(_init_weight)

    def forward(self, object_feats, bboxes_embedding):
        """
        object_feats: (50, 512)
        bboxes_embedding: (50, 50, 512)
        mask: (50, 50)
        """
        pairs = torch.ones([object_feats.size(0),object_feats.size(0)])
        pairs = torch.nonzero(pairs, as_tuple =False).long() 

        features = object_feats[pairs.long()]
        bboxes_need_embedding = bboxes_embedding[pairs[:,0],pairs[:,1]]
        features = torch.cat((features[:,0],features[:,1], bboxes_need_embedding),1) 

        features = self.position_encoding(features)

        features = features.reshape([object_feats.size(0),object_feats.size(0),-1])

        return features


class SemanticRelation(nn.Module):
    def __init__(self, input_dim, class_emb_dim, hidden_dim, output_dim, relation_dim, layers=1, dropout_prob = 0.0):
        super(SemanticRelation, self).__init__()

        self.input_dim = input_dim
        self.class_emb_dim = class_emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.relation_dim = relation_dim
        self.layers = layers
        self.dropout_prob = dropout_prob

        self.semantic_encoding_layers = nn.ModuleList()
        for i in range(self.layers):
            self.semantic_encoding_layers.append(
                SemanticRelationUnit(input_dim, class_emb_dim, hidden_dim, output_dim, dropout_prob)
            )
        
        self.semantic_relation_predict_mlp = nn.Linear(output_dim, relation_dim)

        self._init_weights()
    
    def _init_weights(self):
        self.semantic_relation_predict_mlp.apply(_init_weight)

    def forward(self, object_feats, classes_embedding):
        features1 = None
        for i in range(self.layers):
            features, object_feats, classes_embedding = self.semantic_encoding_layers[i](object_feats, classes_embedding)
            if i==0:
                features1 = features
        
        relation_cls = self.semantic_relation_predict_mlp(features)

        features = features.reshape([object_feats.size(0),object_feats.size(0),-1]) 
        relation_cls = relation_cls.reshape([object_feats.size(0),object_feats.size(0),-1]) 

        return features, object_feats, relation_cls


class SemanticRelationUnit(nn.Module):
    def __init__(self, input_dim, class_emb_dim, hidden_dim, output_dim, dropout_prob = 0.0):
        super(SemanticRelationUnit, self).__init__()
        
        self.input_dim = input_dim
        self.class_emb_dim = class_emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        self.semantic_encoding = nn.Sequential(
            nn.Linear(input_dim*2 + class_emb_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim, output_dim),
        )

        self.add_weights_mlp = nn.Linear(output_dim, 1)

        self._init_weights()
    
    def _init_weights(self):
        self.semantic_encoding.apply(_init_weight)
        self.add_weights_mlp.apply(_init_weight)

    def forward(self, object_feats, classes_embedding):
        pairs = torch.ones([object_feats.size(0),object_feats.size(0)])
        pairs = torch.nonzero(pairs, as_tuple =False).long() 

        features = object_feats[pairs.long()] 
        classes_need_embedding = classes_embedding[pairs[:,0], pairs[:,1]] 

        features = torch.cat((features[:,0],features[:,1], classes_need_embedding),1) 

        # senmantic encoding
        features_out = self.semantic_encoding(features) 
        features = features_out

        features = features.reshape([object_feats.size(0),object_feats.size(0),-1])
        confidence = self.add_weights_mlp(features).squeeze(2) 
        confidence = torch.softmax(confidence, dim=1)

        enhanced_object_feats = torch.sum(confidence.unsqueeze(2) * features, dim=1 ,keepdim=False)

        return features_out, enhanced_object_feats, classes_embedding

def BoxRelationalEmbedding(f_g, dim_g=64, wave_len=1000, trignometric_embedding= False):
    """
    Given a tensor with bbox coordinates for detected objects on each batch image,
    this function computes a matrix for each image
    with entry (i,j) given by a vector representation of the
    displacement between the coordinates of bbox_i, and bbox_j
    input: np.array of shape=(batch_size, max_nr_bounding_boxes, 4)
    output: np.array of shape=(batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, 64)
    """
    #returns a relational embedding for each pair of bboxes, with dimension = dim_g
    #follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055

    batch_size = f_g.size(0)

    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=-1)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    #cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return embedding