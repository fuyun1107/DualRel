import torch


def regularization(model, weight_decay, p=2):
    reg_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            l2_reg = torch.norm(param, p=p)
            reg_loss = reg_loss + l2_reg
    reg_loss = weight_decay*reg_loss
    return reg_loss


def idx2caption(encode_caption, idx2word):
    para = []
    for i in range(encode_caption.size(0)):
        if idx2word[str(encode_caption[i].item())]=='<pad><bos><eos>':
            continue
        para.append(idx2word[str(encode_caption[i].item())])
    return " ".join(para)

def idxs2captions(encode_captions, idx2word):
    paragraphs = []
    for i in range(encode_captions.size(0)):
        para = []
        for j in range(encode_captions.size(1)):
            if idx2word[str(encode_captions[i][j].item())]=='<pad><bos><eos>':
                continue
            para.append(idx2word[str(encode_captions[i][j].item())])
        paragraphs.append(" ".join(para)) 
    return paragraphs


def save_model(model, optim, scheduler, opt, epoch, metrics):
    state = {'args': opt,
             'model': model.state_dict(),
             'epoch': epoch,
             'metrics': metrics,
             'optimizer': 0,
             'scheduler': 0}
    METEOR = metrics['METEOR']
    CIDEr = metrics['CIDEr']
    filename = opt.config_path+opt.model_name + f'_METEOR-{METEOR}_CIDEr-{CIDEr}.h5'
    torch.save(state, filename)
    print('save model to {}'.format(filename))


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
    return(embedding)