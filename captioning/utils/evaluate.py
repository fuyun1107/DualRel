from tqdm import tqdm
import torch

import sys
sys.path.append("./coco-caption")
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from captioning.utils.model_utils import idxs2captions


def eval_model_a_coco(model, eval_loader, gt_caption_path, opt, stop=False):
    with torch.no_grad():
        res = [] 
        coco_format_candidates = [] 

        for batch, (gt_ids, fc_feats, region_feats, box_infos, con_stop, encode_caption, caption_len, masks, region_masks) in tqdm(enumerate(eval_loader)):
            fc_feats = fc_feats.to(next(model.parameters()).device)
            region_feats = region_feats.to(next(model.parameters()).device)
            box_infos = box_infos.to(next(model.parameters()).device)
            con_stop = con_stop.to(next(model.parameters()).device)
            encode_caption = encode_caption.to(next(model.parameters()).device)
            caption_len = caption_len.to(next(model.parameters()).device)
            masks = masks.to(next(model.parameters()).device)
            att_masks = None
            att_feats = region_feats
            labels = encode_caption

            caption_len = caption_len.cpu().numpy().tolist()
            paragraphs_gt = idxs2captions(encode_caption, opt.idx2word)


            seq, seqLogprobs= model(fc_feats, att_feats, att_masks, opt=vars(opt), mode='sample')

            paragraphs_pred = idxs2captions(seq, opt.idx2word)

            for i in range(encode_caption.shape[0]):
                coco_format_candidates.append({
                    "image_id": int(gt_ids[i]),
                    "caption": paragraphs_pred[i]
                })
                res.append({
                    'img_id': int(gt_ids[i]),
                    'gt': paragraphs_gt[i],
                    'pred': paragraphs_pred[i]
                })

        import json
        with open('./eval_result.json','w') as f:
            json.dump(res,f,indent=1)

        coco = COCO(gt_caption_path)  # load coco format ground truth
        cocoRes = coco.loadRes(coco_format_candidates)  # list or path
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.evaluate()
        metrics = cocoEval.eval

        for key,value in metrics.items():
            metrics[key]=round(value,4)
        return metrics, res

def eval_model_bu_coco(model, eval_loader, gt_caption_path, opt):
    with torch.no_grad():
        res = []
        coco_format_candidates = [] 

        for batch, (gt_ids, fc_feats, region_feats, box_infos, con_stop, encode_caption, caption_len, masks, region_masks) in tqdm(enumerate(eval_loader)):
            fc_feats = fc_feats.to(next(model.parameters()).device)
            region_feats = region_feats.to(next(model.parameters()).device)
            box_infos = box_infos.to(next(model.parameters()).device)
            con_stop = con_stop.to(next(model.parameters()).device)
            encode_caption = encode_caption.to(next(model.parameters()).device)
            caption_len = caption_len.to(next(model.parameters()).device)
            masks = masks.to(next(model.parameters()).device)
            masks = masks.to(opt.device)
            if region_masks.sum()==region_masks.numel():
                att_masks = None
            else:
                att_masks = region_masks.to(opt.device)
            att_feats = region_feats
            labels = encode_caption


            caption_len = caption_len.cpu().numpy().tolist()
            paragraphs_gt = idxs2captions(encode_caption, opt.idx2word)


            seq, seqLogprobs= model(fc_feats, att_feats, att_masks, opt=vars(opt), mode='sample')

            paragraphs_pred = idxs2captions(seq, opt.idx2word)


            for i in range(encode_caption.shape[0]):
                coco_format_candidates.append({
                    "image_id": int(gt_ids[i]),
                    "caption": paragraphs_pred[i]
                })
                res.append({
                    'img_id': int(gt_ids[i]),
                    'gt': paragraphs_gt[i],
                    'pred': paragraphs_pred[i]
                })

        import json
        with open('./eval_result.json','w') as f:
            json.dump(res,f,indent=1)

        coco = COCO(gt_caption_path)  # load coco format ground truth
        cocoRes = coco.loadRes(coco_format_candidates)  # list or path
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.evaluate()
        metrics = cocoEval.eval

        for key,value in metrics.items():
            metrics[key]=round(value,4)
        return metrics, res


def eval_model_bu_dualearly_coco(model, eval_loader, gt_caption_path, opt):
    with torch.no_grad():
        res = [] 
        coco_format_candidates = [] 


        for batch, (gt_ids, fc_feats, region_feats, box_infos, con_stop, encode_caption, caption_len, masks, region_masks) in tqdm(enumerate(eval_loader)):
            fc_feats = fc_feats.to(next(model.parameters()).device)
            region_feats = region_feats.to(next(model.parameters()).device)
            box_infos = box_infos.to(next(model.parameters()).device)
            con_stop = con_stop.to(next(model.parameters()).device)
            encode_caption = encode_caption.to(next(model.parameters()).device)
            caption_len = caption_len.to(next(model.parameters()).device)
            masks = masks.to(next(model.parameters()).device)
            masks = masks.to(opt.device)
            att_masks = region_masks.to(opt.device)
            att_feats = region_feats
            labels = encode_caption


            caption_len = caption_len.cpu().numpy().tolist()
            paragraphs_gt = idxs2captions(encode_caption, opt.idx2word)


            seq, seqLogprobs= model(fc_feats, att_feats, att_masks, opt=vars(opt), mode='sample', bboxes = box_infos, rel_iou_threshold=0.1)

            paragraphs_pred = idxs2captions(seq, opt.idx2word)


            for i in range(encode_caption.shape[0]):
                coco_format_candidates.append({
                    "image_id": int(gt_ids[i]),
                    "caption": paragraphs_pred[i]
                })
                res.append({
                    'img_id': int(gt_ids[i]),
                    'gt': paragraphs_gt[i],
                    'pred': paragraphs_pred[i]
                })

        import json
        with open('./eval_result.json','w') as f:
            json.dump(res,f,indent=1)

        coco = COCO(gt_caption_path)  # load coco format ground truth
        cocoRes = coco.loadRes(coco_format_candidates)  # list or path
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.evaluate()
        metrics = cocoEval.eval

        for key,value in metrics.items():
            metrics[key]=round(value,4)
        return metrics, res

def eval_model_bu_x_coco(model, eval_loader, gt_caption_path, opt):
    with torch.no_grad():
        res = [] 
        coco_format_candidates = []

        for batch, data in tqdm(enumerate(eval_loader)):
            gt_ids = data['gt_ids']
            fc_feats = data['fc_feats'].to(next(model.parameters()).device)
            att_feats = data['att_feats'].to(next(model.parameters()).device)
            boxes = data['boxes'].to(next(model.parameters()).device)
            labels = data['labels'].to(next(model.parameters()).device)
            caption_len = data['lens'].to(next(model.parameters()).device)
            masks = data['masks'].to(next(model.parameters()).device)
            att_masks = data['att_masks'].to(next(model.parameters()).device)
            object_classes = data['object_classes'].to(next(model.parameters()).device)
            attrs = data['attrs'].to(next(model.parameters()).device)

            caption_len = caption_len.cpu().numpy().tolist()
            paragraphs_gt = idxs2captions(labels, opt.idx2word)


            seq, seqLogprobs= model(
                fc_feats, att_feats, att_masks, opt=vars(opt), mode='sample', 
                bboxes = boxes, bboxes_class = object_classes, rel_iou_threshold=0.1
            )

            paragraphs_pred = idxs2captions(seq, opt.idx2word)

            for i in range(labels.shape[0]):
                coco_format_candidates.append({
                    "image_id": int(gt_ids[i]),
                    "caption": paragraphs_pred[i]
                })
                res.append({
                    'img_id': int(gt_ids[i]),
                    'gt': paragraphs_gt[i],
                    'pred': paragraphs_pred[i]
                })

        import json
        with open('./eval_result.json','w') as f:
            json.dump(res,f,indent=1)

        coco = COCO(gt_caption_path)  # load coco format ground truth
        cocoRes = coco.loadRes(coco_format_candidates)  # list or path
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.evaluate()
        metrics = cocoEval.eval

        for key,value in metrics.items():
            metrics[key]=round(value,4)
        return metrics, res