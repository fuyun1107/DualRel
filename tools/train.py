from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, json, time, datetime, random, sys
sys.path.append("./")

import tools.opts_dual_r  as opts
opt = opts.parse_opt()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_visible_devices

import torch
from torch import nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import numpy as np
from loguru import logger
from torch.utils.data import DataLoader

# ruotianluo
import captioning.models as models
import captioning.utils.misc as utils
from captioning.utils.rewards import init_scorer
from captioning.modules.loss_wrapper import LossWrapper

# para
from captioning.utils.model_utils import save_model
from captioning.data.dataset import CaptionDatasetBUWithOA_trigram_vocab as CaptionDataset
from captioning.utils.evaluate import eval_model_bu_x_coco as eval_model


def train(opt):

    ################################
    # Build dataloader
    ################################
    start_time = time.time()
    train_dataset = CaptionDataset(mapping_file_path=opt.train_mapping_file_path, opt=opt)
    train_loader = DataLoader(train_dataset, opt.batch_size, shuffle=opt.shuffle, num_workers=opt.num_workers, pin_memory=opt.pin_memory)

    val_dataset = CaptionDataset(mapping_file_path=opt.val_mapping_file_path, opt=opt)
    val_loader = DataLoader(val_dataset, opt.batch_size, shuffle=opt.shuffle, num_workers=opt.num_workers, pin_memory=opt.pin_memory)
    ###########################
    # tensorboard logger
    ###########################
    if opt.use_tb:
        logdir = os.path.join('runs', opt.model_name)
        writer = SummaryWriter(logdir=logdir)

    ##########################
    # Build model
    ##########################
    start_time = time.time()
    model = models.setup(opt)
    if opt.pretrained:
        model.load_state_dict(torch.load(opt.pretrained_path, map_location='cpu')['model'])
    lw_model = LossWrapper(model, opt).to(opt.device)
    dp_lw_model = lw_model
    dp_lw_model.train()

    ##########################
    #  Build optimizer
    ##########################
    if opt.noamopt:
        assert opt.caption_model in ['transformer', 'bert', 'm2transformer'], 'noamopt can only work with transformer'
        optimizer = utils.get_std_opt(model, optim_func=opt.optim, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
        assert False
    elif opt.reduce_on_plateau:
        optimizer = utils.build_optimizer(model.parameters(), opt)
        optimizer = utils.ReduceLROnPlateau(optimizer,
                                            factor=opt.reduce_on_plateau_factor,
                                            patience=opt.reduce_on_plateau_patience)
        assert False
    else:
        optimizer = utils.build_optimizer(model.parameters(), opt)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.learning_rate_decay_every, gamma=opt.learning_rate_decay_rate)
    # Load the optimizer
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from,"optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    #######################
    # Start training
    #######################
    loss_history={}
    lr_history={}
    ss_prob_history={}
    iteration = 0
    best_eval_score = 0.10
    for epoch in range(0, opt.max_epochs):
        # Assign the scheduled sampling prob
        if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
            frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
            opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
            model.ss_prob = opt.ss_prob

        # If start self critical training
        if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
            assert opt.self_critical_after==0
            sc_flag = True
            init_scorer(opt.cached_tokens)
        else:
            sc_flag = False
        
        # If start structure loss training
        if opt.structure_after != -1 and epoch >= opt.structure_after:
            struc_flag = True
            init_scorer(opt.cached_tokens)
            assert False
        else:
            struc_flag = False


        start_time = time.time()
        for batch, data in enumerate(train_loader):
            gt_ids = data['gt_ids']
            fc_feats = data['fc_feats'].to(opt.device)
            att_feats = data['att_feats'].to(opt.device)
            boxes = data['boxes'].to(opt.device)
            labels = data['labels'].to(opt.device)
            caption_len = data['lens'].to(opt.device)
            masks = data['masks'].to(opt.device)
            att_masks = data['att_masks'].to(opt.device)
            object_classes = data['object_classes'].to(opt.device)
            attrs = data['attrs'].to(opt.device)

            gts = labels[:,1:].unsqueeze(1) 
            gts = gts.data.cpu().numpy()

            data_time = time.time() - start

            if opt.use_warmup and (iteration < opt.noamopt_warmup):
                opt.current_lr = opt.learning_rate * (iteration+1) / opt.noamopt_warmup
                utils.set_lr(optimizer, opt.current_lr)
                assert False

            # Forward pass and loss
            torch.cuda.synchronize()
            optimizer.zero_grad()
            model_out = dp_lw_model(
                fc_feats, att_feats, labels, masks, att_masks, gts, torch.arange(0, len(gts)), 
                sc_flag, struc_flag, bboxes = boxes, bboxes_class = object_classes, rel_iou_threshold=opt.rel_iou_threshold
            )
            loss = model_out['loss'].mean()

            # Backward pass
            loss.backward()
            if opt.grad_clip_value != 0:
                nn.utils.clip_grad_value_(model.parameters(),opt.grad_clip_value)
            optimizer.step()
            train_loss = loss.item()
            if not sc_flag:
                relation_loss = model_out["relation_loss"].item()
            torch.cuda.synchronize()

            # Print 
            total_time = time.time() - start
            if iteration % opt.print_freq == 1:
                if struc_flag:
                    logger.info("iter {} (epoch {}), train_loss = {:.3f}, lm_loss = {:.3f}, struc_loss = {:.3f}, data_time = {:.3f}, total_time = {:.3f}" \
                        .format(iteration, epoch, train_loss, model_out['lm_loss'].mean().item(), model_out['struc_loss'].mean().item(), data_time, total_time))
                if not sc_flag:
                    logger.info("iter {} (epoch {}), train_loss = {:.3f}, word_loss={:.3f}, relation_loss = {:.3f}, data_time = {:.3f}, total_time = {:.3f}" \
                        .format(iteration, epoch, train_loss, train_loss-relation_loss, relation_loss, data_time, total_time))
                else:
                    logger.info("iter {} (epoch {}), avg_reward = {:.3f}, data_time = {:.3f}, total_time = {:.3f}" \
                        .format(iteration, epoch, model_out['reward'].mean(), data_time, total_time))

            if opt.use_tb and iteration % opt.losses_log_every == 0:
                writer.add_scalar('loss/total', train_loss, iteration)
                writer.add_scalar('others/scheduled_sampling_prob', model.ss_prob, iteration)
                writer.add_scalar('others/learning_rate', optimizer.param_groups[0]['lr'], epoch)
                if sc_flag:
                    writer.add_scalar('others/learning_rate1', optimizer.param_groups[1]['lr'], epoch)
                if not sc_flag:
                    writer.add_scalar('loss/relation_loss', relation_loss, iteration)
                    writer.add_scalar('loss/word_loss', train_loss-relation_loss, iteration)
                if sc_flag:
                    writer.add_scalar('others/avg_reward', model_out['reward'].mean(), iteration)
                elif struc_flag:
                    writer.add_scalar('others/lm_loss', model_out['lm_loss'].mean().item(), iteration)
                    writer.add_scalar('others/struc_loss', model_out['struc_loss'].mean().item(), iteration)
                    writer.add_scalar('others/reward', model_out['reward'].mean().item(), iteration)
                    writer.add_scalar('others/reward_var', model_out['reward'].var(1).mean(), iteration)
                loss_history[iteration] = train_loss if not sc_flag else model_out['reward'].mean()
                lr_history[iteration] = optimizer.param_groups[0]['lr']
                ss_prob_history[iteration] = model.ss_prob
            
            iteration += 1
            if True and iteration % opt.save_checkpoint_every == 1: 
                start_time_ = time.time()
                model.eval()
                logger.info("start evluate...")
                metrics, _ = eval_model(model, val_loader, opt.coco_caption_path_test, opt)
                model.train()
                eval_score = metrics['METEOR'] + metrics['CIDEr']
                if not sc_flag:
                    if eval_score > best_eval_score:
                        if opt.training_method == "xe" or opt.training_method == "scst":
                            best_eval_score = eval_score
                        infos={}
                        histories={}
                        infos['iter'] = iteration
                        infos['epoch'] = epoch
                        infos['best_val_score'] = best_eval_score
                        infos['opt'] = opt
                        histories['loss_history'] = loss_history
                        histories['lr_history'] = lr_history
                        histories['ss_prob_history'] = ss_prob_history
                        save_model(model, optimizer, scheduler, opt, epoch, metrics)

                if opt.use_tb: 
                    writer.add_scalar('metrics_i/Bleu-1', metrics['Bleu_1'], iteration)
                    writer.add_scalar('metrics_i/Bleu-2', metrics['Bleu_2'], iteration)
                    writer.add_scalar('metrics_i/Bleu-3', metrics['Bleu_3'], iteration)
                    writer.add_scalar('metrics_i/Bleu-4', metrics['Bleu_4'], iteration)
                    writer.add_scalar('metrics_i/METEOR', metrics['METEOR'], iteration)
                    writer.add_scalar('metrics_i/CIDEr', metrics['CIDEr'], iteration)
            start = time.time()
        
        scheduler.step()
        if opt.use_tb:
            writer.add_scalar('others/epoch_time_use', time.time()-start_time, epoch)


def update_and_save_opt(opt):
    vocab = json.load(open(opt.paratalk_vocab_path, 'r'))
    idx2word = vocab['ix_to_word']
    idx2word["0"] = "<pad><bos><eos>"
    word2idx = {}
    for idx in idx2word:
        word2idx[idx2word[idx]]=idx


    opt.word2idx = word2idx
    opt.idx2word = idx2word
    opt.vocab = idx2word
    opt.vocab_size = len(word2idx)
    if not os.path.exists(opt.config_path):
        os.mkdir(opt.config_path)
    with open(opt.config_path+opt.model_name+'_config.json', 'w') as f:
        args=vars(opt)
        json.dump(args, f, indent=1)
    return opt


def setup_seed(seed):
    # random seed keep
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


start_time = time.time()
current_time = datetime.datetime.now().strftime('%m%d_%H-%M-%S')
opt.model_name = current_time + opt.model_name
opt = update_and_save_opt(opt)

trace = logger.add(opt.config_path+opt.model_name+"_runtime.log",encoding='utf-8')
setup_seed(opt.seed)

train(opt)

time_use = time.time()-start_time
logger.info(f'trainind time using: {datetime.timedelta(seconds=time_use)} hours.')