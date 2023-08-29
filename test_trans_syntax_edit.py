import os
import argparse
import json
import time
import numpy as np
import torch
torch.backends.cudnn.enabled  = True
import torch.nn as nn
import torch.nn.functional as F

from configs.config_transformer import cfg, merge_cfg_from_file
from datasets.datasets import create_dataset
from models.NCT import ChangeDetectorDoubleAttDyn, AddSpatialInfo
from models.transformer_decoder_syntax import DynamicSpeaker

from utils.utils import AverageMeter, accuracy, set_mode, load_checkpoint, \
                        decode_sequence, decode_sequence_transformer, coco_gen_format_save
from utils.vis_utils import visualize_att
from tqdm import tqdm

# Load config
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True)
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--snapshot', type=int, required=True)
parser.add_argument('--gpu', type=int, default=-1)
args = parser.parse_args()
merge_cfg_from_file(args.cfg)
# assert cfg.exp_name == os.path.basename(args.cfg).replace('.yaml', '')

# Device configuration
use_cuda = torch.cuda.is_available()
if args.gpu == -1:
    gpu_ids = cfg.gpu_id
else:
    gpu_ids = [args.gpu]
torch.backends.cudnn.enabled  = True
default_gpu_device = gpu_ids[0]
torch.cuda.set_device(default_gpu_device)
device = torch.device("cuda" if use_cuda else "cpu")

# Experiment configuration
exp_dir = cfg.exp_dir
exp_name = cfg.exp_name

output_dir = os.path.join(exp_dir, exp_name)

test_output_dir = os.path.join(output_dir, 'test_output')
if not os.path.exists(test_output_dir):
    os.makedirs(test_output_dir)
caption_output_path = os.path.join(test_output_dir, 'captions', 'test')
if not os.path.exists(caption_output_path):
    os.makedirs(caption_output_path)
att_output_path = os.path.join(test_output_dir, 'attentions', 'test')
if not os.path.exists(att_output_path):
    os.makedirs(att_output_path)

if args.visualize:
    visualize_save_dir = os.path.join(test_output_dir, 'visualizations')
    if not os.path.exists(visualize_save_dir):
        os.makedirs(visualize_save_dir)

snapshot_dir = os.path.join(output_dir, 'snapshots')
snapshot_file = '%s_checkpoint_%d.pt' % (exp_name, args.snapshot)
snapshot_full_path = os.path.join(snapshot_dir, snapshot_file)
checkpoint = load_checkpoint(snapshot_full_path)
change_detector_state = checkpoint['change_detector_state']
speaker_state = checkpoint['speaker_state']


# Load modules
change_detector = ChangeDetectorDoubleAttDyn(cfg)
change_detector.load_state_dict(change_detector_state)
change_detector = change_detector.to(device)

speaker = DynamicSpeaker(cfg)
speaker.load_state_dict(speaker_state)
speaker.to(device)


print(change_detector)
print(speaker)


# Data loading part
train_dataset, train_loader = create_dataset(cfg, 'train')
idx_to_word = train_dataset.get_idx_to_word()
test_dataset, test_loader = create_dataset(cfg, 'test')


set_mode('eval', [change_detector, speaker])
with torch.no_grad():
    test_iter_start_time = time.time()

    result_sents_pos = {}
    result_sents_neg = {}
    for i, batch in tqdm(enumerate(test_loader)):

        d_feats, sc_feats, \
        labels, labels_with_ignore, \
        masks, \
        d_img_paths, sc_img_paths, deps, deps_with_ignore = batch

        batch_size = d_feats.size(0)

        d_feats, sc_feats = d_feats.to(device), sc_feats.to(device)
        # d_feats, sc_feats = \
        #     spatial_info(d_feats), spatial_info(nsc_feats), spatial_info(sc_feats)
        labels, labels_with_ignore, masks = labels.to(device), labels_with_ignore.to(device), masks.to(device)
        # no_chg_labels, no_chg_labels_with_ignore, no_chg_masks = no_chg_labels.to(device), no_chg_labels_with_ignore.to(
        #     device), no_chg_masks.to(device)
        # sc_pos_labels, sc_pos_labels_with_ignore = sc_pos_labels.to(device), sc_pos_labels_with_ignore.to(
        #     device)

        encoder_output_pos,chg_pos_att_bef, chg_pos_att_aft = change_detector(d_feats, sc_feats)
        # encoder_output_neg, chg_neg_att_bef, chg_neg_att_aft = change_detector(d_feats, nsc_feats)

        speaker_output_pos, pos_dynamic_atts = speaker.sample(encoder_output_pos, sample_max=1)

        # speaker_output_neg= speaker.sample(encoder_output_neg, sample_max=1)

        gen_sents_pos = decode_sequence_transformer(idx_to_word, speaker_output_pos[:, 1:])
        # gen_sents_neg = decode_sequence_transformer(idx_to_word, speaker_output_neg[:, 1:])
        chg_pos_att_bef = chg_pos_att_bef.cpu().numpy()
        chg_pos_att_aft = chg_pos_att_aft.cpu().numpy()

        pos_dynamic_atts = pos_dynamic_atts.cpu().numpy()

        dummy = np.ones_like(chg_pos_att_bef)

        for j in range(batch_size):
            gts = decode_sequence_transformer(idx_to_word, labels[j][:, 1:])
            # gts_neg = decode_sequence_transformer(idx_to_word, no_chg_labels[j][:, 1:])
            sent_pos = gen_sents_pos[j]
            # sent_neg = gen_sents_neg[j]
            image_id = d_img_paths[j].split('/')[-1]
            result_sents_pos[image_id] = sent_pos
            # result_sents_neg[image_id + '_n'] = sent_neg
            image_num = image_id.split('.')[0]

            att_bef_path = os.path.join(att_output_path, image_num + '_before')
            att_aft_path = os.path.join(att_output_path, image_num + '_after')
            np.save(att_bef_path, chg_pos_att_bef[j])
            np.save(att_aft_path, chg_pos_att_aft[j])

            if args.visualize:
                visualize_att(d_img_paths[j], sc_img_paths[j],
                              chg_pos_att_bef[j], dummy[j], chg_pos_att_aft[j],
                              pos_dynamic_atts[j], gen_sents_pos[j], gts,
                              -1, -1, visualize_save_dir, 'sc_')


    test_iter_end_time = time.time() - test_iter_start_time
    print('Test took %.4f seconds' % test_iter_end_time)

    result_save_path_pos = os.path.join(caption_output_path, 'sc_results.json')
    result_save_path_neg = os.path.join(caption_output_path, 'nsc_results.json')
    coco_gen_format_save(result_sents_pos, result_save_path_pos)
    coco_gen_format_save(result_sents_neg, result_save_path_neg)

