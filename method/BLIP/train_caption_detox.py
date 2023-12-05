'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from models.blip import blip_decoder
import utils
from data import create_dataset, create_sampler, create_loader
from tqdm import tqdm


def train(model, data_loader, detox_optimizers, clf_optimizers, epoch, device):
    # train
    model.train()  
    
    # freeze blip
    for name, param in model.named_parameters():
        if 'detoxification_mlp' in name or 'toxicity_classification_mlp' in name:
            print(f'Unfreezing {name}')
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Caption Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, caption, toxicity_labels) in tqdm(enumerate(data_loader), total=len(data_loader), desc='Epoch: [{}]'.format(epoch)):
        image = image.to(device)       
    
        # train detoxification_mlp
        for p in model.detox_model.toxicity_classification_mlp.parameters():
            p.requires_grad = False
        loss, toxicity_logits = model(image, caption)
        clf_prob = F.softmax(toxicity_logits, dim=-1)
        detox_loss = torch.mean(clf_prob[:, -1] / 0.5 - torch.sum(torch.square(clf_prob), dim=1))
        total_loss = loss + 0.01 * detox_loss
        detox_optimizers.zero_grad()
        total_loss.backward()
        detox_optimizers.step()    
        for p in model.detox_model.toxicity_classification_mlp.parameters():
            p.requires_grad = True

        # train toxicity_classification_mlp
        for p in model.detox_model.detoxification_mlp.parameters():
            p.requires_grad = False
        loss, toxicity_logits = model(image, caption)
        clf_optimizers.zero_grad()
        clf_loss = F.cross_entropy(toxicity_logits, toxicity_labels.long().to(device))
        clf_loss.backward()
        clf_optimizers.step()
        for p in model.detox_model.detoxification_mlp.parameters():
            p.requires_grad = True

        # write loss
        with open(os.path.join(args.output_dir, 'loss.txt'), 'a') as f:
            f.write(f'Epoch: {str(epoch)} loss: {str(loss.item())}\n')
        with open(os.path.join(args.output_dir, 'detox_loss.txt'), 'a') as f:
            f.write(f'Epoch: {str(epoch)} detox_loss: {str(detox_loss.item())}\n')
        with open(os.path.join(args.output_dir, 'clf_loss.txt'), 'a') as f:
            f.write(f'Epoch: {str(epoch)} clf_loss: {str(clf_loss.item())}\n')
        with open(os.path.join(args.output_dir, 'total_loss.txt'), 'a') as f:
            f.write(f'Epoch: {str(epoch)} total_loss: {str(total_loss.item())}\n')

        if (i * config['batch_size']) % 500 == 0:
            torch.save(model.detox_model.detoxification_mlp.state_dict(), os.path.join(args.output_dir, f'detoxification_mlp_epoch{epoch}_it{i}.pt'))
            torch.save(model.detox_model.toxicity_classification_mlp.state_dict(), os.path.join(args.output_dir, f'toxicity_classification_mlp_epoch{epoch}_it{i}.pt'))
    return {}  


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # evaluate
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10

    result = []
    for image, image_id in metric_logger.log_every(data_loader, print_freq, header): 
        
        image = image.to(device)       
        
        captions = model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], 
                                  min_length=config['min_length'])
        
        for caption, img_id in zip(captions, image_id):
            result.append({"image_id": img_id.item(), "caption": caption})
  
    return result


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating captioning dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('detox', config)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset,val_dataset,test_dataset], [True,False,False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]
    
    train_loader = create_loader([train_dataset],[None],
                                        batch_size=[config['batch_size']],num_workers=[4],
                                        is_trains=[True], collate_fns=[None])[0]
    
    #### Model #### 
    print("Creating model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                           prompt=config['prompt'], detox_model=config['detox_model'])

    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    detox_optimizer = torch.optim.AdamW(params=model.detox_model.detoxification_mlp.parameters(), lr=float(config['init_lr']))
    clf_optimizer = torch.optim.AdamW(params=model.detox_model.toxicity_classification_mlp.parameters(), lr=float(config['init_lr']))
            
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()    
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            # training
            train_stats = train(model, train_loader, detox_optimizer, clf_optimizer, epoch, device) 

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/detox.yaml')
    parser.add_argument('--output_dir', default='outputs/detox')
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)