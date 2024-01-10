from losses.poincare import poincare_loss
from utils_intermediate.stylegan import project_onto_l1_ball
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Optimization():
    def __init__(self, target_model, augmented_models, synthesis, discriminator, transformations, num_ws, config):
        self.synthesis = synthesis
        self.target = target_model
        self.augmentations = augmented_models
        self.discriminator = discriminator
        self.config = config
        self.transformations = transformations
        self.discriminator_weight = self.config.attack['discriminator_loss_weight']
        self.augment = self.config.attack['augmentation_num']
        self.num_ws = num_ws
        self.clip = config.attack['clip']
        self.mid_vector = [None]      # 中间层的向量
        self.intermediate_imgs = {i:[] for i in range(len(config.intermediate['steps']))}
        self.intermediate_w = {i:[] for i in range(len(config.intermediate['steps']))}
    
    def flush_imgs(self):
        self.intermediate_imgs = {i:[] for i in range(len(self.config.intermediate['steps']))}
        self.intermediate_w = {i:[] for i in range(len(self.config.intermediate['steps']))}

    def optimize(self, w_batch, targets_batch):
        print("-------------start intermidiate space search---------------")
        self.mid_vector = [None]

        # 每一轮就是搜一层
        for i, steps in enumerate(self.config.intermediate['steps']):
            torch.cuda.empty_cache()
            start_layer = i + self.config.intermediate['start']
            if i > self.config.intermediate['end']:
                raise Exception('Attemping to go after end layer')
            print(f'w shape {w_batch.shape}')
            imgs, w_batch = self.intermediate(
                w_batch, start_layer, targets_batch, steps, i)
            self.intermediate_imgs[i].append(imgs.detach().cpu())
            self.intermediate_w[i].append(w_batch.detach().cpu())

    # 定义中间层搜索一层的函数
    def intermediate(self, w, start_layer, targets_batch, steps, index):
        print(
            f'---------search Layer {start_layer} in {steps} iterations---------')

        # 中间层搜索前的准备工作
        with torch.no_grad():
            if start_layer == 0:
                var_list = [w.requires_grad_()]
            else:
                self.mid_vector[-1].requires_grad = True
                var_list = [w.requires_grad_()] + [self.mid_vector[-1]]
                print(self.mid_vector[-1].shape)
                prev_mid_vector = torch.ones(
                    self.mid_vector[-1].shape, device=self.mid_vector[-1].device) * self.mid_vector[-1]
            prev_w = torch.ones(w.shape, device=w.device) * w
            self.synthesis.module.set_layer(
                start_layer, self.config.intermediate['end'])

        # 设置优化器
        optimizer = self.config.create_optimizer(params=var_list)
        scheduler = self.config.create_lr_scheduler(optimizer)
        origin_imgs = None

        # 开始中间层搜索
        for i in range(steps):
            imgs = self.synthesize(
                w, layer_in=self.mid_vector[-1], num_ws=self.num_ws)
            origin_imgs = imgs

            # compute discriminator loss
            if self.discriminator_weight > 0:
                discriminator_loss = self.compute_discriminator_loss(
                    imgs)
            else:
                discriminator_loss = torch.tensor(0.0)

            # perform image transformations
            if self.clip:
                imgs = self.clip_images(imgs)
            if self.transformations:
                imgs = self.transformations(imgs)

            # Compute target loss
            outputs = self.target(imgs)
            target_loss = poincare_loss(
                outputs, targets_batch).mean()
            # target_loss = F.nll_loss(outputs, targets_batch, reduction='mean')

            # 计算增强模型的identity loss
            augment_loss = torch.tensor(0.0).cuda()
            if self.augment > 0:
                for index in range(self.augment):
                    augment_outputs = self.augmentations[index](imgs)
                    augment_loss += poincare_loss(
                        augment_outputs, targets_batch).mean()
                    # augment_loss += F.nll_loss(augment_outputs, targets_batch, reduction='mean')

            # combine losses and compute gradients
            optimizer.zero_grad()
            if self.augment > 0:
                gamma_t = 1.0 / (self.augment+1)
                gamma_aug = gamma_t
                loss = (gamma_t*target_loss + gamma_aug*augment_loss) + \
                    discriminator_loss * self.discriminator_weight
            else:
                loss = target_loss + discriminator_loss * self.discriminator_weight
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            # 限制在L1球内，包括中间层和w-space隐向量
            if start_layer != 0 and self.config.intermediate['max_radius_mid_vecor'][index] > 0:
                deviation = project_onto_l1_ball(self.mid_vector[-1] - prev_mid_vector,
                                                 self.config.intermediate['max_radius_mid_vecor'][index])
                var_list[-1].data = (prev_mid_vector + deviation).data
            if self.config.intermediate['max_radius_w'][index] > 0:
                deviation = project_onto_l1_ball(w - prev_w,
                                                 self.config.intermediate['max_radius_w'][index])
                var_list[0].data = (prev_w + deviation).data

            # Log results
            if self.config.log_progress:
                with torch.no_grad():
                    confidence_vector = outputs.softmax(dim=1)
                    confidences = torch.gather(
                        confidence_vector, 1, targets_batch.unsqueeze(1))
                    mean_conf = confidences.mean().detach().cpu()

                if torch.cuda.current_device() == 0 and (i+1) % 10 == 0:
                    print(
                        f'iteration {i}: \t total_loss={loss:.4f} \t target_loss={target_loss:.4f} \t',
                        f'discriminator_loss={discriminator_loss:.4f} \t augment_loss={augment_loss:.4f} \t',
                        f'mean_conf={mean_conf:.4f}'
                    )
                    # if self.config.intermediate['max_radius_w'][index] > 0:
                    #     w_diff = torch.abs(prev_w - var_list[0]).mean(dim=0)
                    #     print(f'w diff mean {w_diff.mean()} sum {w_diff.sum()}')
                    # if start_layer != 0 and self.config.intermediate['max_radius_mid_vecor'][index] > 0:
                    #     mid_diff = torch.abs(prev_mid_vector - var_list[-1]).mean(dim=0)
                    #     print(f'mid diff mean {mid_diff.mean()} sum {mid_diff.sum()}')
                        

        # 搜索完成，为下一层的搜索做准备
        with torch.no_grad():
            self.synthesis.module.set_layer(start_layer, start_layer)
            w_expanded = torch.repeat_interleave(w,
                                                 repeats=self.num_ws,
                                                 dim=1)
            # print(self.mid_vector[-1].shape)
            mid_vector, _ = self.synthesis(
                w_expanded, layer_in=self.mid_vector[-1], noise_mode='const', force_fp32=True)
            # print(mid_vector.shape)
            # exit()
            if self.mid_vector[-1] is not None:
                self.mid_vector[-1] = self.mid_vector[-1].detach().cpu()
            self.mid_vector.append(mid_vector)
            self.synthesis.module.set_layer(
                start_layer, self.config.intermediate['end'])

        return origin_imgs, w.detach()

    # 中间层搜索用的图片合成函数
    def synthesize(self, w, layer_in, num_ws):
        if w.shape[1] == 1:
            # 先扩展w
            w_expanded = torch.repeat_interleave(w,
                                                 repeats=num_ws,
                                                 dim=1)
            imgs = self.synthesis(w_expanded,
                                  layer_in=layer_in,
                                  noise_mode='const',
                                  force_fp32=True)
        else:
            imgs = self.synthesis(w, layer_in=layer_in,
                                  noise_mode='const', force_fp32=True)
        return imgs

    # def synthesize(self, w, num_ws):
    #     if w.shape[1] == 1:
    #         # 先扩展w
    #         w_expanded = torch.repeat_interleave(w,
    #                                              repeats=num_ws,
    #                                              dim=1)
    #         imgs = self.synthesis(w_expanded,
    #                               noise_mode='const',
    #                               force_fp32=True)
    #     else:
    #         imgs = self.synthesis(w, noise_mode='const', force_fp32=True)
    #     return imgs

    def clip_images(self, imgs):
        lower_limit = torch.tensor(-1.0).float().to(imgs.device)
        upper_limit = torch.tensor(1.0).float().to(imgs.device)
        imgs = torch.where(imgs > upper_limit, upper_limit, imgs)
        imgs = torch.where(imgs < lower_limit, lower_limit, imgs)
        return imgs

    def compute_discriminator_loss(self, imgs):
        discriminator_logits = self.discriminator(imgs, None)
        discriminator_loss = nn.functional.softplus(
            -discriminator_logits).mean()
        return discriminator_loss
