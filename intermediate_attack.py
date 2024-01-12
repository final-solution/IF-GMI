import sys
import argparse
import csv
import math
import os
import random
import traceback
import yaml
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import wandb
from facenet_pytorch import InceptionResnetV1
from rtpt import RTPT
from torch.utils.data import TensorDataset

from attacks_intermediate.final_selection import perform_final_selection
from attacks_intermediate.optimize import Optimization
from datasets.custom_subset import ClassSubset
from metrics_intermediate.classification_acc import ClassificationAccuracy
from metrics_intermediate.fid_score import FID_Score
from metrics_intermediate.prcd import PRCD
from utils_intermediate.attack_config_parser import AttackConfigParser
from utils_intermediate.datasets import (create_target_dataset, get_facescrub_idx_to_class,
                                         get_stanford_dogs_idx_to_class)
from utils_intermediate.stylegan import create_image, load_discrimator, load_generator
from utils_intermediate.wandb import *

os.environ["WANDB_MODE"] = "offline"


class Tee(object):
    """A workaround method to print in console and write to log file
    """

    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        if not '...' in data:
            self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


def main():
    ####################################
    #        Attack Preparation        #
    ####################################

    import time
    start_time = time.perf_counter()

    now_time = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))

    # Set devices: 设备驱动
    torch.set_num_threads(24)
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # Define and parse attack arguments: 参数管理
    parser = create_parser()
    config, args = parse_arguments(parser)
    layer_num = len(config.intermediate['steps'])

    # Set seeds: 随机种子
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Load idx to class mappings: 加载目标类别
    idx_to_class = None
    if config.dataset.lower() == 'facescrub':
        idx_to_class = get_facescrub_idx_to_class()
    elif config.dataset.lower() == 'stanford_dogs':
        idx_to_class = get_stanford_dogs_idx_to_class()
    else:

        class KeyDict(dict):

            def __missing__(self, key):
                return key

        idx_to_class = KeyDict()

    # Load pre-trained StyleGan2 components: 加载预训练GAN
    G = load_generator(config.stylegan_model)
    # D = load_discrimator(config.stylegan_model)
    num_ws = G.num_ws

    # Load target model and set dataset: 加载目标模型与数据集
    target_model = config.create_target_model()
    target_model_name = target_model.name
    target_dataset = config.get_target_dataset()

    # Load augmented models: 加载增强模型，用于克服过拟合
    aug_num = config.attack['augmentation_num']
    augmented_models = []
    augmented_models_name = []
    for i in range(aug_num):
        augmented_model = config.create_augmented_models(i)
        augmented_model_name = augmented_model.name
        augmented_models.append(augmented_model)
        augmented_models_name.append(augmented_model_name)

    # Distribute models: 设置为分布式部署在多个GPU上
    target_model = torch.nn.DataParallel(target_model, device_ids=gpu_devices)
    target_model.name = target_model_name
    for i in range(aug_num):
        augmented_models[i] = torch.nn.DataParallel(
            augmented_models[i], device_ids=gpu_devices)
        augmented_models[i].name = augmented_models_name[i]
    synthesis = torch.nn.DataParallel(G.synthesis, device_ids=gpu_devices)
    synthesis.num_ws = num_ws
    # discriminator = torch.nn.DataParallel(D, device_ids=gpu_devices)
    discriminator = None

    # Load basic attack parameters: 加载基础攻击参数
    batch_size_single = config.attack['batch_size']
    batch_size = config.attack['batch_size'] * len(gpu_devices)
    targets = config.create_target_vector()

    # 加载评价模型Incv3
    evaluation_model = config.create_evaluation_model()
    evaluation_model = torch.nn.DataParallel(evaluation_model)
    evaluation_model.to(device)
    evaluation_model.eval()
    class_acc_evaluator = ClassificationAccuracy(evaluation_model,
                                                 layer_num=layer_num,
                                                 device=device)
    class_acc_evaluator_selected = ClassificationAccuracy(evaluation_model,
                                                          layer_num=layer_num,
                                                          device=device)

    # set transformations: 设置图片变换方式
    crop_size = config.attack_center_crop
    target_transform = T.Compose([
        T.ToTensor(),
        T.Resize((299, 299), antialias=True),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 加载FID、PRCD计算所需模型
    full_training_dataset = create_target_dataset(target_dataset,
                                                  target_transform)
    fid_evaluation = FID_Score(layer_num,
                               device=device,
                               crop_size=crop_size,
                               batch_size=batch_size * 3,
                               dims=2048,
                               num_workers=8,
                               gpu_devices=gpu_devices)
    prcd = PRCD(layer_num,
                device=device,
                crop_size=crop_size,
                generator=synthesis,
                batch_size=batch_size * 3,
                dims=2048,
                num_workers=8,
                gpu_devices=gpu_devices)

    # Create initial style vectors: 执行初始筛选
    w, w_init, x, V = create_initial_vectors(config, G, target_model, targets,
                                             device)
    del G

    # Initialize wandb logging: 使用wandb的日志记录操作

    result_path = config.path
    if config.logging:
        optimizer = config.create_optimizer(params=[w])
        wandb_run, save_config = init_wandb_logging(optimizer, target_model_name, config,
                                                    args)
        run_id = wandb_run.id
        result_path = os.path.join(config.path, run_id)
        Path(f"{result_path}").mkdir(parents=True, exist_ok=True)
        save_dict_to_yaml(
            save_config, f"{result_path}/{config.wandb['wandb_init_args']['name']}.yaml")
        tee = Tee(f'{result_path}/inter_{now_time}.log', 'w')

    # Print attack configuration: 打印攻击参数设置
    print(
        f'Start attack against {target_model.name} optimizing w with shape {list(w.shape)} ',
        f'and targets {dict(Counter(targets.cpu().numpy()))}.')
    print(f'\nAttack parameters')
    for key in config.attack:
        print(f'\t{key}: {config.attack[key]}')
    print(
        f'Performing attack on {len(gpu_devices)} gpus and an effective batch size of {batch_size} images.'
    )

    # Initialize RTPT
    rtpt = None
    if args.rtpt:
        max_iterations = math.ceil(w.shape[0] / batch_size) \
            + int(math.ceil(w.shape[0] / (batch_size * 3))) \
            + 2 * int(math.ceil(config.final_selection['samples_per_target'] * len(set(targets.cpu().tolist())) / (batch_size * 3))) \
            + 2 * len(set(targets.cpu().tolist()))
        rtpt = RTPT(name_initials='LS',
                    experiment_name='Model_Inversion',
                    max_iterations=max_iterations)
        rtpt.start()

    # Log initial vectors: 记录选择出来的初始隐向量
    if config.logging:
        init_w_path = f"{result_path}/init_w_{run_id}.pt"
        torch.save(w.detach(), init_w_path)
        wandb.save(init_w_path, policy='now')

    # Create attack transformations: 用到的数据增强方式
    attack_transformations = config.create_attack_transformations()

    ####################################
    #         Attack Iteration         #
    ####################################
    optimization = Optimization(target_model, augmented_models, synthesis, discriminator,
                                attack_transformations, num_ws, config)

    # Collect results: 收集结果
    w_optimized_unselected_all = {i: [] for i in range(layer_num)}
    final_w_all = {i: [] for i in range(layer_num)}
    final_targets_all = []

    # 每个class分别迭代计算，减少内存消耗
    for idx, target in enumerate(config.targets):
        num_candidates = config.candidates['num_candidates']
        optimization.flush_imgs()

        # Prepare batches for attack：准备攻击的batch
        for i in range(math.ceil(num_candidates / batch_size)):
            start_idx = idx * num_candidates + i * batch_size
            end_idx = min(start_idx + batch_size, (idx+1)*num_candidates)
            w_batch = w[start_idx:end_idx].cuda()
            targets_batch = targets[start_idx:end_idx].cuda()
            print(
                f'\nOptimizing batch {i+1} of {math.ceil(num_candidates / batch_size)} targeting classes {set(targets_batch.cpu().tolist())}.'
            )

            # Run attack iteration: 执行攻击
            torch.cuda.empty_cache()
            optimization.optimize(w_batch, targets_batch)

            if rtpt:
                num_batches = math.ceil(w.shape[0] / batch_size)
                rtpt.step(
                    subtitle=f'batch {i+1+idx*math.ceil(num_candidates / batch_size)} of {num_batches}')

        # Concatenate optimized style vectors: 将没有最终筛选的优化结果拼在一起
        w_optimized_unselected = optimization.intermediate_w
        imgs_optimized_unselected = optimization.intermediate_imgs
        for k, v in imgs_optimized_unselected.items():
            imgs_optimized_unselected[k] = torch.cat(v, dim=0)
        for k, v in w_optimized_unselected.items():
            w_optimized_unselected[k] = torch.cat(v, dim=0)
            w_optimized_unselected_all[k].append(torch.cat(v, dim=0))

        torch.cuda.empty_cache()
        # del discriminator, synthesis
        # synthesis = None

        ####################################
        #          Filter Results          #
        ####################################

        # Filter results: 执行最终阶段筛选
        final_imgs = {}
        target_list = targets[idx*num_candidates:(idx+1)*num_candidates]
        if config.final_selection:
            print(
                f'\nSelect final set of max. {config.final_selection["samples_per_target"]} ',
                f'images per target using {config.final_selection["approach"]} approach.'
            )
            for j in range(layer_num):
                final_w, final_targets, final_layer_imgs = perform_final_selection(
                    w_optimized_unselected[j],
                    imgs_optimized_unselected[j],
                    config,
                    target_list,
                    target_model,
                    device=device,
                    batch_size=batch_size * 10,
                    **config.final_selection,
                    rtpt=rtpt)
                final_imgs[j] = final_layer_imgs
                final_w_all[j].append(final_w)
            print(f'Selected a total of {final_w.shape[0]} final images ',
                  f'of target classes {set(final_targets.cpu().tolist())}.')
        else:
            final_targets, final_w, final_imgs = target_list, w_optimized_unselected, imgs_optimized_unselected
        final_targets_all.append(final_targets)
        # del target_model

        ####################################
        #         Attack Accuracy          #
        ####################################

        # 计算acc指标
        # Compute attack accuracy with evaluation model on all generated samples
        try:
            # 计算准确率acc
            for layer in range(layer_num):
                class_acc_evaluator.compute_acc(
                    layer,
                    imgs_optimized_unselected[layer],
                    target_list,
                    config,
                    batch_size=batch_size * 2,
                    resize=299,
                    rtpt=rtpt)

            # Compute attack accuracy on filtered samples: 在筛选过的样本中计算acc
            if config.final_selection:
                for layer in range(layer_num):
                    class_acc_evaluator_selected.compute_acc(
                        layer,
                        final_imgs[layer],
                        final_targets,
                        config,
                        batch_size=batch_size * 2,
                        resize=299,
                        rtpt=rtpt)
            # del evaluation_model

        except Exception:
            print(traceback.format_exc())

        ####################################
        #    FID Score and GAN Metrics     #
        ####################################
        try:
            training_dataset = ClassSubset(
                full_training_dataset,
                target_classes=torch.unique(final_targets).cpu().tolist())
            for layer in range(layer_num):
                # create datasets: 创建待使用的数据集
                attack_dataset = TensorDataset(
                    final_imgs[layer], final_targets)
                attack_dataset.targets = final_targets

                # compute FID score: 计算fid指标（暂时不考虑计算这个指标）
                # fid_evaluation.set(training_dataset, attack_dataset)
                # fid_evaluation.compute_fid(layer, rtpt)

                # compute precision, recall, density, coverage: 计算指标
                prcd.set(training_dataset, attack_dataset)
                prcd.compute_metric(
                    layer, num_classes=config.num_classes, k=3, rtpt=rtpt)

        except Exception:
            print(traceback.format_exc())

    # 处理最终结果
    for k in range(layer_num):
        w_optimized_unselected_all[k] = torch.cat(
            w_optimized_unselected_all[k], dim=0)
        final_w_all[k] = torch.cat(final_w_all[k], dim=0)

    # Log optimized vectors: 记录优化得到的隐向量
    if config.logging:
        optimized_w_path = f"{result_path}/optimized_w_{run_id}.pt"
        torch.save(w_optimized_unselected_all, optimized_w_path)
        wandb.save(optimized_w_path, policy='now')

        # Log selected vectors: 记录选择结果
        optimized_w_path_selected = f"{result_path}/optimized_w_selected_{run_id}.pt"
        torch.save(final_w_all, optimized_w_path_selected)
        wandb.save(optimized_w_path_selected, policy='now')
        wandb.config.update({'w_path': optimized_w_path})

        # 记录acc相关结果
        for i in range(layer_num):
            best_layer_result = [0]
            acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, target_confidences, maximum_confidences, precision_list = class_acc_evaluator.get_compute_result(i,
                                                                                                                                                                                targets)
            if acc_top1 > best_layer_result[0]:
                best_layer_result = [acc_top1, acc_top5, predictions, avg_correct_conf,
                                     avg_total_conf, target_confidences, maximum_confidences, precision_list, i]
            print(
                f'\nUnfiltered Evaluation of {final_w_all[0].shape[0]} images on Inception-v3 and layer {i}: \taccuracy@1={acc_top1:4f}',
                f', accuracy@5={acc_top5:4f}, correct_confidence={avg_correct_conf:4f}, total_confidence={avg_total_conf:4f}'
            )
        try:
            filename_precision = write_precision_list(
                f'{result_path}/precision_list_unfiltered_{run_id}',
                best_layer_result[-2]
            )
            wandb.save(filename_precision, policy='now')
        except:
            pass
        best_layer = best_layer_result[-1]
        print(
            f'\nUnfiltered Evaluation of {final_w_all[0].shape[0]} images on Inception-v3 and best layer is {best_layer}!'
        )

        if config.final_selection:
            final_targets_all = torch.cat(final_targets_all, dim=0)
            for i in range(layer_num):
                best_layer_result = [0]
                acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, target_confidences, maximum_confidences, precision_list = class_acc_evaluator_selected.get_compute_result(i,
                                                                                                                                                                                             final_targets_all)
                if acc_top1 > best_layer_result[0]:
                    best_layer_result = [acc_top1, acc_top5, predictions, avg_correct_conf,
                                         avg_total_conf, target_confidences, maximum_confidences, precision_list, i]
                print(
                    f'\nFiltered Evaluation of {final_w_all[0].shape[0]} images on Inception-v3 and layer {i}: \taccuracy@1={acc_top1:4f}',
                    f', accuracy@5={acc_top5:4f}, correct_confidence={avg_correct_conf:4f}, total_confidence={avg_total_conf:4f}'
                )
            try:
                filename_precision = write_precision_list(
                    f'{result_path}/precision_list_filtered_{run_id}',
                    best_layer_result[-2]
                )
                wandb.save(filename_precision, policy='now')
            except:
                pass
            best_layer = best_layer_result[-1]
            print(
                f'\nFiltered Evaluation of {final_w_all[0].shape[0]} images on Inception-v3 and best layer is {best_layer}!'
            )
        
        # 记录fid和prcd相关结果
        for i in range(layer_num):
            fid_score = fid_evaluation.get_fid(i)
            precision, recall, density, coverage = prcd.get_prcd(i)
            print(f'Metrics of layer {i}:')
            print(
                f'\tFID score computed on {final_w_all[0].shape[0]} attack samples and {config.dataset}: {fid_score:.4f}'
            )
            print(
                f' \tPrecision: {precision:.4f}, Recall: {recall:.4f}, Density: {density:.4f}, Coverage: {coverage:.4f}'
            )

    exit()

    ####################################
    #         Feature Distance         #
    ####################################
    avg_dist_inception_list = []
    avg_dist_facenet_list = []
    try:
        # Load Inception-v3 evaluation model and remove final layer: 加载评估模型
        evaluation_model_dist = config.create_evaluation_model()
        evaluation_model_dist.model.fc = torch.nn.Sequential()
        evaluation_model_dist = torch.nn.DataParallel(evaluation_model_dist,
                                                      device_ids=gpu_devices)
        evaluation_model_dist.to(device)
        evaluation_model_dist.eval()

        # Compute average feature distance on Inception-v3: 计算评估模型上的平均特征距离
        evaluate_inception = DistanceEvaluation(evaluation_model_dist,
                                                synthesis, 299,
                                                config.attack_center_crop,
                                                target_dataset, config.seed)
        mean_distances_lists = []
        for i in range(layer_num):
            avg_dist_inception, mean_distances_list = evaluate_inception.compute_dist(
                final_w,
                final_imgs[i],
                final_targets,
                batch_size=batch_size_single * 5,
                rtpt=rtpt)
            avg_dist_inception_list.append(avg_dist_inception)
            mean_distances_lists.append(mean_distances_list)
            print(f'Mean Distance on Inception-v3 and layer {i}: ',
                  avg_dist_inception.cpu().item())

        # 记录结果
        if config.logging:
            try:
                filename_distance = write_precision_list(
                    f'{result_path}/distance_inceptionv3_list_filtered_{run_id}',
                    mean_distances_lists[best_layer])
                wandb.save(filename_distance, policy='now')
            except:
                pass

        # Compute feature distance only for facial images
        if target_dataset in [
                'facescrub', 'celeba_identities', 'celeba_attributes'
        ]:
            # Load FaceNet model for face recognition: 加载面部识别用的模型
            facenet = InceptionResnetV1(pretrained='vggface2')
            facenet = torch.nn.DataParallel(facenet, device_ids=gpu_devices)
            facenet.to(device)
            facenet.eval()

            # Compute average feature distance on facenet: 计算面部识别模型上的平均特征距离
            evaluater_facenet = DistanceEvaluation(facenet, synthesis, 160,
                                                   config.attack_center_crop,
                                                   target_dataset, config.seed)

            mean_distances_lists = []
            for i in range(layer_num):
                avg_dist_facenet, mean_distances_list = evaluater_facenet.compute_dist(
                    final_w,
                    final_imgs[i],
                    final_targets,
                    batch_size=batch_size_single * 8,
                    rtpt=rtpt)
                avg_dist_facenet_list.append(avg_dist_facenet)
                mean_distances_lists.append(mean_distances_list)
                print(
                    f'Mean Distance on FaceNet and layer {i}', avg_dist_facenet.cpu().item())

            # 记录结果
            if config.logging:
                filename_distance = write_precision_list(
                    f'{result_path}/distance_facenet_list_filtered_{run_id}',
                    mean_distances_lists[best_layer])
                wandb.save(filename_distance, policy='now')
    except Exception:
        print(traceback.format_exc())

    ####################################
    #          Finish Logging          #
    ####################################

    if rtpt:
        rtpt.step(subtitle=f'Finishing up')

    # Logging of final results: 记录最终结果
    if config.logging:
        print('Finishing attack, logging results and creating sample images.')
        num_classes = 10
        num_imgs = 8

        # 从第一个和最后一个类别中采样最终图片
        # Sample final images from the first and last classes
        label_subset = set(
            list(set(targets.tolist()))[:int(num_classes / 2)] +
            list(set(targets.tolist()))[-int(num_classes / 2):])
        log_imgs = []
        log_targets = []
        log_predictions = []
        log_max_confidences = []
        log_target_confidences = []

        # 记录具有最小特征距离的图片
        # Log images with smallest feature distance
        for label in label_subset:
            mask = torch.where(final_targets == label, True, False)
            # w_masked = final_w[mask][:num_imgs]
            # imgs = create_image(w_masked,
            #                     synthesis,
            #                     crop_size=config.attack_center_crop,
            #                     resize=config.attack_resize)
            imgs_masked = final_imgs[best_layer][mask][:num_imgs]
            imgs = create_image(
                imgs_masked, crop_size=config.attack_center_crop, resize=config.attack_resize)
            log_imgs.append(imgs)
            log_targets += [label for i in range(num_imgs)]
            log_predictions.append(torch.tensor(
                best_layer_result[2])[mask][:num_imgs])
            log_max_confidences.append(
                torch.tensor(best_layer_result[-3])[mask][:num_imgs])
            log_target_confidences.append(
                torch.tensor(best_layer_result[-4])[mask][:num_imgs])

        log_imgs = torch.cat(log_imgs, dim=0)
        log_predictions = torch.cat(log_predictions, dim=0)
        log_max_confidences = torch.cat(log_max_confidences, dim=0)
        log_target_confidences = torch.cat(log_target_confidences, dim=0)

        # 记录最终的图片结果
        log_final_images(log_imgs, log_predictions, log_max_confidences,
                         log_target_confidences, idx_to_class)

        # Find closest training samples to final results: 为最终结果匹配最近的训练样本
        log_nearest_neighbors(log_imgs,
                              log_targets,
                              evaluation_model_dist,
                              'InceptionV3',
                              target_dataset,
                              img_size=299,
                              seed=config.seed)

        # Use FaceNet only for facial images: 仅对面部图片使用FaceNet模型
        facenet = InceptionResnetV1(pretrained='vggface2')
        facenet = torch.nn.DataParallel(facenet, device_ids=gpu_devices)
        facenet.to(device)
        facenet.eval()
        if target_dataset in [
                'facescrub', 'celeba_identities', 'celeba_attributes'
        ]:
            log_nearest_neighbors(log_imgs,
                                  log_targets,
                                  facenet,
                                  'FaceNet',
                                  target_dataset,
                                  img_size=160,
                                  seed=config.seed)
        # 最终记录
        # Final logging
        final_wandb_logging(avg_correct_conf, avg_total_conf, acc_top1,
                            acc_top5, avg_dist_facenet, avg_dist_inception,
                            fid_score, precision, recall, density, coverage)

    end_time = time.perf_counter()
    with open('time.txt', 'w') as file:
        file.write(f'运行时间：{end_time-start_time}秒')


def create_parser():
    parser = argparse.ArgumentParser(
        description='Performing model inversion attack')
    parser.add_argument('-c',
                        '--config',
                        default=None,
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    parser.add_argument('--no_rtpt',
                        action='store_false',
                        dest="rtpt",
                        help='Disable RTPT')
    return parser


def parse_arguments(parser):
    args = parser.parse_args()

    if not args.config:
        print(
            "Configuration file is missing. Please check the provided path. Execution is stopped."
        )
        exit()

    # Load attack config
    config = AttackConfigParser(args.config)

    return config, args


def create_initial_vectors(config, G, target_model, targets, device):
    with torch.no_grad():
        w = config.create_candidates(G, target_model, targets).cpu()
        if config.attack['single_w']:
            w = w[:, 0].unsqueeze(1)
        w_init = deepcopy(w)
        x = None
        V = None
    return w, w_init, x, V


def write_precision_list(filename, precision_list):
    filename = f"{filename}.csv"
    with open(filename, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        for row in precision_list:
            wr.writerow(row)
    return filename


def log_attack_progress(loss,
                        target_loss,
                        discriminator_loss,
                        discriminator_weight,
                        mean_conf,
                        lr,
                        imgs=None,
                        captions=None):
    if imgs is not None:
        imgs = [
            wandb.Image(img.permute(1, 2, 0).numpy(), caption=caption)
            for img, caption in zip(imgs, captions)
        ]
        wandb.log({
            'total_loss': loss,
            'target_loss': target_loss,
            'discriminator_loss': discriminator_loss,
            'discriminator_weight': discriminator_weight,
            'mean_conf': mean_conf,
            'learning_rate': lr,
            'samples': imgs
        })
    else:
        wandb.log({
            'total_loss': loss,
            'target_loss': target_loss,
            'discriminator_loss': discriminator_loss,
            'discriminator_weight': discriminator_weight,
            'mean_conf': mean_conf,
            'learning_rate': lr
        })


def save_dict_to_yaml(dict_value: dict, save_path: str):
    with open(save_path, 'w') as file:
        file.write(yaml.dump(dict_value, allow_unicode=True))


def init_wandb_logging(optimizer, target_model_name, config, args):
    lr = optimizer.param_groups[0]['lr']
    optimizer_name = type(optimizer).__name__
    if not 'name' in config.wandb['wandb_init_args']:
        config.wandb['wandb_init_args'][
            'name'] = f'{optimizer_name}_{lr}_{target_model_name}'
    wandb_config = config.create_wandb_config()
    run = wandb.init(config=wandb_config, **config.wandb['wandb_init_args'])
    wandb.save(args.config, policy='now')
    return run, wandb_config


def intermediate_wandb_logging(optimizer, targets, confidences, loss,
                               target_loss, discriminator_loss,
                               discriminator_weight, mean_conf, imgs, idx2cls):
    lr = optimizer.param_groups[0]['lr']
    target_classes = [idx2cls[idx.item()] for idx in targets.cpu()]
    conf_list = [conf.item() for conf in confidences]
    if imgs is not None:
        img_captions = [
            f'{target} ({conf:.4f})'
            for target, conf in zip(target_classes, conf_list)
        ]
        log_attack_progress(loss,
                            target_loss,
                            discriminator_loss,
                            discriminator_weight,
                            mean_conf,
                            lr,
                            imgs,
                            captions=img_captions)
    else:
        log_attack_progress(loss, target_loss, discriminator_loss,
                            discriminator_weight, mean_conf, lr)


def log_nearest_neighbors(imgs, targets, eval_model, model_name, dataset,
                          img_size, seed):
    # Find closest training samples to final results
    evaluater = DistanceEvaluation(eval_model, None, img_size, None, dataset,
                                   seed)
    closest_samples, distances = evaluater.find_closest_training_sample(
        imgs, targets)
    closest_samples = [
        wandb.Image(img.permute(1, 2, 0).cpu().numpy(),
                    caption=f'distance={d:.4f}')
        for img, d in zip(closest_samples, distances)
    ]
    wandb.log({f'closest_samples {model_name}': closest_samples})


def log_final_images(imgs, predictions, max_confidences, target_confidences,
                     idx2cls):
    wand_imgs = [
        wandb.Image(
            img.permute(1, 2, 0).numpy(),
            caption=f'pred={idx2cls[pred.item()]} ({max_conf:.2f}), target_conf={target_conf:.2f}'
        ) for img, pred, max_conf, target_conf in zip(
            imgs.cpu(), predictions, max_confidences, target_confidences)
    ]
    wandb.log({'final_images': wand_imgs})


def final_wandb_logging(avg_correct_conf, avg_total_conf, acc_top1, acc_top5,
                        avg_dist_facenet, avg_dist_eval, fid_score, precision,
                        recall, density, coverage):
    wandb.save('attacks/gradient_based.py', policy='now')
    wandb.run.summary['correct_avg_conf'] = avg_correct_conf
    wandb.run.summary['total_avg_conf'] = avg_total_conf
    wandb.run.summary['evaluation_acc@1'] = acc_top1
    wandb.run.summary['evaluation_acc@5'] = acc_top5
    wandb.run.summary['avg_dist_facenet'] = avg_dist_facenet
    wandb.run.summary['avg_dist_evaluation'] = avg_dist_eval
    wandb.run.summary['fid_score'] = fid_score
    wandb.run.summary['precision'] = precision
    wandb.run.summary['recall'] = recall
    wandb.run.summary['density'] = density
    wandb.run.summary['coverage'] = coverage

    wandb.finish()


if __name__ == '__main__':
    main()
