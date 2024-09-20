import argparse
import csv
import math
import os
import random
import traceback
import yaml
import psutil
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import time
import torch
import torchvision.transforms as T
from facenet_pytorch import InceptionResnetV1
from rtpt import RTPT
from torch.utils.data import TensorDataset

from attacks.optimize import Optimization
from datasets.custom_subset import ClassSubset
from metrics.classification_acc import ClassificationAccuracy
from metrics.fid_score import FID_Score
from metrics.distance_metrics import DistanceEvaluation
from metrics.prdc import PRDC
from utils.logger import Tee
from utils.logger import Tee
from utils.attack_config_parser import AttackConfigParser
from utils.datasets import (create_target_dataset, get_facescrub_idx_to_class,
                                         get_stanford_dogs_idx_to_class)
from utils.stylegan import create_image, load_generator


def main():
    ####################################
    #        Attack Preparation        #
    ####################################
    
    # Record running time and occupied memory
    start_time = time.perf_counter()
    now_time = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
    init_mem = psutil.virtual_memory().free
    min_mem = init_mem

    # Set devices
    torch.set_num_threads(24)
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # Define and parse attack arguments
    parser = create_parser()
    config, args = parse_arguments(parser)
    layer_num = len(config.intermediate['steps'])

    # Set seeds
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Load idx to class mappings
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

    # Load pre-trained StyleGan2 generator
    G = load_generator(config.stylegan_model)
    num_ws = G.num_ws

    # Load target model and dataset
    target_model, target_name = config.create_target_model()
    target_model_name = target_model.name
    target_dataset = config.get_target_dataset()

    # Distribute models in multiple GPUs
    target_model = torch.nn.DataParallel(target_model, device_ids=gpu_devices)
    target_model.name = target_model_name
    synthesis = torch.nn.DataParallel(G.synthesis, device_ids=gpu_devices)
    synthesis.num_ws = num_ws

    # Load basic attack parameters
    batch_size_single = config.attack['batch_size']
    batch_size = config.attack['batch_size'] * len(gpu_devices)
    targets = config.create_target_vector()
    
    # set transformations for images
    crop_size = config.attack_center_crop
    target_transform = T.Compose([
        T.ToTensor(),
        T.Resize((299, 299), antialias=True),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load evaluation model Incv3
    evaluation_model, eval_name = config.create_evaluation_model()
    evaluation_model = torch.nn.DataParallel(evaluation_model, device_ids=gpu_devices)
    evaluation_model.to(device)
    evaluation_model.eval()
    class_acc_evaluator = ClassificationAccuracy(evaluation_model,
                                                 layer_num=layer_num,
                                                 device=device)

    # Load models for FID and PRDC
    full_training_dataset = create_target_dataset(target_dataset,
                                                  target_transform)
    fid_evaluation = FID_Score(layer_num,
                                  device=device,
                                  crop_size=crop_size,
                                  batch_size=batch_size * 3,
                                  dims=2048,
                                  num_workers=8,
                                  gpu_devices=gpu_devices)

    prdc = PRDC(layer_num,
                   device=device,
                   crop_size=crop_size,
                   batch_size=batch_size * 3,
                   dims=2048,
                   num_workers=8,
                   gpu_devices=gpu_devices)

    # Load Inception-v3 evaluation model and remove final layer
    evaluation_model_dist, _ = config.create_evaluation_model()
    evaluation_model_dist.model.fc = torch.nn.Sequential()
    evaluation_model_dist = torch.nn.DataParallel(evaluation_model_dist,
                                                  device_ids=gpu_devices)
    evaluation_model_dist.to(device)
    evaluation_model_dist.eval()

    inception_dist = DistanceEvaluation(
        layer_num, evaluation_model_dist,
        299,
        config.attack_center_crop,
        target_dataset, config.seed)

    # Load FaceNet model for face recognition
    facenet = InceptionResnetV1(pretrained='vggface2')
    facenet = torch.nn.DataParallel(
        facenet, device_ids=gpu_devices)
    facenet.to(device)
    facenet.eval()

    facenet_dist = DistanceEvaluation(layer_num, facenet, 160,
                                              config.attack_center_crop,
                                              target_dataset, config.seed)

    ####################################
    #              Attack              #
    ####################################

    # Create initial style vectors
    w = create_initial_vectors(config, G, target_model, targets,
                                             device)
    del G

    # Initialize logging
    result_path = config.path
    if config.logging:
        save_config = config.create_saved_config()
        run_id = now_time
        result_path = os.path.join(config.path, run_id)
        Path(f"{result_path}").mkdir(parents=True, exist_ok=True)
        save_dict_to_yaml(
            save_config, f"{result_path}/{config.name}.yaml")
        tee = Tee(f'{result_path}/exp_{now_time}.log', 'w')
        print(f'initial free memory:{(init_mem / (1024**3)):.4f}GB')
        print('path of GAN: ', config.stylegan_model)
        print('target model: ', target_name)
        print('target dataset: ', config.dataset.lower())
        print('evaluation model: ', eval_name)
        init_w_path = f"{result_path}/init_w_{run_id}.pt"
        torch.save(w.detach(), init_w_path)


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
            + 2 * int(math.ceil(config.candidates['num_candidates'] * len(set(targets.cpu().tolist())) / (batch_size * 3))) \
            + 2 * len(set(targets.cpu().tolist()))
        rtpt = RTPT(name_initials='IF-GMI',
                    experiment_name='Model_Inversion_Attack',
                    max_iterations=max_iterations)
        rtpt.start()

    # Create attack transformations
    attack_transformations = config.create_attack_transformations()
    now_mem = psutil.virtual_memory().free
    print(f'free memory after attack preparation:{(now_mem / (1024**3)):.4f}GB')
    min_mem = min(now_mem, min_mem)

    optimization = Optimization(target_model, synthesis, attack_transformations, num_ws, config)

    # Prepare to collect results
    w_optimized_all = {i: [] for i in range(layer_num)}
    final_targets_all = []

    # iteratively compute each target class (reduce memory cost)
    for idx in config.targets:
        num_candidates = config.candidates['num_candidates']
        optimization.flush_imgs()

        now_mem = psutil.virtual_memory().free
        print(f'free memory before attacking {idx}:{(now_mem / (1024**3)):.4f}GB')
        min_mem = min(now_mem, min_mem)

        # Prepare batches for attack
        for i in range(math.ceil(num_candidates / batch_size)):
            start_idx = idx * num_candidates + i * batch_size
            end_idx = min(start_idx + batch_size, (idx+1)*num_candidates)
            w_batch = w[start_idx:end_idx].cuda()
            targets_batch = targets[start_idx:end_idx].cuda()
            print(
                f'\nOptimizing batch {i+1} of {math.ceil(num_candidates / batch_size)} targeting classes {set(targets_batch.cpu().tolist())}.'
            )

            # Run attack iteration
            torch.cuda.empty_cache()
            optimization.optimize(w_batch, targets_batch)

            if rtpt:
                num_batches = math.ceil(w.shape[0] / batch_size)
                rtpt.step(
                    subtitle=f'batch {i+1+idx*math.ceil(num_candidates / batch_size)} of {num_batches}')

        # Concatenate optimized style vectors
        w_optimized = optimization.intermediate_w
        imgs_optimized = optimization.intermediate_imgs
        for k, v in imgs_optimized.items():
            imgs_optimized[k] = torch.cat(v, dim=0)
        for k, v in w_optimized.items():
            w_optimized[k] = torch.cat(v, dim=0)
            w_optimized_all[k].append(w_optimized[k])

        torch.cuda.empty_cache()
        
        # record results
        final_imgs = {}
        target_list = targets[idx*num_candidates:(idx+1)*num_candidates]
        final_targets, final_w, final_imgs = target_list, w_optimized, imgs_optimized
        final_targets_all.append(final_targets)

        now_mem = psutil.virtual_memory().free
        print(f'free memory after attacking {idx}: {(now_mem / (1024**3)):.4f}GB')
        min_mem = min(now_mem, min_mem)

        ####################################
        #         Attack Accuracy          #
        ####################################
        
        # Compute attack accuracy with evaluation model on all generated samples
        try:
            print('calculate acc')
            for layer in range(layer_num):
                class_acc_evaluator.compute_acc(
                    layer,
                    imgs_optimized[layer],
                    target_list,
                    config,
                    batch_size=batch_size * 2,
                    resize=299,
                    rtpt=rtpt)

        except Exception:
            print(traceback.format_exc())

        ####################################
        #    FID Score and GAN Metrics     #
        ####################################
        target_list = target_list.cpu()
        print('calculate fid and prdc')
        try:
            training_dataset = ClassSubset(
                full_training_dataset,
                target_classes=torch.unique(target_list).cpu().tolist())
            for layer in range(layer_num):
                # create datasets
                attack_dataset = TensorDataset(
                    imgs_optimized[layer], target_list)
                attack_dataset.targets = target_list

                # compute FID score
                fid_evaluation.set(training_dataset, attack_dataset)
                fid_evaluation.compute_fid(layer, rtpt)

                # compute precision, recall, density, coverage
                prdc.set(training_dataset, attack_dataset)
                prdc.compute_metric(
                    layer, int(target_list[0]), k=3, rtpt=rtpt)

        except Exception:
            print(traceback.format_exc())

        ####################################
        #         Feature Distance         #
        ####################################
        try:
            print('calculate feature distance')
            for layer in range(layer_num):
                inception_dist.compute_dist(
                    layer,
                    imgs_optimized[layer],
                    target_list,
                    batch_size=batch_size_single * 5,
                    rtpt=rtpt)

            # Compute feature distance only for facial images
            is_face = False
            if target_dataset in [
                    'facescrub', 'celeba_identities'
            ]:
                is_face = True
                for layer in range(layer_num):
                    facenet_dist.compute_dist(
                        layer,
                        imgs_optimized[layer],
                        target_list,
                        batch_size=batch_size_single * 5,
                        rtpt=rtpt)
        except Exception:
            print(traceback.format_exc())

        now_mem = psutil.virtual_memory().free
        print(f'free memory when evaluation {idx}: {(now_mem / (1024**3)):.4f}GB')
        min_mem = min(now_mem, min_mem)

    print(f'maxima occupied memory:{((init_mem-min_mem) / (1024**3)):.4f}GB')

    # aggregate
    for k in range(layer_num):
        w_optimized_all[k] = torch.cat(
            w_optimized_all[k], dim=0)

    ####################################
    #          Finish Logging          #
    ####################################
    if config.logging:
        print('Finishing attack, logging results and creating sample images.')
        optimized_w_path = f"{result_path}/optimized_w_{run_id}.pt"
        torch.save(w_optimized_all, optimized_w_path)

        # save accuracy
        best_layer_result = [0]
        for i in range(layer_num):
            acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, target_confidences, maximum_confidences, precision_list = class_acc_evaluator.get_compute_result(i,
                                                                                                                                                                                targets)
            if acc_top1 > best_layer_result[0]:
                best_layer_result = [acc_top1, acc_top5, predictions, avg_correct_conf,
                                     avg_total_conf, target_confidences, maximum_confidences, precision_list, i]
            print(
                f'Evaluation of {w_optimized_all[0].shape[0]} images on Inception-v3 and layer {i}: \taccuracy@1={acc_top1:4f}',
                f', accuracy@5={acc_top5:4f}, correct_confidence={avg_correct_conf:4f}, total_confidence={avg_total_conf:4f}'
            )
        try:
            write_precision_list(
                f'{result_path}/precision_list_best_{run_id}',
                best_layer_result[-2]
            )
        except:
            pass
        best_layer = best_layer_result[-1]
        print(
            f'Evaluation of {w_optimized_all[0].shape[0]} images on Inception-v3 and best layer is {best_layer}!'
        )

        # save fid and prdc
        for i in range(layer_num):
            fid_score = fid_evaluation.get_fid(i)
            precision, recall, density, coverage = prdc.get_prdc(i)
            print(f'Evaluation metrics of layer {i}:')
            print(
                f'\tFID score computed on {w_optimized_all[0].shape[0]} attack samples and {config.dataset}: {fid_score:.4f}'
            )
            print(
                f' \tPrecision: {precision:.4f}, Recall: {recall:.4f}, Density: {density:.4f}, Coverage: {coverage:.4f}'
            )
        print('\n')
        
        # save feature distance
        mean_distances_lists = []
        for i in range(layer_num):
            avg_dist_inception, mean_distances_list = inception_dist.get_eval_dist(
                i)
            mean_distances_lists.append(mean_distances_list)
            print(f'Mean Distance on Inception-v3 and layer {i}: ',
                  avg_dist_inception.cpu().item())
        try:
            write_precision_list(
                f'{result_path}/distance_inceptionv3_list_best_{run_id}',
                mean_distances_lists[best_layer]
            )
        except:
            pass
        
        if is_face:
            mean_distances_lists = []
            for i in range(layer_num):
                avg_dist_facenet, mean_distances_list = facenet_dist.get_eval_dist(
                    i)
                mean_distances_lists.append(mean_distances_list)
                print(f'Mean Distance on FaceNet and layer {i}: ',
                    avg_dist_facenet.cpu().item())
            try:
                write_precision_list(
                    f'{result_path}/distance_facenet_list_best_{run_id}',
                    mean_distances_lists[best_layer]
                    )
            except:
                pass

        # save time
        end_time = time.perf_counter()
        with open(f'{result_path}/time.txt', 'w') as file:
            file.write(f'running time: {end_time-start_time} seconds')

    if rtpt:
        rtpt.step(subtitle=f'Finishing up')

    exit()
    # Logging of final images
    if config.logging:
        num_classes = 10
        num_imgs = 8

        # Sample final images from the first and last classes
        label_subset = set(
            list(set(targets.tolist()))[:int(num_classes / 2)] +
            list(set(targets.tolist()))[-int(num_classes / 2):])
        log_imgs = []
        log_targets = []
        log_predictions = []
        log_max_confidences = []
        log_target_confidences = []

        # Log images with smallest feature distance
        for label in label_subset:
            mask = torch.where(final_targets == label, True, False)
            imgs_masked = final_imgs[best_layer][mask][:num_imgs]
            imgs = create_image(
                imgs_masked, crop_size=config.attack_center_crop, resize=config.attack_resize)
            log_imgs.append(imgs)
            log_targets += [label for _ in range(num_imgs)]
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


def create_parser():
    parser = argparse.ArgumentParser(
        description='Performing attack')
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
    return w


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
