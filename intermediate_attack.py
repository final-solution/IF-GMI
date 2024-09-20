import math
import os
import random
import traceback
import psutil
from collections import Counter
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
from utils.logger import *
from utils.datasets import (create_target_dataset, get_facescrub_idx_to_class,
                                         get_stanford_dogs_idx_to_class)
from utils.stylegan import create_image, load_generator


if __name__ == '__main__':
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
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4'
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
        
        if config.logging_images:
            log_images(config, result_path, evaluation_model, idx, layer_num, final_imgs, idx_to_class)

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
