import math

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from utils.stylegan import create_image

from metrics.accuracy import Accuracy, AccuracyTopK


class ClassificationAccuracy():

    def __init__(self, evaluation_network, layer_num, device='cuda'):
        self.evaluation_network = evaluation_network
        self.device = device
        self.acc_top1 = [Accuracy() for i in range(layer_num)]
        self.acc_top5 = [AccuracyTopK(k=5) for i in range(layer_num)]
        self.predictions = {i: [] for i in range(layer_num)}
        self.correct_confidences = {i: [] for i in range(layer_num)}
        self.total_confidences = {i: [] for i in range(layer_num)}
        self.maximum_confidences = {i: [] for i in range(layer_num)}

    def compute_acc(self,
                    layer,
                    images,
                    targets,
                    config,
                    batch_size=64,
                    resize=299,
                    rtpt=None):
        self.evaluation_network.eval()
        self.evaluation_network.to(self.device)
        # dataset = TensorDataset(w, targets)
        dataset = TensorDataset(images, targets)
        max_iter = math.ceil(len(dataset) / batch_size)

        with torch.no_grad():
            for step, (imgs, target_batch) in enumerate(
                    DataLoader(dataset, batch_size=batch_size, shuffle=False)):
                imgs, target_batch = imgs.to(
                    self.device), target_batch.to(self.device)
                imgs = create_image(imgs,
                                    crop_size=config.attack_center_crop,
                                    resize=resize)
                # imgs = imgs.to(self.device)
                output = self.evaluation_network(imgs)

                self.acc_top1[layer].update(output, target_batch)
                self.acc_top5[layer].update(output, target_batch)

                pred = torch.argmax(output, dim=1)
                self.predictions[layer].append(pred)
                confidences = output.softmax(1)
                target_confidences = torch.gather(confidences, 1,
                                                  target_batch.unsqueeze(1))
                self.correct_confidences[layer].append(
                    target_confidences[pred == target_batch])
                self.total_confidences[layer].append(target_confidences)
                self.maximum_confidences[layer].append(
                    torch.max(confidences, dim=1)[0])

                if rtpt:
                    rtpt.step(
                        subtitle=f'Classification Evaluation step {step} of {max_iter}')

    def get_compute_result(self, layer, targets):
        acc_top1 = self.acc_top1[layer].compute_metric()
        acc_top5 = self.acc_top5[layer].compute_metric()
        correct_confidences = torch.cat(self.correct_confidences[layer], dim=0)
        avg_correct_conf = correct_confidences.mean().cpu().item()
        confidences = torch.cat(self.total_confidences[layer], dim=0).cpu()
        confidences = torch.flatten(confidences)
        maximum_confidences = torch.cat(self.maximum_confidences[layer],
                                        dim=0).cpu().tolist()
        avg_total_conf = torch.cat(self.total_confidences[layer],
                                   dim=0).mean().cpu().item()
        predictions = torch.cat(self.predictions[layer], dim=0).cpu()

        # Compute class-wise precision
        target_list = targets.cpu().tolist()
        precision_list = [['target', 'mean_conf', 'precision']]
        for t in set(target_list):
            mask = torch.where(targets == t, True, False).cpu()
            conf_masked = confidences[mask]
            precision = torch.sum(predictions[mask] == t) / torch.sum(
                targets == t)
            precision_list.append(
                [t, conf_masked.mean().item(),
                    precision.cpu().item()])
        confidences = confidences.tolist()
        predictions = predictions.tolist()

        return acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, \
            confidences, maximum_confidences, precision_list
