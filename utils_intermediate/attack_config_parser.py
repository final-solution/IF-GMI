from copy import copy
from typing import List

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
import yaml
from attacks_intermediate.initial_selection import find_initial_w
from matplotlib.pyplot import fill
from models.classifier import Classifier

import wandb
from utils_intermediate.wandb import load_model


class AttackConfigParser:

    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        if type(config['attack']['targets']) == int:
            i = config['attack']['targets']
            config['attack']['targets'] = list(range(i)) if i > 0 else 0
        self._config = config

    def create_target_model(self):
        if 'wandb_target_run' in self._config:
            model = load_model(self._config['wandb_target_run'])
        elif 'target_model' in self._config:
            config = self._config['target_model']
            model = Classifier(num_classes=config['num_classes'],
                               architecture=config['architecture'])
            model.load_state_dict(torch.load(config['weights']))
            model.wandb_name = None
        else:
            raise RuntimeError('No target model stated in the config file.')

        model.eval()
        self.model = model
        return model, config['architecture']

    def create_augmented_models(self, index):
        if 'wandb_target_run' in self._config:
            model = load_model(self._config['wandb_target_run'])
        elif 'augmented_models' in self._config:
            config = self._config['augmented_models']
            model = Classifier(num_classes=config['num_classes'],
                               architecture=config['architecture'][index])
            model.load_state_dict(torch.load(
                config['weights'][index])['state_dict'])
            model.wandb_name = None
        else:
            raise RuntimeError('No augmented model stated in the config file.')

        model.eval()
        self.model = model
        return model

    def get_target_dataset(self):
        try:
            api = wandb.Api(timeout=60)
            run = api.run(self._config['wandb_target_run'])
            return run.config['Dataset'].strip().lower()
        except:
            return self._config['dataset']

    def create_evaluation_model(self):
        if 'wandb_evaluation_run' in self._config:
            evaluation_model = load_model(self._config['wandb_evaluation_run'])
        elif 'evaluation_model' in self._config:
            config = self._config['evaluation_model']
            evaluation_model = Classifier(num_classes=config['num_classes'],
                                          architecture=config['architecture'])
            evaluation_model.load_state_dict(torch.load(config['weights']))
        else:
            raise RuntimeError(
                'No evaluation model stated in the config file.')

        evaluation_model.eval()
        self.evaluation_model = evaluation_model
        return evaluation_model, config['architecture']

    def create_optimizer(self, params, config=None):
        if config is None:
            config = self._config['attack']['optimizer']

        optimizer_config = self._config['attack']['optimizer']
        for optimizer_type, args in optimizer_config.items():
            if not hasattr(optim, optimizer_type):
                raise Exception(
                    f'{optimizer_type} is no valid optimizer. Please write the type exactly as the PyTorch class'
                )

            optimizer_class = getattr(optim, optimizer_type)
            optimizer = optimizer_class(params, **args)
            break
        return optimizer

    def create_lr_scheduler(self, optimizer):
        if not 'lr_scheduler' in self._config['attack']:
            return None

        scheduler_config = self._config['attack']['lr_scheduler']
        for scheduler_type, args in scheduler_config.items():
            if not hasattr(optim.lr_scheduler, scheduler_type):
                raise Exception(
                    f'{scheduler_type} is no valid learning rate scheduler. Please write the type exactly as the PyTorch class.'
                )

            scheduler_class = getattr(optim.lr_scheduler, scheduler_type)
            scheduler_instance = scheduler_class(optimizer, **args)
            break
        return scheduler_instance

    def create_candidates(self, generator, target_model, targets):
        candidate_config = self._config['candidates']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if 'candidate_file' in candidate_config:
            candidate_file = candidate_config['candidate_file']
            w = torch.load(candidate_file)
            w = w[:self._config['num_candidates']]
            w = w.to(device)
            print(f'Loaded {w.shape[0]} candidates from {candidate_file}.')
            return w

        elif 'candidate_search' in candidate_config:
            search_config = candidate_config['candidate_search']
            w = find_initial_w(generator=generator,
                               target_model=target_model,
                               targets=targets,
                               seed=self.seed,
                               **search_config)
            print(f'Created {w.shape[0]} candidates randomly in w space.')
        else:
            raise Exception(f'No valid candidate initialization stated.')

        w = w.to(device)
        return w

    def create_target_vector(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        attack_config = self._config['attack']
        targets = None
        target_classes = attack_config['targets']
        num_candidates = self._config['candidates']['num_candidates']
        if type(target_classes) is list:
            targets = torch.tensor(target_classes)
            targets = torch.repeat_interleave(targets, num_candidates)
        elif target_classes == 'all':
            targets = torch.tensor([i for i in range(self.model.num_classes)])
            targets = torch.repeat_interleave(targets, num_candidates)
            self.attack['targets'] = list(range(self.model.num_classes))
        elif type(target_classes) == int:
            targets = torch.full(size=(num_candidates, ),
                                 fill_value=target_classes)
        else:
            raise Exception(
                f' Please specify a target class or state a target vector.')

        targets = targets.to(device)
        return targets

    def create_wandb_config(self):
        for _, args in self.optimizer.items():
            lr = args['lr']
            break

        tmp = self.attack['targets']
        self.attack['targets'] = len(tmp)
        config = {
            **self.attack, **self.intermediate,
            **self.candidates,
            'GAN model': self.stylegan_model,
            'target dataset': self.dataset,
            'result path': self.result_path,
            'optimizer': self.optimizer,
            'lr': lr,
            'use_scheduler': 'lr_scheduler' in self._config,
            'target_architecture': self.model.architecture,
            'target_extended': self.model.wandb_name,
            'enable final selection': 'final_selection' in self._config
        }
        if 'lr_scheduler' in self._config:
            config['lr_scheduler'] = self.lr_scheduler
        self.attack['targets'] = tmp

        return config

    def create_attack_transformations(self):
        transformation_list = []
        if 'transformations' in self._config['attack']:
            transformations = self._config['attack']['transformations']
            for transform, args in transformations.items():
                if not hasattr(T, transform):
                    raise Exception(
                        f'{transform} is no valid transformation. Please write the type exactly as the Torchvision class'
                    )
                transformation_class = getattr(T, transform)
                transformation_list.append(transformation_class(**args))
        if len(transformation_list) > 0:
            attack_transformations = T.Compose(transformation_list)
            return attack_transformations

        return None

    @property
    def result_path(self):
        return self._config['result_path']

    @property
    def candidates(self):
        return self._config['candidates']

    @property
    def wandb_target_run(self):
        return self._config['wandb_target_run']

    @property
    def logging(self):
        return self._config['wandb']['enable_logging']

    @property
    def wandb_init_args(self):
        return self._config['wandb']['wandb_init_args']

    @property
    def attack(self):
        return self._config['attack']

    @property
    def targets(self):
        return self._config['attack']['targets']

    @property
    def wandb(self):
        return self._config['wandb']

    @property
    def optimizer(self):
        return self._config['attack']['optimizer']

    @property
    def lr_scheduler(self):
        return self._config['attack']['lr_scheduler']

    @property
    def intermediate(self):
        if 'intermediate' in self._config:
            return self._config['intermediate']
        else:
            return None

    @property
    def final_selection(self):
        if 'final_selection' in self._config:
            return self._config['final_selection']
        else:
            return None

    @property
    def stylegan_model(self):
        return self._config['stylegan_model']

    @property
    def seed(self):
        return self._config['seed']

    @property
    def path(self):
        return self._config['result_path']

    @property
    def cas_evaluation(self):
        return self._config['cas_evaluation']

    @property
    def dataset(self):
        return self._config['dataset']

    @property
    def fid_evaluation(self):
        return self._config['fid_evaluation']

    @property
    def attack_center_crop(self):
        if 'transformations' in self._config['attack']:
            if 'CenterCrop' in self._config['attack']['transformations']:
                return self._config['attack']['transformations']['CenterCrop'][
                    'size']
        else:
            return None

    @property
    def attack_resize(self):
        if 'transformations' in self._config['attack']:
            if 'Resize' in self._config['attack']['transformations']:
                return self._config['attack']['transformations']['Resize'][
                    'size']
        else:
            return None

    @property
    def num_classes(self):
        targets = self._config['attack']['targets']
        if isinstance(targets, int):
            return 1
        else:
            return len(targets)

    @property
    def log_progress(self):
        if 'log_progress' in self._config['attack']:
            return self._config['attack']['log_progress']
        else:
            return True
