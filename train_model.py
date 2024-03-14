import argparse
import os
import time
from copy import copy

import torch

from metrics.accuracy import Accuracy
from utils.training_config_parser import TrainingConfigParser

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def main():
    # Define and parse arguments
    parser = argparse.ArgumentParser(
        description='Training a target classifier')
    parser.add_argument('-c',
                        '--config',
                        default=None,
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    args = parser.parse_args()

    if not args.config:
        print(
            "Configuration file is missing. Please check the provided path. Execution is stopped."
        )
        exit()

    # Load json config file
    config = TrainingConfigParser(args.config.strip())
    
    # Build the datasets
    train_set, valid_set, test_set = config.create_datasets()
    import torchvision
    # [341,335,483,35,512,521,290,195,415,137]
    for i in range(len(train_set)):
        _, t = train_set[i]
        img, _ = train_set.dataset[i]
        if t in [218,137]:
            label_dir = f'samplefacescrub/{t}'
            os.makedirs(label_dir, exist_ok=True)
            img.save(os.path.join(label_dir, f'{t}_{i}.png'))
            # torchvision.utils.save_image(img, os.path.join(label_dir, f'{t}_{i}.png'))
    exit()

    # Set seeds and make deterministic
    seed = config.seed
    torch.manual_seed(seed)

    # Create the target model architecture
    target_model = config.create_model()
    if torch.__version__.startswith('2.'):
        print('Compiling model with torch.compile')
        target_model.model = torch.compile(target_model.model)

    
    criterion = torch.nn.CrossEntropyLoss()
    metric = Accuracy

    # Set up optimizer and scheduler
    optimizer = config.create_optimizer(target_model)
    lr_scheduler = config.create_lr_scheduler(optimizer)

    # Create and start RTPT object
    rtpt = config.create_rtpt()
    rtpt.start()

    # Modify the save_path such that subfolders with a timestamp and the name of the run are created
    time_stamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(config.training['save_path'],
                             f"{config.model['architecture']}_{time_stamp}")

    # Start training
    target_model.fit(
        training_data=train_set,
        validation_data=valid_set,
        test_data=test_set,
        criterion=criterion,
        metric=metric,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        rtpt=rtpt,
        config=config,
        batch_size=config.training['batch_size'],
        num_epochs=config.training['num_epochs'],
        dataloader_num_workers=config.training['dataloader_num_workers'],
        enable_logging=config.wandb['enable_logging'],
        wandb_init_args=config.wandb['args'],
        save_base_path=save_path,
        config_file=args.config)


if __name__ == '__main__':
    main()
