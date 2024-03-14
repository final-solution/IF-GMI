import torch
a = torch.load('/data/yuhongyao/intermediate-MIA/intermediate-MIA/pretrained/resnet18_celeba_1.pt', map_location='cpu')['model_state_dict']

def on_load_checkpoint(checkpoint):
    keys_list = list(checkpoint.keys())
    for key in keys_list:
        if 'orig_mod.' in key:
            deal_key = key.replace('_orig_mod.', '')
            checkpoint[deal_key] = checkpoint[key]
            del checkpoint[key]
            
on_load_checkpoint(a)
torch.save(a, '/data/yuhongyao/intermediate-MIA/intermediate-MIA/pretrained/resnet18_celeba.pt')