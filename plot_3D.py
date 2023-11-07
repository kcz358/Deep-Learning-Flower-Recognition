import argparse
import yaml

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from utils import build_from_config
from utils.net_plotter import plot_3D

def parse_arguments():
    parser = argparse.ArgumentParser(description="A script that takes dataset path, model saving path, and tensorboard log path as arguments.")

    parser.add_argument("--config", "-c", required=True, type=str, help="Path to the config file.")

    return parser.parse_args()   

if __name__ == "__main__":
    
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    
    continue_ckpt = config['training'][0].get('continue_ckpt', None)
    if continue_ckpt is not None:
        ckpt = torch.load(continue_ckpt, map_location=device)
        try:
            model, _, _, criterion, train_dataset, valid_dataset, test_dataset = build_from_config(ckpt['training_config'])
        except:
            print("Config not found in the checkpoint, use config given")
            model, _, _, criterion, train_dataset, valid_dataset, test_dataset = build_from_config(config)
        model.load_state_dict(ckpt['state_dict'])
    else:
        print("No config specify in the ckpt")
        model, _, _, criterion, train_dataset, valid_dataset, test_dataset = build_from_config(config)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    model.to(device)
    import time
    start = time.perf_counter()
    plot_3D(-1, 1, -1, 1, model, train_dataloader, criterion, device)
    end = time.perf_counter()
    print(f"Time elapsed : {end - start}")