import argparse
import yaml

import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from utils import build_from_config

labels_name = [
 'pink primrose',
 'hard-leaved pocket orchid',
 'canterbury bells',
 'sweet pea',
 'english marigold',
 'tiger lily',
 'moon orchid',
 'bird of paradise',
 'monkshood',
 'globe thistle',
 'snapdragon',
 "colt's foot",
 'king protea',
 'spear thistle',
 'yellow iris',
 'globe-flower',
 'purple coneflower',
 'peruvian lily',
 'balloon flower',
 'giant white arum lily',
 'fire lily',
 'pincushion flower',
 'fritillary',
 'red ginger',
 'grape hyacinth',
 'corn poppy',
 'prince of wales feathers',
 'stemless gentian',
 'artichoke',
 'sweet william',
 'carnation',
 'garden phlox',
 'love in the mist',
 'mexican aster',
 'alpine sea holly',
 'ruby-lipped cattleya',
 'cape flower',
 'great masterwort',
 'siam tulip',
 'lenten rose',
 'barbeton daisy',
 'daffodil',
 'sword lily',
 'poinsettia',
 'bolero deep blue',
 'wallflower',
 'marigold',
 'buttercup',
 'oxeye daisy',
 'common dandelion',
 'petunia',
 'wild pansy',
 'primula',
 'sunflower',
 'pelargonium',
 'bishop of llandaff',
 'gaura',
 'geranium',
 'orange dahlia',
 'pink-yellow dahlia?',
 'cautleya spicata',
 'japanese anemone',
 'black-eyed susan',
 'silverbush',
 'californian poppy',
 'osteospermum',
 'spring crocus',
 'bearded iris',
 'windflower',
 'tree poppy',
 'gazania',
 'azalea',
 'water lily',
 'rose',
 'thorn apple',
 'morning glory',
 'passion flower',
 'lotus',
 'toad lily',
 'anthurium',
 'frangipani',
 'clematis',
 'hibiscus',
 'columbine',
 'desert-rose',
 'tree mallow',
 'magnolia',
 'cyclamen ',
 'watercress',
 'canna lily',
 'hippeastrum ',
 'bee balm',
 'ball moss',
 'foxglove',
 'bougainvillea',
 'camellia',
 'mallow',
 'mexican petunia',
 'bromelia',
 'blanket flower',
 'trumpet creeper',
 'blackberry lily',
]

def parse_arguments():
    parser = argparse.ArgumentParser(description="A script that takes dataset path, model saving path, and tensorboard log path as arguments.")

    parser.add_argument("--config", "-c", required=True, type=str, help="Path to the config file.")

    return parser.parse_args()   

if __name__ == "__main__":
    
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    _, _, _, _, train_dataset, valid_dataset, test_dataset = build_from_config(config)
    test_dataset.transform = None
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    model = model.to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    prompt = [f"a photo of a {label.strip()}" for label in labels_name]
    print("Load clip model successfully")
    
    correct = 0
    
    for iteration, (image, label) in enumerate(tqdm(test_dataset)):
        #image = image.to(device)
        label = label.to(device)
        
        inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)
        inputs = {k : v.to(device) for k,v in inputs.items()}
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)
        #import pdb;pdb.set_trace()
        correct += probs.argmax() == label
    
        
    test_accuracies = correct / len(test_dataset)
    print(f"Zero Shot Clip Test accuracies {test_accuracies}")