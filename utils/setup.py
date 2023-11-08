from datasets.base_dataset import load_dataset
from models.base_model import load_model
from loss.base_loss import load_loss
from optimizer.base_optimizer import load_optimizer
from scheduler.base_scheduler import load_scheduler
import utils.transformation_constant as transformation_constant

def build_from_config(config):
    model_config = config['model'][0]
    dataset_config = config['dataset'][0]
    optimizer_config = config['optimizer'][0]
    scheduler_config = config['scheduler'][0]
    loss_config = config['loss'][0]
    
    try:
        transformation = getattr(transformation_constant, model_config['transformation'])
        model_config['transformation'] = transformation
    except:
        print("No transformation specified, use default one")
        if 'transformation' in model_config:
            model_config.pop('transformation')
    model = build_model_from_config(model_config)
    optimizer_config['param_groups'] = model.parameters()
    optimizer = build_optimizer_from_config(optimizer_config)
    scheduler_config['optimizer'] = optimizer
    scheduler = build_scheduler_from_config(scheduler_config)
    criterion = build_loss_from_config(loss_config)
    
    dataset_config['transform'] = model.transformation
    train_dataset = build_dataset_from_config(dataset_config, split='train')
    # For valid and test, we don't do aug or mixup
    dataset_config['transform'] = model.default_transformation
    valid_dataset = build_dataset_from_config(dataset_config, split='valid')
    test_dataset = build_dataset_from_config(dataset_config, split='test')
    
    return model, optimizer, scheduler, criterion, train_dataset, valid_dataset, test_dataset
    
    

def build_model_from_config(model_config):
    name = model_config['model_name']
    #model_config.pop('model_name')
    return load_model(name, model_config)

def build_dataset_from_config(dataset_config, split):
    name = dataset_config['dataset_name']
    dataset_config['split'] = split
    #dataset_config.pop('dataset_name')
    return load_dataset(name, dataset_config)

def build_loss_from_config(loss_config):
    name = loss_config['loss_name']
    #loss_config.pop('loss_name')
    return load_loss(name, loss_config)

def build_optimizer_from_config(optimizer_config):
    name = optimizer_config['opt_name']
    #optimizer_config.pop('opt_name')
    return load_optimizer(name, optimizer_config)

def build_scheduler_from_config(scheduler_config):
    name = scheduler_config['scheduler_name']
    #scheduler_config.pop('scheduler_name')
    return load_scheduler(name, scheduler_config)