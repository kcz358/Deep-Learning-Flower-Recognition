from torch.utils.data import default_collate
from torchvision.transforms import v2


def gen_collate(num_class):
    def collate(batch):
        cutmix = v2.CutMix(num_classes=num_class)
        mixup = v2.MixUp(num_classes=num_class)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
        return cutmix_or_mixup(*default_collate(batch))
    return collate