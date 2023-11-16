import math


def lr_multi_step(epoch, warmup_epoch, milestones):
    if (epoch + 1) <= warmup_epoch:
        lr = (epoch + 1) / (warmup_epoch + 1)
    else:
        lr = 0.1 ** len([m for m in milestones if m <= epoch])
    return lr


def lr_cosine(epoch, warmup_epoch, total_epochs):
    if (epoch + 1) <= warmup_epoch:
        lr = (epoch + 1) / (warmup_epoch + 1)
    else:
        k = (epoch - warmup_epoch) / (total_epochs - warmup_epoch)
        lr = 0.5 * (math.cos(k * math.pi) + 1)
    return lr