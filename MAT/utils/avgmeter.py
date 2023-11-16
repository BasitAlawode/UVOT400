class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, wind_size):
        self.reset()
        self.wind_size = wind_size

    def reset(self):
        self.val = []
        self.avg = 0
        self.sum = 0

    def update(self, val):
        self.val.append(val)

        if self.wind_size is not None:
            if len(self.val) > self.wind_size:
                self.val.pop(0)

        self.sum = sum(self.val)
        self.avg = self.sum / len(self.val)


def build_meter(wind_size=None):
    meter = {
        'loss': AverageMeter(wind_size=wind_size),
        'giou': AverageMeter(wind_size=wind_size),
        'l1': AverageMeter(wind_size=wind_size),
        'miou': AverageMeter(wind_size=wind_size),
        'speed': AverageMeter(wind_size=wind_size),
    }

    return meter
