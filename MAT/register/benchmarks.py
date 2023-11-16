from functools import partial
from register.paths import path_register as path
from lib.benchmark.benchmark_loaders import (load_got10k, load_lasot, load_trackingnet, load_tnl2k, load_votlt,
                                             load_otb_lang, load_itb, load_votlt22,
                                             load_otb, load_nfs, load_uav)

benchmark_register = dict()

# #################################################################################

benchmark_register.update({
    'itb': partial(load_itb, root=path.benchmark.itb),
    'got10k_train': partial(load_got10k, root=path.benchmark.got10k_train),
    'got10k_test': partial(load_got10k, root=path.benchmark.got10k_test),
    'got10k_val': partial(load_got10k, root=path.benchmark.got10k_val),
    'lasot': partial(load_lasot, root=path.benchmark.lasot),
    'trackingnet': partial(load_trackingnet, root=path.benchmark.trackingnet),
    'tnl2k': partial(load_tnl2k, root=path.benchmark.tnl2k),
    'vot20': NotImplemented,
    'vot2019lt': partial(load_votlt, root=path.benchmark.vot19lt),
    'vot2018lt': partial(load_votlt, root=path.benchmark.vot18lt),
    'vot2022lt': partial(load_votlt22, root=path.benchmark.vot22lt),
    'otb99lang': partial(load_otb_lang, root=path.benchmark.otb99lang),

    'otb': partial(load_otb, root=path.benchmark.otb),
    'nfs': partial(load_nfs, root=path.benchmark.nfs),
    'uav123': partial(load_uav, root=path.benchmark.uav123),
})

benchmark_register.update({
    'choices': list(benchmark_register.keys()),
})

if __name__ == '__main__':
    name = 'itb'
    print(benchmark_register['choices'])
    for k, load_fn in benchmark_register.items():
        if k == name:
            if load_fn is NotImplemented:
                print(k, 'NotImplemented')
            else:
                load_fn()
