from lib.benchmark.benchmark_evaluators import (eval_got10k_test, eval_got10k_val, eval_lasot, eval_trackingnet,
                                                eval_otb)

evaluator_register = dict()

# #################################################################################
evaluator_register.update({
    'got10k_test': eval_got10k_test
})

evaluator_register.update({
    'got10k_val': eval_got10k_val
})

evaluator_register.update({
    'lasot': eval_lasot
})

evaluator_register.update({
    'trackingnet': eval_trackingnet
})

evaluator_register.update({
    'otb': eval_otb
})

evaluator_register.update({
    'choices': list(evaluator_register.keys())
})

if __name__ == '__main__':
    print(evaluator_register['choices'])
    for k, load_fn in evaluator_register.items():
        if load_fn is NotImplemented:
            print(k, 'NotImplemented')
        else:
            load_fn()
