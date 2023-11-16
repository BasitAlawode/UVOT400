data_register = dict()

# #################################################################################
data_register.update({
    'got10k': [
        ['got10k_train'],  # 9,335
        ['got10k_val'],  # 180
    ],

    'common': [
        [
            'lasot_train',  # 1,120
            'vid_sent_train',  # 6,582
            'got10k_train_vot',  # 8,335
            'trackingnet_train_p0',  # 10,044
            # 'trackingnet_train_p1',  # 10,038
            # ['trackingnet_train_p2', 3000],  # 9,261
            'coco2017_train',  # 10,044
        ],
        ['got10k_val']
    ],

    'common_pretrain': [
        [
            'lasot_train',  # 1,120
            'vid_sent_train',  # 6,582
            'got10k_train_vot',  # 8,335
            'trackingnet_train_p0',  # 10,044
            'trackingnet_train_p1',  # 10,038
            ['trackingnet_train_p2', 3000],  # 9,261
            # ['coco2017_train', 25000],  # 10,044
        ],
        ['got10k_val']
    ],

})
