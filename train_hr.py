import copy
import argparse

from train_base import *

from module import DeepGuidedFilter, DeepGuidedFilterAdvanced, DeepGuidedFilterConvGF, DeepGuidedFilterGuidedMapConvGF
if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Train Deep Guided Filtering Networks')
    parser.add_argument('--task',  type=str, default='auto_hdr',           help='TASK')
    parser.add_argument('--name',  type=str, default='swf',                 help='NAME')
    parser.add_argument('--model', type=str, default='deep_guided_filter', help='model')
    parser.add_argument('--keep_training', type=bool, default=False, help='use latest model')
    args = parser.parse_args()

    config = copy.deepcopy(default_config)

    config.TASK = args.task
    config.NAME = args.name
    config.N_EPOCH = 150
    config.DATA_SET = 512
    config.keep_training = args.keep_training

    # model
    if args.model == 'deep_guided_filter':
        config.model = DeepGuidedFilter().cuda()
        if config.keep_training == True:
            config.model.init_lr(os.path.join('checkpoints',config.TASK,config.NAME,'snapshots/net_epoch_54.pth'))
    elif args.model == 'deep_guided_filter_advanced':
        config.model = DeepGuidedFilterAdvanced()
    elif args.model == 'deep_conv_guided_filter':
        config.model = DeepGuidedFilterConvGF()
    elif args.model == 'deep_conv_guided_filter_adv':
        config.model = DeepGuidedFilterGuidedMapConvGF()
    else:
        print('Not a valid model!')
        exit(-1)

    def forward(imgs, config):
        x_hr, gt_hr, x_lr = imgs[:3]
        if config.GPU >= 0:
            with torch.cuda.device(config.GPU):
                x_hr, gt_hr, x_lr = x_hr.cuda(), gt_hr.cuda(), x_lr.cuda()

        return config.model(Variable(x_lr), Variable(x_hr)), gt_hr

    config.forward = forward
    config.clip = 0.01

    run(config, keep_vis=True)

    ##########################################
    config.N_START = config.N_EPOCH
    config.N_EPOCH = 30
    config.DATA_SET = 'random'
    config.exceed_limit = lambda size: size[0]*size[1] > 2048**2

    run(config)