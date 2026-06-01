import sys
sys.path.append('/public/home/tanyh_25/project/Mamba/SSDFNet')

import argparse
import os
import time

import numpy as np

from buildingextraction.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from buildingextraction.datasets.make_data_loader import BuildingDatset, make_data_loader
from buildingextraction.utils_func.metrics import Evaluator
from buildingextraction.models.SSDFNet import SSDFNet
import imageio
import numpy as np

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


ori_label_value_dict = {
    'background': (0, 0, 0),
    'building': (255, 255, 255)
}

target_label_value_dict = {
    'background': 0,
    'building': 1
}

def format_big_number(n):
    if n >= 1e12: return f"{n/1e12:.2f}T"
    if n >= 1e9:  return f"{n/1e9:.2f}G"
    if n >= 1e6:  return f"{n/1e6:.2f}M"
    if n >= 1e3:  return f"{n/1e3:.2f}K"
    return str(n)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.evaluator_loc = Evaluator(num_class=2)

        self.deep_model = SSDFNet(
            output_building=2,
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        ) 
        self.deep_model = self.deep_model.cuda()
        self.lr = args.learning_rate
        self.epoch = args.max_iters // args.batch_size

        self.building_map_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model_type, 'building_map_GT')

        if not os.path.exists(self.building_map_saved_path):
            os.makedirs(self.building_map_saved_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.deep_model.eval()


    def infer(self):
        torch.cuda.empty_cache()
        dataset = BuildingDatset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=4, drop_last=False)
        torch.cuda.empty_cache()
        self.evaluator_loc.reset()      
        # vbar = tqdm(val_data_loader, ncols=50)

        # 假输入：根据你的数据实际通道数调整
        # x_opt = torch.randn(1, 3, 512, 512, device='cuda')
        # x_sar = torch.randn(1, 3, 512, 512, device='cuda')

        # with torch.no_grad():
        #     from thop import profile
        #     macs, params = profile(self.deep_model, inputs=(x_opt, x_sar), verbose=False)
        # flops = 2 * macs

        # print(f"Params: {format_big_number(params)}")
        # print(f"MACs:   {format_big_number(macs)}")
        # print(f"FLOPs:  {format_big_number(flops)}")

        with torch.no_grad():
            for itera, data in enumerate(tqdm(val_data_loader)):
                input_imgs_opt, input_imgs_sar, labels_loc, _, names = data

                input_imgs_opt = input_imgs_opt.cuda()
                input_imgs_sar = input_imgs_sar.cuda()
                labels_loc = labels_loc.cuda().long()

                output_loc, _ = self.deep_model(input_imgs_opt, input_imgs_sar)

                output_loc = output_loc.data.cpu().numpy()
                output_loc = np.argmax(output_loc, axis=1)
                labels_loc = labels_loc.cpu().numpy()

                self.evaluator_loc.add_batch(labels_loc, output_loc)

                image_name = names[0] + '.png'

                output_loc = np.squeeze(output_loc)
                output_loc[output_loc > 0] = 255

                imageio.imwrite(os.path.join(self.building_map_saved_path, image_name), output_loc.astype(np.uint8))

        loc_iou = self.evaluator_loc.Intersection_over_Union()
        loc_F1 = self.evaluator_loc.Pixel_F1_score()
        loc_pre = self.evaluator_loc.Pixel_Precision_Rate()
        loc_rec = self.evaluator_loc.Pixel_Recall_Rate()
        print(f'loc_iou is {loc_iou}, loc_F1 is {loc_F1}, loc_pre is {loc_pre}, loc_rec is {loc_rec}')


def main():
    parser = argparse.ArgumentParser(description="Inference on WHU Building dataset")
    parser.add_argument('--cfg', type=str, default='')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--type', type=str, default='test')
    parser.add_argument('--test_dataset_path', type=str, default='')
    parser.add_argument('--test_data_list_path', type=str, default='')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, default='SSDFNet')
    parser.add_argument('--result_saved_path', type=str, default='')

    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()

    with open(args.test_data_list_path, "r") as f:
        # data_name_list = f.read()
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    trainer = Trainer(args)
    trainer.infer()


if __name__ == "__main__":
    main()
