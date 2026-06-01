import sys
sys.path.append('/public/home/tanyh_25/project/Mamba/SSDFNet')

import argparse
import os
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import numpy as np

from buildingextraction.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from buildingextraction.datasets.make_data_loader import make_data_loader, BuildingDatset
from buildingextraction.utils_func.metrics import Evaluator
from buildingextraction.models.SSDFNet import SSDFNet

import buildingextraction.utils_func.lovasz_loss as L


class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        # dataloader
        self.train_data_loader = make_data_loader(args)

        # metrics
        self.evaluator_loc = Evaluator(num_class=2)

        # model
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

        # paths & hparams
        self.model_save_path = os.path.join(
            args.model_param_path, args.dataset, args.model_type + '_' + str(time.time())
        )
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        self.lr = args.learning_rate
        self.num_epochs = args.epochs  # <<< use epochs now

        # resume (optional)
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
            print(f"=> Loaded checkpoint from {args.resume}")

        # optimizer
        self.optim = optim.AdamW(
            self.deep_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

    def training(self):
        best_iou = 0.0
        best_round = []
        best_epoch = -1

        torch.cuda.empty_cache()

        global_iter = 0
        for epoch in range(1, self.num_epochs + 1):
            self.deep_model.train()
            pbar = tqdm(enumerate(self.train_data_loader, start=1), total=len(self.train_data_loader))
            for itera, data in pbar:
                global_iter += 1

                input_imgs_opt, input_imgs_sar, labels_loc, label_edge, _ = data
                input_imgs_opt = input_imgs_opt.cuda(non_blocking=True)
                input_imgs_sar = input_imgs_sar.cuda(non_blocking=True)
                labels_loc = labels_loc.cuda(non_blocking=True).long()
                label_edge = label_edge.cuda(non_blocking=True).long()

                output_loc, feat_edge = self.deep_model(input_imgs_opt, input_imgs_sar)

                self.optim.zero_grad()

                ce_loss_loc = F.cross_entropy(output_loc, labels_loc, ignore_index=255)
                lovasz_loss_loc = L.lovasz_softmax(F.softmax(output_loc, dim=1), labels_loc, ignore=255)
                ce_loss_edge_1 = F.cross_entropy(feat_edge, label_edge, ignore_index=255)

                final_loss = ce_loss_loc + 0.5 * lovasz_loss_loc + 0.3 * ce_loss_edge_1
                final_loss.backward()
                self.optim.step()

                if itera % 10 == 0:
                    pbar.set_description(
                        f"Epoch [{epoch}/{self.num_epochs}] Iter [{itera}/{len(self.train_data_loader)}] "
                        f"Loss {final_loss:.4f} | CE_loc {ce_loss_loc:.4f} | Lovasz {lovasz_loss_loc:.4f} | CE_edge {ce_loss_edge_1:.4f}"
                    )

            # ===== end of one epoch: do validation & checkpoint =====
            self.deep_model.eval()
            loc_iou, loc_F1, loc_pre, loc_rec = self.validation()
            if loc_iou > best_iou:
                save_path = os.path.join(self.model_save_path, f'epoch{epoch}_{loc_iou}_model.pth')
                torch.save(self.deep_model.state_dict(), save_path)
                best_iou = loc_iou
                best_round = [loc_iou, loc_F1, loc_pre, loc_rec]
                best_epoch = epoch
                print(f"=> New best model saved at {save_path}")

        print(f'Best epoch: {best_epoch}. Metrics = {best_round}')

    def validation(self):
        print('---------starting evaluation-----------')
        self.evaluator_loc.reset()

        dataset = BuildingDatset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=8, num_workers=4, drop_last=False)
        torch.cuda.empty_cache()

        with torch.no_grad():
            for _, data in enumerate(val_data_loader):
                input_imgs_opt, input_imgs_sar, labels_loc, _, _ = data

                input_imgs_opt = input_imgs_opt.cuda(non_blocking=True)
                input_imgs_sar = input_imgs_sar.cuda(non_blocking=True)
                labels_loc = labels_loc.cuda(non_blocking=True).long()

                output_loc, _ = self.deep_model(input_imgs_opt, input_imgs_sar)

                output_loc = output_loc.data.cpu().numpy()
                output_loc = np.argmax(output_loc, axis=1)
                labels_loc = labels_loc.cpu().numpy()

                self.evaluator_loc.add_batch(labels_loc, output_loc)

        loc_iou = self.evaluator_loc.Intersection_over_Union()
        loc_F1 = self.evaluator_loc.Pixel_F1_score()
        loc_pre = self.evaluator_loc.Pixel_Precision_Rate()
        loc_rec = self.evaluator_loc.Pixel_Recall_Rate()

        print(f'loc_iou is {loc_iou}, loc_F1 is {loc_F1}, loc_pre is {loc_pre}, loc_rec is {loc_rec}')
        return loc_iou, loc_F1, loc_pre, loc_rec


def main():
    parser = argparse.ArgumentParser(description="Training on Building Extraction dataset")
    parser.add_argument('--cfg', type=str, default='')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str, default='')

    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_dataset_path', type=str, default='')
    parser.add_argument('--train_data_list_path', type=str, default='')
    parser.add_argument('--test_dataset_path', type=str, default='')
    parser.add_argument('--test_data_list_path', type=str, default='')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)

    # <<< use epochs instead of max_iters >>>
    parser.add_argument('--epochs', type=int, default=250)

    parser.add_argument('--model_type', type=str, default='SSDFNet')
    parser.add_argument('--model_param_path', type=str, default='')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-3)

    args = parser.parse_args()

    # load train/val name lists
    with open(args.train_data_list_path, "r") as f:
        data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = data_name_list

    with open(args.test_data_list_path, "r") as f:
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
