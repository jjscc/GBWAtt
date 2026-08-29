import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
import torch
from utils.functions import train_network, save_res
from mAtt.mAtt import GBWAttMamem
from utils.GetMamem import get_all_dataloader
import argparse
import hydra
from omegaconf import DictConfig



@hydra.main(config_path="./conf/", config_name="mamem.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    repeat = 1
    res = torch.zeros(11, repeat)
    for i in range(1):
        for j in range(repeat):
            ap = argparse.ArgumentParser()
            ap.add_argument('--repeat', type=int, default=1, help='No.xxx repeat for training model')
            ap.add_argument('--sub', type=int, default=j+1, help='subjectxx you want to triain')
            ap.add_argument('--lr', type=float, default=cfg.lr, help='learning rate')
            ap.add_argument('--wd', type=float, default=cfg.wd, help='weight decay')
            ap.add_argument('--iterations', type=int, default=180, help='number of training iterations')
            ap.add_argument('--epochs', type=int, default=cfg.epochs,
                            help='number of epochs that you want to use for split EEG signals')
            ap.add_argument('--bs', type=int, default=64, help='batch size')
            ap.add_argument('--model_path', type=str, default='./checkpoint/mamem/',
                            help='the folder path for saving the model')
            ap.add_argument('--data_path', type=str, default='./data/MAMEM/', help='data path')
            ap.add_argument('--res_path', type=str, default='./result/mamem/ablation_tangent.',
                            help='data path')
            ap.add_argument("--scoring_metric", choices=["acc", "auc"], default="acc", help="performance_metric")
            ap.add_argument('--optim', type=str, default='adam', help='Optimization method.')
            ap.add_argument('--device', type=str, default='cuda', help='Optimization method.')
            ap.add_argument('--spd_device', type=str, default='cpu', help='Optimization method.')
            ap.add_argument('--in_size', type=int, default=cfg.in_size)
            ap.add_argument('--out_size', type=int, default=cfg.out_size)
            ap.add_argument('--power', type=float, default=cfg.power)
            args, unknown = ap.parse_known_args()

            trainLoader, validLoader, testLoader = get_all_dataloader(args)
            net = GBWAttMamem(args)
            if not os.path.exists(args.res_path):
                os.makedirs(args.res_path)
            final_path = os.path.join(args.res_path,
                                      f'gbwm_lr{args.lr}_wd{args.wd}_{args.optim}_power{cfg.power}.txt')

            acc = train_network(net, trainLoader, validLoader, testLoader, args)

            print(f'{acc * 100:.2f}')
            res[i, j] = acc * 100
            print(f'{acc * 100:.2f}')
            print(acc)
            print(res)
            save_res(res, final_path)


if __name__ == '__main__':
    main()
