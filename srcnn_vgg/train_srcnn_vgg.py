import argparse
import json
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from torchvision.models import vgg19
from tqdm import tqdm

from utils.dataset import TrainDataset, EvalDataset
from utils.model import SRCNN
from utils.utils import AverageMeter, calc_psnr, get_features, gram_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default='../data/train_data_y.h5')
    parser.add_argument('--eval-file', type=str, default='../data/test_data_y.h5')
    parser.add_argument('--outputs-dir', type=str, default='./result')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    vgg = vgg19(pretrained=True).cuda().features
    for param in vgg.parameters():
        param.requires_grad_(False)

    content_layers = ['21']
    style_layers = ['0', '5', '10', '19', '28']

    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  pin_memory=True)

    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    result_srcnn_vgg = {"loss": [], "psnr": [0, ]}
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset))) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for i, data in enumerate(train_dataloader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = model(inputs)
                loss_c = criterion(preds, labels)

                preds_features = get_features(vgg, torch.cat([preds, preds, preds], dim=1), content_layers + style_layers)
                labels_features = get_features(vgg, torch.cat([labels, labels, labels], dim=1), content_layers + style_layers)

                # # 计算损失函数
                content_loss_value = 0
                for content_layer in content_layers:
                    content_loss_value += criterion(preds_features[content_layer], labels_features[content_layer].detach())

                style_loss_value = 0
                for style_layer in style_layers:
                    preds_feature = preds_features[style_layer]
                    preds_gram = gram_matrix(preds_feature)
                    labels_feature = labels_features[style_layer]
                    labels_gram = gram_matrix(labels_feature)
                    style_loss_value += criterion(preds_gram, labels_gram.detach())

                loss = loss_c + 1e-3 * content_loss_value + 1e-4 * style_loss_value
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
                if i % 1000 == 0:
                    result_srcnn_vgg["loss"].append(epoch_losses.avg)
        # torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels).item(), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        result_srcnn_vgg["psnr"].append(epoch_psnr.avg)

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())
    with open(os.path.join(args.outputs_dir, "result_srcnn_vgg.json.json"), "w") as f:
        json.dump(result_srcnn_vgg, f)

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
