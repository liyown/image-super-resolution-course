import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import vgg19

from utils.dataset import TrainDataset

from utils.model import Discriminator

adversarial_loss = nn.BCEWithLogitsLoss()
content_loss = nn.MSELoss()
perception_loss = nn.MSELoss()

# 定义超参数
lr = 0.0002
epochs = 100

# 加载数据集
train_dataset = TrainDataset("data/train_data_y.h5")
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=16,
                              shuffle=True,
                              drop_last=True)

# 初始化生成器和判别器
generator = Discriminator().cuda()
print(generator._modules.items())
discriminator = Discriminator().cuda()
VGG_model = vgg19(pretrained=True)
VGG_feature_model = nn.Sequential(*list(VGG_model.features)[:-1]).eval()
for param in VGG_feature_model.parameters():
    param.requires_grad = False
VGG_feature_model.cuda()

# 定义优化器
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_G = optim.Adam([
     {'params': generator.conv1.parameters()},
     {'params': generator.conv2.parameters()},
     {'params': generator.conv3.parameters(), 'lr': lr * 0.1}
 ], lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# 训练模型
for epoch in range(epochs):
    for i, (low_res, high_res) in enumerate(train_dataloader):
        imgs = []
        low_res, high_res = low_res.cuda(), high_res.cuda()
        imgs.append(low_res[0])
        imgs.append(high_res[0])
        # 训练判别器
        fake_high_res = generator(low_res)
        imgs.append(fake_high_res[0])

        real_labels = torch.ones((low_res.size(0), 1)).cuda()
        fake_labels = torch.zeros((low_res.size(0), 1)).cuda()

        real_loss = adversarial_loss(discriminator(high_res), real_labels)
        fake_loss = adversarial_loss(discriminator(fake_high_res.detach()), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss = d_loss

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        fake_high_res = generator(low_res)
        g_loss = content_loss(fake_high_res, high_res) + \
                 1e-3 * adversarial_loss(discriminator(fake_high_res), real_labels) + \
                 2e-6 * perception_loss(VGG_feature_model(fake_high_res), VGG_feature_model(high_res))

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        # 打印损失
        if i % 100 == 0:
            # 保存图像到本地文件
            torchvision.utils.save_image(imgs, './result/generated_images_{0}_{1}.png'.format(epoch + 1, i + 1),
                                         normalize=True)

            print("[Epoch{}/{}] [Batch{}/{}] [D loss:{:.4f}] [G loss:{:.4f}]".format(epoch + 1, epochs, i + 1,
                                                                                     len(train_dataloader),
                                                                                     d_loss.item(), g_loss.item()))
    torch.save(generator.state_dict(), './result/model_params_{}.pth'.format(epoch + 1))
