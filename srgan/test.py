import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from utils.model import Generator
from utils.utils import convert_rgb_to_ycbcr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default='result/model_params_100.pth')
    parser.add_argument('--image-file', type=str, default='testImage/lena.png')
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Generator().to(device)
    model.load_state_dict(torch.load(args.weights_file))
    # state_dict = model.state_dict()
    # for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
    #     if n in state_dict.keys():
    #         state_dict[n].copy_(p)
    #     else:
    #         raise KeyError(n)


    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
    image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)

    image.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)


    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    # psnr = calc_psnr(y, preds)
    # print('PSNR: {:.2f}'.format(psnr))
    #
    # preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    #
    # output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    # output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    preds = preds.mul(255.0).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    output = pil_image.fromarray(preds)
    output.save(args.image_file.replace('.', '_srgan_x{}.'.format(args.scale)))
