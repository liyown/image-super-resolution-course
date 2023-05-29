import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from tqdm import tqdm

from utils import convert_rgb_to_y, convert


def train(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_patches = []
    hr_patches = []

    for image_path in tqdm(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        hr = pil_image.open(image_path).convert('RGB')

        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale

        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)

        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)

        # hr = convert(hr)
        # lr = convert(lr)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        len_img = 0
        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                len_img = len_img + 1
                lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])
                hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])

        for i in range(0, len_img):
            x = np.random.randint(0, lr.shape[0] - args.patch_size + 1)
            y = np.random.randint(0, lr.shape[1] - args.patch_size + 1)
            lr_patches.append(lr[x:x + args.patch_size, y:y + args.patch_size])
            hr_patches.append(hr[x:x + args.patch_size, y:y + args.patch_size])


        # 遍历所有可能的扫描区域，并计算方差
        variances = []
        for i in range(0, hr.shape[0] - args.patch_size + 1, 2):
            for j in range(0, hr.shape[1] - args.patch_size + 1, 2):
                window = hr[i:i + args.patch_size, j:j + args.patch_size]
                var = np.var(window)
                variances.append((var, i, j))
        # 对方差从大到小排序
        variances.sort(reverse=True)
        # 计算要截取的区域大小
        num_regions = int(0.06 * len(variances))
        # 截取出前 20% 方差最大的区域
        for i in range(num_regions):
            var, x, y = variances[i]
            lr_patches.append(lr[x:x + args.patch_size, y:y + args.patch_size])
            hr_patches.append(hr[x:x + args.patch_size, y:y + args.patch_size])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)
    print(len(lr_patches))
    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        print(i)
        hr = pil_image.open(image_path).convert('RGB')

        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale

        hr = hr.resize((hr_width, hr_height), resample=pil_image.Resampling.BICUBIC)
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.Resampling.BICUBIC)
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.Resampling.BICUBIC)

        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)

        # hr = convert(hr)
        # lr = convert(lr)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)
    print(len(lr_group))
    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--images-dir', type=str, default='data/T91-train')
    # parser.add_argument('--output-path', type=str, default='data/train_data_y.h5')
    # parser.add_argument('--eval', action='store_true', default=False)

    parser.add_argument('--images-dir', type=str, default='data/Set5-test')
    parser.add_argument('--output-path', type=str, default='data/test_data_y.h5')
    parser.add_argument('--eval', action='store_true', default=True)

    parser.add_argument('--patch-size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        eval(args)
