import argparse
import cv2
import numpy as np
import os
import sys
from basicsr.utils import scandir
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm


def main(args):
    """A multi-thread tool to crop large images to sub-images for faster IO.

    opt (dict): Configuration dict. It contains:
        n_thread (int): Thread number.
        compression_level (int):  CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size
            and longer compression time. Use 0 for faster CPU decompression. Default: 3, same in cv2.
        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        crop_size (int): Crop size.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower than thresh_size will be dropped.

    Usage:
        For each folder, run this script.
        Typically, there are GT folder and LQ folder to be processed for DIV2K dataset.
        After process, each sub_folder should have the same number of subimages.
        Remember to modify opt configurations according to your settings.
    """

    opt = {}
    opt['n_thread'] = args.n_thread
    opt['compression_level'] = args.compression_level
    opt['input_folder'] = args.input
    opt['save_folder'] = args.output
    opt['crop_size'] = args.crop_size
    opt['step'] = args.step
    opt['scale'] = args.scale
    opt['thresh_size'] = args.thresh_size
    extract_subimages(opt)


def extract_subimages(opt):
    """Crop images to subimages.

    Args:
        opt (dict): Configuration dict. It contains:
            input_folder (str): Path to the input folder.
            save_folder (str): Path to save folder.
            n_thread (int): Thread number.
    """
    input_folders = opt['input_folder']
    save_folders = opt['save_folder']
    for save_folder in save_folders:
        os.makedirs(save_folder, exist_ok=True)
        print(f'mkdir {save_folder} ...')

    # scan all images
    hr_list = list(scandir(input_folders[0], full_path=True))
    lr_list = list(scandir(input_folders[1], full_path=True))
    assert len(hr_list) == len(lr_list)

    img_list = hr_list
    opt['save_folder'] = save_folders[0]
    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    worker(img_list[0], opt)
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('HR All processes done.')

    if opt['scale'] == 4:
        opt['resize'] = True
    img_list = lr_list
    opt['crop_size'] /= opt['scale']
    opt['step'] /= opt['scale']
    opt['save_folder'] = save_folders[1]
    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    worker(img_list[0], opt)
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('LR All processes done.')

def worker(path, opt):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            crop_size (int): Crop size.
            step (int): Step for overlapped sliding window.
            thresh_size (int): Threshold size. Patches whose size is lower than thresh_size will be dropped.
            save_folder (str): Path to save folder.
            compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    crop_size = int(opt['crop_size'])
    step = int(opt['step'])
    thresh_size = opt['thresh_size']
    resize = opt['resize'] if 'resize' in opt else False
    img_name, extension = osp.splitext(osp.basename(path))

    # remove the x2, x3, x4 and x8 in the filename for DIV2K
    img_name = img_name.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if resize:
        img = cv2.resize(img, dsize=(0, 0), fx=3/4, fy=3/4, interpolation=cv2.INTER_LANCZOS4)

    h, w = img.shape[0:2]
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imwrite(
                osp.join(opt['save_folder'], f'{img_name}_s{index:03d}{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processing {img_name} ...'
    return process_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=['datasets/ARCHI4K/ARCHI4K_HR', 'datasets/ARCHI4K/ARCHI4K_LR'] , help='Input folder')
    parser.add_argument('--output', type=str, default=['datasets/ARCHI4K/ARCHI4K_HR_sub', 'datasets/ARCHI4K/ARCHI4K_LR_sub'], help='Output folder')
    parser.add_argument('--crop_size', type=int, default=480, help='Crop size')
    parser.add_argument('--step', type=int, default=360, help='Step for overlapped sliding window')
    parser.add_argument('--scale', type=float, default=3.0, help='LR scale')
    parser.add_argument(
        '--thresh_size',
        type=int,
        default=0,
        help='Threshold size. Patches whose size is lower than thresh_size will be dropped.')
    parser.add_argument('--n_thread', type=int, default=20, help='Thread number.')
    parser.add_argument('--compression_level', type=int, default=3, help='Compression level')
    args = parser.parse_args()

    main(args)
