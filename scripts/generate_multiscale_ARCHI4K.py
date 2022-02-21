import argparse
import glob
import os
from PIL import Image


def main(args):
    # For DF2K, we consider the following three scales,
    # and the smallest image whose shortest edge is 400
    scale_list = [3 / 4, 2 / 4, 1 / 3]
    # shortest_edge = 400

    path_list = sorted(glob.glob(os.path.join(args.input[0], '*')))
    for path in path_list:
        print(path)
        basename = os.path.splitext(os.path.basename(path))[0]

        img = Image.open(path)
        width, height = img.size
        for idx, scale in enumerate(scale_list):
            print(f'\t{scale:.2f}')
            rlt = img.resize((int(width * scale), int(height * scale)), resample=Image.LANCZOS)
            rlt.save(os.path.join(args.output, f'{basename}T{idx}.png'))

        # save the smallest image which the shortest edge is 400
        # if width < height:
        #     ratio = height / width
        #     width = shortest_edge
        #     height = int(width * ratio)
        # else:
        #     ratio = width / height
        #     height = shortest_edge
        #     width = int(height * ratio)
        # rlt = img.resize((int(width), int(height)), resample=Image.LANCZOS)
        # rlt.save(os.path.join(args.output, f'{basename}T{idx+1}.png'))
        img.save(os.path.join(args.output, f'{basename}T{idx+1}.png'))


if __name__ == '__main__':
    """Generate multi-scale versions for GT images with LANCZOS resampling.
    It is now used for ARCHI4K dataset
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=['datasets/ARCHI4K/ARCHI4K_HR', 'datasets/ARCHI4K/ARCHI4K_LR'], help='Input folder')
    parser.add_argument('--output', type=str, default='datasets/ARCHI4K/ARCHI4K_multiscale', help='Output folder')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    main(args)
