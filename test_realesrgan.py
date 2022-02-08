import os
import urllib.request as request
import cv2
import numpy as np
import unittest
import time

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

output = 'results_render'

# os.makedirs(input, exist_ok=True)
os.makedirs(output, exist_ok=True)
resolutions = {
    "low": [550, 960],
    "std": [1280],
    "fhd": [1920],
    "qhd": [2560],
    "4k": [4000, 3840],
    "8k": [8000, 7680]
}

test_cnt = 100

class Test_realesrgan(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

        model_name = 'RealESRGAN_x4plus'
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        tile = 960

        # determine model paths
        model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
        if not os.path.isfile(model_path):
            model_path = os.path.join('realesrgan/weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            raise ValueError(f'Model {model_name} does not exist.')

        # restorer
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=tile)

    def get_resolution(self, width):
        return [list(resolutions.keys())[i] for i, val in enumerate(resolutions.values()) if width in val][0]

    def url_to_image(self, url):
        # download the image, convert it to a NumPy array, and then read
        # it into OpenCV format
        resp = request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # return the image
        return image

    def save_results(self, testId, sr, lr, hr=None):
        save_path = os.path.join(output, f'{testId}')
        cv2.imwrite(f"{save_path}_LR({self.get_resolution(lr.shape[1])}).png", lr)
        if hr is not None:
            cv2.imwrite(f"{save_path}_HR({self.get_resolution(hr.shape[1])}).png", hr)
        cv2.imwrite(f"{save_path}_SR({self.get_resolution(sr.shape[1])})_x{sr.shape[1] / lr.shape[1]}.png", sr)

    def save_downscale_results(self, testId, sr_down, lr=None):
        save_path = os.path.join(output, f'{testId}')
        if lr is not None:
            cv2.imwrite(f"{save_path}_LR({self.get_resolution(lr.shape[1])}).png", lr)
        cv2.imwrite(f"{save_path}_SR({self.get_resolution(sr_down.shape[1])})_down.png", sr_down)

    def downscale(self, image, method='bicubic', scale=0.25):
        if method == 'bicubic':
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_LINEAR
        return cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale, interpolation=interpolation)

    def test_a_standard_to_4k(self):
        lr_urls = ['https://resources.archisketch.com/images/X5G8IYXB1A2FB9205824FC3/1280x960/X5G8IYXB1A2FB9205824FC3.png',
                  'https://archisketch-resources.s3.ap-northeast-2.amazonaws.com/images/X5RUPVA679DE247E6624DDC/1280x720/X5RUPVA679DE247E6624DDC.png',
                  'https://resources.archisketch.com/images/X6QfDPQ8501E1AC4CD749BB/1280x720/X6QfDPQ8501E1AC4CD749BB.png']

        url_times = 0
        sr_times = 0
        print('====== test_a_standard_to_4k start ======')
        for i in range(test_cnt):
            # url_to_image
            start = time.time()
            lr = self.url_to_image(lr_urls[i % len(lr_urls)])
            url_time = time.time() - start
            print(f'    url - proc_time = {round(url_time, 2)} s')

            # sr
            start = time.time()
            sr, _ = self.upsampler.enhance(lr, outscale=3.0)
            sr_time = time.time() - start
            print(f'    sr  - proc_time = {round(sr_time, 2)} s')

            url_times += url_time
            sr_times += sr_time
            time.sleep(1)
        print(f'total = {round((url_times+sr_times)/test_cnt, 2)} s, '
              f'url_time = {round(url_times/test_cnt, 2)} s, '
              f'sr_time = {round(sr_times/test_cnt, 2)} s')
        print('====== test_a_standard_to_4k end ======')


    def test_b_fhd_to_4k(self):
        lr_urls = [
            'https://archisketch-resources.s3.ap-northeast-2.amazonaws.com/images/X5RUuVWC8AB408CC27240D0/1920x1080/X5RUuVWC8AB408CC27240D0.png',
            'https://resources.archisketch.com/images/X6QfHzTC867AEEB593248F9/1920x1080/X6QfHzTC867AEEB593248F9.png']
        url_times = 0
        sr_times = 0
        print('====== test_b_fhd_to_4k start ======')
        for i in range(test_cnt):
            # url_to_image
            start = time.time()
            lr = self.url_to_image(lr_urls[i % len(lr_urls)])
            url_time = time.time() - start
            print(f'    url - proc_time = {round(url_time, 2)} s')

            # sr
            start = time.time()
            sr, _ = self.upsampler.enhance(lr, outscale=2.0)
            sr_time = time.time() - start
            print(f'    sr  - proc_time = {round(sr_time, 2)} s')

            url_times += url_time
            sr_times += sr_time
            time.sleep(1)
        print(f'total = {round((url_times + sr_times) / test_cnt, 2)} s, '
              f'url_time = {round(url_times / test_cnt, 2)} s, '
              f'sr_time = {round(sr_times / test_cnt, 2)} s')
        print('====== test_b_fhd_to_4k end ======')

if __name__ == '__main__':
    unittest.main()
