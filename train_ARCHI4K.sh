python generate_multiscale_ARCHI4K.py
python scripts/generate_meta_info.py --input datasets/ARCHI4K/ARCHI4K_HR, datasets/ARCHI4K/ARCHI4K_multiscale --root datasets/ARCHI4K, datasets/ARCHI4K --meta_info datasets/ARCHI4K/meta_info/meta_info_ARCHI4Kmultiscale.txt
python realesrgan/train.py -opt options/train_realesrnet_x4plus_ARCHI4K.yml --auto_resume --debug