python scripts/extract_subimages_ARCHI4K.py
python scripts/generate_meta_info_pairdata_ARCHI4K.py
python realesrgan/train.py -opt options/finetune_realesrgan_x4plus_pairdata_ARCHI4K.yml --auto_resume
python realesrgan/train.py -opt options/train_realesrgan_x3plus_ARCHI4K.yml --auto_resume