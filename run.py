import os
os.system("python ./srcnn/train_srcnn.py --train-file ./data/train_data_y.h5 --eval-file ./data/test_data_y.h5 "
          "--outputs-dir srcnn/result")

os.system("python ./srcnn_vgg/train_srcnn_vgg.py --train-file ./data/train_data_y.h5 --eval-file ./data/test_data_y.h5 "
          "--outputs-dir srcnn_vgg/result")

os.system("python ./srunet/train.py --train-file ./data/train_data_y.h5 --eval-file ./data/test_data_y.h5 "
          "--outputs-dir srunet/result")
