from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
import fastai
import cv2 as cv
import matplotlib.pyplot as plt
import warnings
import random, string

warnings.filterwarnings("ignore")

new_path = Path("../data/bh_data")

tfms = get_transforms(do_flip=False)
data = (ImageList.from_folder(new_path)
       .split_by_rand_pct(seed=38)
       .label_from_folder()
       .transform(tfms)
       .databunch(bs=5)
       .normalize(imagenet_stats))

learn = cnn_learner(data, models.resnet18, metrics=accuracy, wd=1e-2)
learn.fit_one_cycle(1,1e-2)
learn.save('bh_learner')
learn.export()