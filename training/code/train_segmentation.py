from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
import fastai
import warnings
warnings.filterwarnings("ignore")
torch.cuda.set_device(1)

path_lbl = Path("../../data/masks")
path_img = Path("../../data/images")

fnames = get_image_files(path_img)
lbl_names = get_image_files(path_lbl)
img_f = fnames[4]

get_y_fn = lambda x: path_lbl/f'{x.stem}.png'
mask = open_mask(get_y_fn(img_f), div=False)
src_size = np.array(mask.shape[1:])

codes = ['background', 'stairs']

def get_dataset(size, path_img, path_lbl, batch_size, codes):
    return (SegmentationItemList.from_folder(path_img)
            .split_by_rand_pct(seed=38)
            .label_from_func(lambda x: path_lbl/f'{x.stem}.png', classes=codes)
            .transform(get_transforms(), size=size, tfm_y=True)
            .databunch(bs=batch_size)
            .normalize(imagenet_stats))

def accuracy(input, target):
    target = target.squeeze(1)
    return (input.argmax(dim=1)==target).float().mean()

metrics=accuracy
wd=1e-2
model = models.resnet34

for stage, size, batchsize, lr in zip([1, 2], [src_size//2, src_size], [8, 4], [1e-3, 1e-5]):
    data = get_dataset(size, path_img, path_lbl, batchsize, codes)
    learn = unet_learner(data, model, metrics=metrics, wd=wd, callback_fns=ShowGraph)
    if stage == 2 : learn.load('stage-1')
    learn.fit_one_cycle(1, slice(lr), pct_start=0.9)
    learn.unfreeze()
    lrs = slice(lr/400, lr/4)
    learn.fit_one_cycle(1, lrs, pct_start=0.8)
    learn.save(f'stage-{stage}')
    print(f"completed stage {stage}")
    learn.export()
    learn.destroy()

print("training completed!")