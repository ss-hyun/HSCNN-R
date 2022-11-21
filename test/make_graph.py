import os
import numpy as np
from utils import rrmse
import h5py
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

img_path = './ss_results/'
model_path = './models/layer14_20-to-100_val-14-15-18+loss+4.60_29.pkl'
input_chan = 20
model_label = model_path.split('/')[2][:-4]
# input_img = 'bio'
# dots = [(65, 65), (65, 145), (65, 225), (65, 305), (65, 65)]
input_img = 'chart'
numbers = [1, 7, 14, 15, 18]
dots = [(65 + 80 * ((i - 1) // 5), 65 + 80 * ((i - 1) % 5)) for i in numbers]
# dots = [(65, 65), (145, 385), (225, 225), (310, 145), (310, 385)]
# dots = [(65, 65), (465, 65), (465, 145), (465, 225), (465, 305)]
chan_label = 'chann+{}'.format(input_chan)

for img_name in sorted(os.listdir(img_path)):
    if input_img in img_name:
        if 'input' in img_name and chan_label in img_name:
            data = h5py.File(os.path.join(img_path, img_name))
            filtered = np.transpose(data.get('F_color_chart')).copy()
            f_w_length = np.squeeze(data.get('filtered_w_length'))
        if 'recon' in img_name and model_label in img_name:
            data = h5py.File(os.path.join(img_path, img_name))
            recon = np.transpose(data.get('reconstruct')).copy()
        if 'GT' in img_name:
            data = h5py.File(os.path.join(img_path, img_name))
            gt = np.transpose(data.get('N_color_chart')).copy()
            w_length = np.squeeze(data.get('w_length'))

# print(filtered.shape)
# print(recon.shape)
# print(gt.shape)
# plt.imshow(gt.sum(axis=2))
# plt.show()
# exit()

mask_gt = gt.copy()
np.putmask(mask_gt, mask_gt == 0, 1000)

f, axes = plt.subplots(2, 3)
f.set_size_inches((17, 9))
f.suptitle('Spectrum Graph ({})'.format(model_label), fontsize=15)
plt.tight_layout(pad=2, h_pad=2.5)

im = axes[0, 0].imshow(abs((recon - gt) / mask_gt).mean(axis=2))
plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
axes[0, 0].set_title('relative error mean')

for i, dot in enumerate(dots):
    axes[0, 0].scatter(dot[0], dot[1], c='red', s=8, marker='*')
    axes[0, 0].text(dot[0] - 25, dot[1] - 10, dot, size=9, color='white')
    i = i + 1
    axes[i // 3, i % 3].plot(w_length, gt[dot[1], dot[0], :])
    axes[i // 3, i % 3].plot(w_length, recon[dot[1], dot[0], :])
    axes[i // 3, i % 3].plot(f_w_length, filtered[dot[1], dot[0], :])
    axes[i // 3, i % 3].set_title(dot)
    axes[i // 3, i % 3].legend(['GT', 'Reconstruction', 'Filtered'])
    # axes[i//3, i%3].set_ylim([0, 1])

print(rrmse(recon, gt, mask_gt))

plt.show()
