import os
import numpy as np
import glob
import logging
import skimage
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageFont, ImageDraw
import scipy.cluster
logging.basicConfig(level=logging.INFO, format="[%(lineno)4s : %(funcName)-30s ] %(message)s")

# GLOBALS
img_width = 224             # For VGG16
img_height = 224            # For VGG16
img_channel = 3

### FUNCTIONS ###
def init_globals(fashion_dataset_path):
    input_shape = (img_width, img_height, img_channel)
    class_names = []
    with open(fashion_dataset_path + 'Anno/list_category_cloth.txt') as f:
        next(f)
        next(f)
        for line in f:
            class_names.append(line.split()[0])
    attr_names = []
    with open(os.path.join(fashion_dataset_path, 'Anno/list_attr_cloth.txt')) as f:
        next(f)
        next(f)
        for line in f:
            attr_names.append('-'.join(line.split()[:-1]))
    logging.info('classes {} {}'.format(len(class_names), class_names))
    logging.info('attributes {} {}'.format(len(attr_names), attr_names))
    return (class_names, input_shape, attr_names)

def bb_intersection_over_union(boxes1, boxes2):
    x11, y11, x12, y12 = boxes1[0], boxes1[1], boxes1[0] + boxes1[2], boxes1[1] + boxes1[3]
    x21, y21, x22, y22 = boxes2[0], boxes2[1], boxes2[0] + boxes2[2], boxes2[1] + boxes2[3]
    xA = np.maximum(x11, x21)
    yA = np.maximum(y11, y21)
    xB = np.minimum(x12, x22)
    yB = np.minimum(y12, y22)
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

def plot_history(output_path, sep=';'):
    def plot_history(metric, names, log):
        plt.style.use("ggplot")
        (fig, ax) = plt.subplots(len(names), 1, figsize=(8, 8))
        if len(names) == 1:
            ax = [ax]
        # loop over the accuracy names
        for (i, l) in enumerate(names):
            # plot the loss for both the training and validation data
            ax[i].set_title("{} for {}".format(metric, l))
            ax[i].set_xlabel("Epoch #")
            ax[i].set_ylabel(metric)
            ax[i].plot(log.index, log[l], label=l)
            ax[i].plot(log.index, log["val_" + l],
                       label="val_" + l)
            ymin = np.min([np.min(log["val_" + l][20:]), np.min(log[l][20:])])
            ymax = np.max([np.max(log["val_" + l][20:]), np.max(log[l][20:])])
            if len(log[l]) > 20:
                ax[i].set_ylim([ymin, ymax])
            ax[i].legend()
        # save the accuracies figure
        plt.tight_layout()
        fig.show()
        fig.savefig(os.path.join(output_path, metric+'.png'))
    log = pd.DataFrame.from_csv(os.path.join(output_path, 'model_train.csv'), sep=sep)
    losses = [i for i in log.keys() if not i.startswith('val_') and i.endswith('loss')]
    accs = [i for i in log.keys() if not i.startswith('val_') and i.endswith('acc')]
    plot_history('Accuracy', accs, log)
    plot_history('Loss', losses, log)
    plt.show()
    return

def get_image_paths(prediction_path):
    images_path_name = sorted(glob.glob(prediction_path + '/*.*g'))
    if os.name == 'nt':
        images_path_name = [x.replace('\\', '/') for x in images_path_name]
    return images_path_name

def draw_rect(ax, img, gt_bbox, text=None, textcolor=(0,0,0), edgecolor='red',linewidth=4):
    def display_bbox_text(img, bbox, text, color=(0, 0, 0), fontsize=32):
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        # font = ImageFont.truetype("sans-serif.ttf", 16)
        # font = ImageFont.truetype("DroidSans.ttf", 16)
        # font = ImageFont.truetype('fonts/alterebro-pixel-font.ttf', 30)
        # font = ImageFont.truetype('/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-C.ttf', 16)
        # font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font = ImageFont.truetype('extra/fonts/Ubuntu-C.ttf', fontsize)
        draw.text((bbox[0], bbox[1]), text, color, font=font)
        # draw.text((bbox[0], bbox[1]), text,(255,0,0),font=font)
    x, y, w, h = gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]
    rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor=edgecolor, linewidth=linewidth)
    ax.add_patch(rect)
    if text is not None:
        display_bbox_text(img, gt_bbox, text, textcolor)
    ax.imshow(img, aspect='equal')

if __name__ == '__main__':
    color_map = []
    with open('../Data/color_table.txt') as f:
        for line in f:
            line = line.split()
            r, g, b = line[1][:2], line[1][2:4], line[1][4:]
            color_map.append(([int(r, 16), int(g, 16), int(b, 16)], line[0], '#'+line[1]))
    colors = np.array([x[0] for x in color_map])
    images_path_name = sorted(glob.glob('../color_pred/*.*g'))
    NUM_COLORS = 5
    for name in images_path_name:
        im = Image.open(name)
        tt = im.getcolors(NUM_COLORS)
        # im = im.resize((150, 150))  # optional, to reduce time
        ar = np.asarray(im)
        shape = ar.shape
        ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

        codes, dist = scipy.cluster.vq.kmeans(ar, NUM_COLORS)
        # print ('cluster centres:\n', codes)
        vecs, dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
        counts, bins = scipy.histogram(vecs, len(codes))  # count occurrences
        s_palette = [x[1]/256 for x in sorted(zip(counts, codes), reverse=True)]
          # find most frequent
        s_peak = codes[scipy.argmax(counts)]
        color_thief = ColorThief(name)
        cf_peak = color_thief.get_color(quality=1)
        cf_palette = [(x[0]/256,x[1]/256,x[2]/256) for x in color_thief.get_palette(color_count=NUM_COLORS)]
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), frameon=False)
        ax.imshow(im)
        for i, cc in enumerate(s_palette):
            # color = [int(cc[0]), int(cc[1]), int(cc[2])]
            ax.add_patch(mpatches.CirclePolygon((100, 50 + 10*i), color=cc))
        for i, cc in enumerate(cf_palette):
            # color = [int(cc[0]), int(cc[1]), int(cc[2])]
            ax.add_patch(mpatches.CirclePolygon((150, 50 + 10*i), color=cc))
        ax.imshow(im)
        plt.show()
        print('scipy', s_peak, s_palette)
        print('cf', cf_peak, cf_palette)
    # import sys
    # if len(sys.argv) == 2:
    #     model_path = sys.argv[1]
    # plot_history(model_path, sep=';')



