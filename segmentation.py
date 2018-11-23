import os
import numpy as np
import glob
import skimage
import selectivesearch
import random
import shutil
from PIL import Image
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
from utils import bb_intersection_over_union, init_globals, get_image_paths, draw_rect
import logging
logging.basicConfig(level=logging.INFO, format="[%(lineno)4s : %(funcName)-30s ] %(message)s")

### GLOBALS
img_width = 224             # For VGG16
img_height = 224            # For VGG16
img_channel = 3
prediction_path = 'prediction/'
fashion_dataset_path='fashion_data/'

def selective_search_bbox_fast(image, region_pixels_threshold, min_edge=1, max_ratio=6):
    img_lbl, regions = selectivesearch.selective_search(image, scale=1, sigma=0)
    candidates = set()
    for r in regions:
        x, y, w, h = r['rect']
        if r['rect'] in candidates:
            continue
        if r['size'] < region_pixels_threshold:
            continue
        if h < min_edge and w / h > max_ratio:
            continue
        if w < min_edge and h / w > max_ratio:
            continue
        candidates.add(r['rect'])
    return candidates

def cluster_bboxes(bboxes,orig_width, orig_height, width, height, preference=-0.5, fast=False):
    bboxes_clustered = []
    X = np.array([[bb[0] / width, bb[1] / height, bb[2] / width, bb[3] / height] for bb in bboxes])
    if len(X) == 1:
        return [(X[0][0] * orig_width, X[0][1] * orig_height, X[0][2] * orig_width, X[0][3] * orig_height)]
    af = AffinityPropagation(preference=preference).fit(X)
    labels = af.labels_
    if fast == True:
        return [(centroid[0] * orig_width, centroid[1] * orig_height, centroid[2] * orig_width, centroid[3] * orig_height) for centroid in af.cluster_centers_]
    for cluster in np.unique(labels):
        bboxes_cluster = X[labels == cluster]
        km = KMeans(n_clusters=1).fit(bboxes_cluster)
        closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, bboxes_cluster)
        centroid = km.cluster_centers_[0]
        bboxes_clustered.append((centroid[0] * orig_width, centroid[1] * orig_height, centroid[2] * orig_width, centroid[3] * orig_height))
    return bboxes_clustered

def selective_search_aggregated(image):
    def selective_search_bboxwh(image, scale, sigma, rescale, min_size=50):
        pl, ph = np.percentile(image, (rescale, 100 - rescale))
        image = skimage.exposure.rescale_intensity(image, in_range=(pl, ph))
        height, width, channels = image.shape
        region_pixels_threshold = (width * height) / 50
        img_lbl, regions = selectivesearch.selective_search(image, scale=scale, sigma=sigma, min_size=min_size)
        candidates = set()
        for r in regions:
            x, y, w, h = r['rect']
            if r['rect'] in candidates:
                continue
            if r['size'] < region_pixels_threshold:
                continue
            if h != 0 and w / h > 6:
                continue
            if w != 0 and h / w > 6:
                continue
            candidates.add(r['rect'])
        return candidates
    bbox_agg = set()
    [bbox_agg.add(y) for x, y, z in [(300, 0, 0), (1, 0.35, 0), (300, 0, 25), (1, 0.35, 25)] for y in selective_search_bboxwh(image, x, y, z)]
    return list(bbox_agg)

def display_seg(image_path, gt_bbox):
    # fig, axes = plt.subplots(2, 3, figsize=(8, 6), frameon=False)
    # xs=[0.25,0.8]
    # bbox_agg = set()
    # for ax, x in zip(axes[:,0], xs):
    #     bboxes, img = selective_search_bboxwh(image_path, sigma=x)
    #     for bbox in bboxes:
    #         bbox_agg.add(bbox)
    #         color = np.random.rand(3,)
    #         draw_rect(ax, img, bbox, edgecolor=color, linewidth=1)
    #         ax.plot(bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2, c=color, marker='o')
    #     ax.set_xlabel(str(len(bboxes)))
    #     ax.set_ylabel(str(x))
    #     x, y, w, h = gt_bbox[0], gt_bbox[1], gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1]
    #     draw_rect(ax, img, (x, y, w, h), edgecolor='green', linewidth=2)
    # xs = [500, 1500]
    # for ax, x in zip(axes[:,1], xs):
    #     bboxes, img = selective_search_bboxwh(image_path, scale=x)
    #     for bbox in bboxes:
    #         bbox_agg.add(bbox)
    #         color = np.random.rand(3, )
    #         draw_rect(ax, img, bbox, edgecolor=color, linewidth=1)
    #         ax.plot(bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, c=color, marker='o')
    #     ax.set_xlabel(str(len(bboxes)))
    #     ax.set_ylabel(str(x))
    #     x, y, w, h = gt_bbox[0], gt_bbox[1], gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1]
    #     draw_rect(ax, img, (x, y, w, h), edgecolor='green', linewidth=2)
    # xs = [2, 25]
    # for ax, x in zip(axes[:,2], xs):
    #     bboxes, img = selective_search_bboxwh(image_path, rescale=x)
    #     ax.imshow(img)
    #     for bbox in bboxes:
    #         bbox_agg.add(bbox)
    #         color = np.random.rand(3, )
    #         draw_rect(ax, img, bbox, edgecolor=color, linewidth=1)
    #         ax.plot(bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, c=color, marker='o')
    #     ax.set_xlabel(str(len(bboxes)))
    #     ax.set_ylabel(str(x))
    #     x, y, w, h = gt_bbox[0], gt_bbox[1], gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1]
    # #     draw_rect(ax, img, (x, y, w, h), edgecolor='green', linewidth=2)
    # image = skimage.io.imread(image_path)
    # width, height = image.shape[0], image.shape[1]
    # bbox_agg = set()
    # [bbox_agg.add(y) for x in [0.25, 0.8] for y in selective_search_bboxwh(image, sigma=x)]
    # [bbox_agg.add(y) for x in [500, 1500] for y in selective_search_bboxwh(image, scale=x)]
    # [bbox_agg.add(y) for x in [2, 25] for y in selective_search_bboxwh(image, rescale=x)]

    # fig2, axes = plt.subplots(2, 4, figsize=(8, 6), frameon=False)
    # xs = [-0.3,-0.4,-0.5,-0.6]
    # img1 = Image.open(image_path)
    # img2 = Image.open(image_path)
    # for ax1, ax2, z in zip(axes[0],axes[1], xs):
    #     bboxes_clustered = []
    #     cluster_colors = []
    #     X = np.array([[bb[0]/width, bb[1]/height, bb[2]/width, bb[3]/height] for bb in bbox_agg])
    #     bbox_colors = np.zeros((len(X), 3))
    #     af = AffinityPropagation(preference=z).fit(X)
    #     labels = af.labels_
    #     for cluster in np.unique(labels):
    #         bboxes_cluster = X[labels == cluster]
    #         cluster_color = np.random.rand(3, )
    #         bbox_colors[labels == cluster] = cluster_color
    #         km = KMeans(n_clusters=1).fit(bboxes_cluster)
    #         closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, bboxes_cluster)
    #         centroid = km.cluster_centers_[0]
    #         bboxes_clustered.append((centroid[0]*width, centroid[1]*height, centroid[2]*width, centroid[3]*height))
    #         cluster_colors.append(cluster_color)
    #
    #     x, y, w, h = gt_bbox[0], gt_bbox[1], gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1]
    #     bboxA = (x, y, w, h)
    #     count = 0
    #     draw_rect(ax1, img1, (x, y, w, h), edgecolor='green', linewidth=2)
    #     draw_rect(ax2, img2, (x, y, w, h), edgecolor='green', linewidth=2)
    #     for i, bbox in enumerate(bboxes_clustered):
    #         iou = bb_intersection_over_union(bboxA, bbox)
    #         draw_rect(ax1, img1, bbox, edgecolor='red', linewidth=1,text=str(iou*100)[:2])
    #         ax1.plot(bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, c=cluster_colors[i], marker='o')
    #         if iou > 0.5:
    #             count += 1
    #             draw_rect(ax2, img2, bbox, edgecolor='red', linewidth=1,text=str(iou*100)[:2].lstrip('0'))
    #             ax2.plot(bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, c=cluster_colors[i], marker='o')
    #     ax1.set_xlabel(str(len(bboxes_clustered)))
    #     ax1.set_ylabel(str(z))
    #     ax2.set_xlabel(str(count))
    #     ax2.set_ylabel(str(z))
    image = skimage.io.imread(image_path)
    width, height = image.shape[1], image.shape[0]
    fig2, axes = plt.subplots(1, 4, figsize=(16, 8), frameon=False)
    x, y, w, h = gt_bbox[0], gt_bbox[1], gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1]
    bboxA = (x, y, w, h)
    bboxes1 = cluster_bboxes(selective_search_bbox_fast(image, int((width * height) / 50)), width, height, preference=-0.25, fast=True)
    bboxes2 = cluster_bboxes(selective_search_bbox_fast(image, int((width * height) / 50)), width, height, preference=-0.3, fast=True)
    bboxes3 = cluster_bboxes(selective_search_bbox_fast(image, int((width * height) / 50)), width, height, preference=-0.35, fast=True)
    bboxes4 = cluster_bboxes(selective_search_bbox_fast(image, int((width * height) / 50)), width, height, preference=-0.4, fast=True)
    bb = [bboxes1, bboxes2, bboxes3, bboxes4]
    for ax, bboex in zip(axes, bb):
        # cl_bb = cluster_bboxes(bboex, width, height)
        ious = []
        for bbox in bboex:
            ious.append(bb_intersection_over_union(bboxA, bbox))
            draw_rect(ax, image, bbox, edgecolor='red', linewidth=1)
            ax.plot(bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, marker='o')
        draw_rect(ax, image, (x, y, w, h), edgecolor='green', linewidth=2)
        ii = ['%.1f' % y for y in [x*100 for x in sorted(ious,reverse=True)]]

        ax.set_xlabel('{}: \n{}'.format(len(bboex), '\n'.join(ii)))
    plt.show()

if __name__ == '__main__':
    global class_names, input_shape, attr_names
    class_names, input_shape, attr_names = init_globals(fashion_dataset_path)
    image_paths = []
    gt_bboxes = []
    with open (os.path.join(prediction_path, 'annotation.txt')) as f:
        for line in f:
            line = line.split()
            image_paths.append(os.path.join(prediction_path,line[0]))
            gt_bboxes.append((int(line[2]), int(line[3]), int(line[4]), int(line[5])))
    import time
    t1 = time.time()
    for img_path, bbox in zip(image_paths, gt_bboxes):
        # display_seg(img_path, bbox)
        image = skimage.io.imread(img_path)
        width, height = image.shape[1], image.shape[0]
        bboxes3 = cluster_bboxes(selective_search_bbox_fast(image, int((width * height) / 50)), width, height,
                                 preference=-0.35, fast=True)
    print('{} secs'.format((time.time()-t1)))
    # t1 = time.time()
    # for img_path, bbox in zip(image_paths, gt_bboxes):
    #     image = Image.open(img_path)
    #     image = skimage.io.imread(img_path)
    #     width, height = image.shape[1], image.shape[0]
    #     bboxes3 = cluster_bboxes(selective_search_bbox_fast(image, int((width * height) / 50)), width, height,
    #                              preference=-0.4, fast=True)
    # print('fast: {} secs'.format((time.time() - t1)))
