from random import shuffle
import sys
import glob
import shutil
import os

images_path_name = sorted(glob.glob('*.jpg'))
for img_nm in images_path_name:
	# print('deleting {}'.format(img_nm))
	os.remove(img_nm)
amount = 3
if len(sys.argv) > 1:
	amount = int(sys.argv[1])

part = 'test'
if len(sys.argv) > 2:
	part = str(sys.argv[2])

fashion_dataset_path='fashion_data/'
predict_path = 'prediction/'
class_names = []
with open(fashion_dataset_path + 'Anno/list_category_cloth.txt') as f:
	next(f)
	next(f)
	for line in f:
		class_names.append(line.split()[0])
# print(class_names)
subset = []
counter = 0
with open(fashion_dataset_path + 'Eval/list_eval_partition.txt') as f:
    next(f)
    next(f)
    for line in f:
        counter += 1
        line = line.split()
        partition = line[1]
        if partition != part:
            continue
        subset.append(counter)
shuffle(subset)
# print(subset[:amount])
subset = subset[:amount]
def get_gt_bbox(image_path_name, file_ptr):
    for line in file_ptr:
        if image_path_name in line:
            x1=int(line.split()[1])
            y1=int(line.split()[2])
            x2=int(line.split()[3])
            y2=int(line.split()[4])
            bbox = [x1, y1, x2, y2]
            # logging.debug('bbox {}'.format(bbox))
            return bbox
counter = 0
with open(predict_path+'annotation.txt', 'w') as anno:
	with open(fashion_dataset_path + 'Anno/list_bbox.txt') as file_list_bbox_ptr:
	    with open(fashion_dataset_path + 'Anno/list_category_img.txt') as file_list_category_img:
	        next(file_list_category_img)
	        next(file_list_category_img)
	        for line in file_list_category_img:
	            counter += 1
	            if counter not in subset:
	                continue
	            line = line.split()
	            image_path_name = line[0]
	            image_name = line[0].split('/')[-1]
	            image_category=class_names[int(line[1:][0]) - 1]
	            image_name = image_category+image_name[3:]
	            gt_x1, gt_y1, gt_x2, gt_y2 = get_gt_bbox(image_path_name, file_list_bbox_ptr)
	            shutil.copyfile(fashion_dataset_path + '/Img/' + image_path_name, predict_path+image_name)
	            anno.write(image_name + ' ' + image_category + ' ' + str(gt_x1) + ' ' + str(gt_y1) + ' ' + str(gt_x2) + ' ' + str(gt_y2) + '\n')
	            print(image_name,image_category,gt_x1,gt_y1,gt_x2,gt_y2)
