# DEEP FASHION

### Setup Environment
```sh
# Virtual environment (optional)
sudo apt install -y virtualenv
# Tensorflow (optional)
sudo apt-get install python-pip python-dev python-virtualenv
virtualenv --system-site-packages -p python3 tensorflow
source tensorflow/bin/activate

# Dependencies
pip install -r requirements.txt 
```

### Download DeepFashion Dataset 
```sh
# http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html
./dataset_download.sh

# The directory structure after downloading and extracting dataset:
# fashion_data/
# ---Anno
# ------list_attr_cloth.txt
# ------list_attr_img.txt
# ------list_bbox.txt
# ------list_category_cloth.txt
# ------list_category_img.txt
# ------list_landmarks.txt
# ---Eval
# ------list_eval_partition.txt
# ---Img
# ------img
```

### Create Dataset
```sh
# For images in fashion_data, apply selective search algo to find ROI/bounding boxes. Crop and copy these ROI inside dataset
python dataset_create.py - If runned dataset split 85/15 created and saved to train85.txt and validation85.txt. Class and attribute weights saved for futher calculations 
```

### Create bottleneck 
```sh
python create_bottleneck.py - Make *npz files of dataset of immidiate results of prediction VGG16 for faster training
```

### Train
```sh
python train.py
```

### Predict
```sh
python predict.py
```

### Misc
prediction	- Contains images used for testing.

output	- Contains trained weights and bottleneck features.

### MODEL
```sh
									->	Bbox Head	(x1, y1, x2, y2) coordinates of surounding bbox

InputImage	->	VGG16 + Layers	--	->  Attribute Head (Attributes)	

									-> Classification Head (Categories)

```

### RESULTS
![alt text](https://raw.githubusercontent.com/v1g1l3m/fashionnet/master/prediction/results/Blazer_00000053.jpg "Prediction")


### Acknowledgment
- [DeepFashion Dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)



