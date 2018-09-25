import matplotlib.pyplot as plt
import sys
import caffe
import os
import numpy as np
import scipy
from sklearn.metrics.pairwise import cosine_similarity
import glob
from tqdm import tqdm

# bi onceki fc8'di unutma
#feature_location = 'fc8'
feature_location = 'fc5'#'conv5_6' #'fc7'

def get_feature_of_one_image(imge, net):
    img_0 = caffe.io.load_image(imge)
    net.predict([img_0])
    return net.blobs[feature_location].data
    #return a
    #return


#averaged_value
def get_feature_val(folder_1, folder_2, net):
    #print folder_1
    image_arr_1 = glob.glob(folder_1 + '/*.jpg')
    image_arr_2 = glob.glob(folder_2 + '/*.jpg')
    #print image_arr_1

    feat_arr_1 = np.zeros((10, 512)) #feat_arr_1 = np.zeros((10, 4096))
    feat_arr_2 = np.zeros((10, 512)) #feat_arr_2 = np.zeros((10, 4096))

    cnt = 0
    for img in image_arr_1:
        feat = get_feature_of_one_image(img, net)

        #print "feature shape = ", feat.shape
        feat_arr_1[cnt, :] = feat
        cnt += 1

    #print "len feat arr = ", feat_arr_1.shape
    image_arr_1_feat = np.mean(feat_arr_1, axis=0)#, keepdims=False)

    cnt = 0
    for img in image_arr_2:
        feat = get_feature_of_one_image(img, net)
        #feat = feat.reshape(feat, (1, 10240))
        #print "feature shape = ", feat.shape
        feat_arr_2[cnt, :] = feat
        cnt += 1

    #print "len feat arr = ", feat_arr_2.shape
    image_arr_2_feat = np.mean(feat_arr_2, axis=0)  # , keepdims=False)

    image_arr_1_feat = image_arr_1_feat.reshape(1, -1)
    image_arr_2_feat = image_arr_2_feat.reshape(1, -1)
    #print "feat 1 shape = ", image_arr_1_feat.shape
    #print "feat 2 shape = ", image_arr_2_feat.shape
    return image_arr_1_feat, image_arr_2_feat


def get_cosine_sim(feat_1, feat_2):
    similarity = cosine_similarity(feat_1, feat_2)
    return similarity[0][0]

def isSame(label1, label2):

    if label1 == label2:
        return True
    else:
        return False

def get_our_prediction(sim, th):
    if sim >= th:
        return True
    else:
        return False

def get_max(th_acc_dict):
    max_val = -999
    th_val = -999
    for k, v in th_acc_dict.items():
        if v > max_val:
            max_val = v
            th_val = k

    print "max accuracy = ", max_val
    print "max threshold value = ", th_val

def main():
    validation_location = '/home/uni/Desktop/DAN/dan_updated2209_Folder/validation_images/'

    #PRETRAINED = '/home/uni/Desktop/DAN/vggface_model/VGG_FACE.caffemodel'
    #MODEL_FILE = '/home/uni/Desktop/DAN/vggface_model/VGG_FACE_deploy.prototxt'

    PRETRAINED = '/home/uni/Desktop/DAN/danfeaturemodel/face_model.caffemodel'
    MODEL_FILE = '/home/uni/Desktop/DAN/danfeaturemodel/face_deploy.prototxt'

    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Classifier(MODEL_FILE, PRETRAINED, channel_swap=(2, 1, 0), raw_scale=255)

    #threshold_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    #threshold_values = [0.8]
    threshold_values = np.arange(0.65, 0.9, 0.005)
    print "len threshold = ", len(threshold_values)
    print threshold_values
    th_acc_dict = {}

    true_label = []
    false_label = []
    threshold_acc_dict = {}
    for th in threshold_values:
        print "th = ", th, " started"
        true_prediction = 0
        total_counter = 0
        pbar = tqdm(total=500)

        for fold in os.listdir(validation_location):
            sub_folders = os.listdir(validation_location + fold)
            sub_folder_1 = validation_location + fold + '/' + sub_folders[0]
            sub_folder_2 = validation_location + fold + '/' + sub_folders[1]
            label_1 = os.path.basename(sub_folder_1).split('-')[0]
            label_2 = os.path.basename(sub_folder_2).split('-')[0]
            #print "label 1 = ", label_1
            #print "label 2 = ", label_2
            feature_1, feature_2 = get_feature_val(sub_folder_1, sub_folder_2, net)
            sim_val = get_cosine_sim(feature_1, feature_2)
            #print "sim val = ", sim_val
            #print sim_val
            is_same_bool = isSame(label_1, label_2)
            our_pred = get_our_prediction(sim_val, th)
            #print "our pred = ", our_pred
            #print "is same bool = ", is_same_bool
            if is_same_bool == our_pred:
                true_prediction += 1

            total_counter += 1
            pbar.update(1)
        accuracy = true_prediction/float(total_counter)
        th_acc_dict[th] = accuracy
        print "true prediction count = ", true_prediction
        print "total count = ", total_counter
        print "accuracy = ", accuracy
        print "--------"

    get_max(th_acc_dict)





main()
