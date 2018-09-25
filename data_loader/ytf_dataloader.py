import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
import os

class YTFDataset(data.Dataset):
    def __init__(self, input_arr, transform, data_loc='/home/uni/Desktop/DAN/aligned_images_DB'):
        temp_arr = []
        temp_label_arr = []
        for val in input_arr:
            label_val = val.split('/')[0]
            val = val.strip()
            label_val = label_val.strip()
            temp_arr.append(data_loc + '/' + val)
            temp_label_arr.append(label_val)

        # input array is like  ['/home/uni/Desktop/DAN/aligned_images_DB/Lucio_Stanca/3', '/home/uni/Desktop/DAN/aligned_images_DB/Lucio_Stanca/4', ...]
        self.input_arr = temp_arr
        self.input_label = temp_label_arr
        self.transform = transform
        print "YTF data loader started"
        print "train input len = ", len(self.input_arr)
        print "train label len = ", len(self.input_label)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        #select one person image folder
        image_folder = os.listdir(self.input_arr[index])
        #get label
        image_folder_label = self.input_label[index]
        #read 20 images of the person
        read_img_count = 20

        read_image_20 = torch.zeros((read_img_count, 3, 112, 96))

        #create 20 random indexes
        imageShuffle = np.random.randint(len(image_folder), size=read_img_count)
        #get random 20 images
        for i in range(read_img_count):
            image_location = self.input_arr[index] + '/' + image_folder[imageShuffle[i]]
            read_image = Image.open(image_location).convert('RGB')
            read_image_20[i, ...] = self.transform(read_image)

        #select one random index out of range(0, 20)
        tpidx = np.random.randint(read_img_count)
        # select one original image out of 20 options
        A = read_image_20[tpidx, ...]
        read_image_20 = read_image_20.view(read_img_count * 3, 112, 96)
        return A, read_image_20, image_folder_label

    def __len__(self):
        return len(self.input_label)

#since data set read is arranged with respect to folder based rather person based. For instance; it has X-1, Y-2, X-2 etc.
#therefore there might be a person's different image folders more than once such as X-1, X-2
def collate_fn(data):
    A, read_image_20, image_folder_label = zip(*data)
    A = torch.stack(A)
    read_image_20 = torch.stack(read_image_20)

    return A, read_image_20, image_folder_label



def get_loader(train_arr, transform, batch_size, shuffle, num_workers):
    ytfDataset = YTFDataset(train_arr, transform)
    data_loader = data.DataLoader(dataset=ytfDataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return data_loader