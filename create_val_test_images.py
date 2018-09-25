import my_utils.utils as u
import ytf_dataloader as ydl
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import torch
import os
import networks.g_net as g
from scipy import misc



def create_test_images(dl_test, test_out_location, netG):
    for i in range(10):
        for step, (read_image_201, read_image_202, label1, label2) in enumerate(dl_test):

            frames1 = read_image_201
            frames2 = read_image_202

            frames1 = Variable(frames1.cuda())
            frames2 = Variable(frames2.cuda())

            gan_img1 = netG(frames1)
            gan_img2 = netG(frames2)
            gan_img1_save = gan_img1.data[0, ...].cpu().numpy().transpose((1, 2, 0))
            gan_img1_save = (gan_img1_save * 255).astype(int)

            gan_img2_save = gan_img2.data[0, ...].cpu().numpy().transpose((1, 2, 0))
            gan_img2_save = (gan_img2_save * 255).astype(int)

            label1 = label1.replace('/', '-')
            label2 = label2.replace('/', '-')
            label1 = label1.strip()
            label2 = label2.strip()

            folder_name = label1 + '--' + label2 + '/'
            pair_1_folder = folder_name + label1 + '/'
            pair_2_folder = folder_name + label2 + '/'

            image1_name = label1 + '--' + str(i + 1) + '.jpg'
            image2_name = label2 + '--' + str(i + 1) + '.jpg'
            image1_loc = test_out_location + pair_1_folder + image1_name
            image2_loc = test_out_location + pair_2_folder + image2_name

            output_loc_1 = test_out_location + pair_1_folder
            output_loc_2 = test_out_location + pair_2_folder

            if not os.path.exists(output_loc_1):
                os.makedirs(output_loc_1)
            if not os.path.exists(output_loc_2):
                os.makedirs(output_loc_2)
            misc.imsave(image1_loc, gan_img1_save)
            misc.imsave(image2_loc, gan_img2_save)
        print "range = ", i, " finished"


def create_val_images(dl_val, validation_location, netG):
    for i in range(10):
        for step, (read_image_201, read_image_202, label1, label2) in enumerate(dl_val):

            frames1 = read_image_201
            frames2 = read_image_202

            frames1 = Variable(frames1.cuda())
            frames2 = Variable(frames2.cuda())

            gan_img1 = netG(frames1)
            gan_img2 = netG(frames2)
            gan_img1_save = gan_img1.data[0, ...].cpu().numpy().transpose((1, 2, 0))
            gan_img1_save = (gan_img1_save * 255).astype(int)

            gan_img2_save = gan_img2.data[0, ...].cpu().numpy().transpose((1, 2, 0))
            gan_img2_save = (gan_img2_save * 255).astype(int)

            label1 = label1.replace('/', '-')
            label2 = label2.replace('/', '-')
            label1 = label1.strip()
            label2 = label2.strip()

            folder_name = label1 + '--' + label2 + '/'
            pair_1_folder = folder_name + label1 + '/'
            pair_2_folder = folder_name + label2 + '/'

            image1_name = label1 + '--' + str(i + 1) + '.jpg'
            image2_name = label2 + '--' + str(i + 1) + '.jpg'
            image1_loc = validation_location + pair_1_folder + image1_name
            image2_loc = validation_location + pair_2_folder + image2_name

            output_loc_1 = validation_location + pair_1_folder
            output_loc_2 = validation_location + pair_2_folder

            if not os.path.exists(output_loc_1):
                os.makedirs(output_loc_1)
            if not os.path.exists(output_loc_2):
                os.makedirs(output_loc_2)
            misc.imsave(image1_loc, gan_img1_save)
            misc.imsave(image2_loc, gan_img2_save)
        print "range = ", i, " finished"


def main():
    txt_location = '/home/uni/Desktop/DAN/splits.txt'
    #todo change these
    validation_out_location = '/home/uni/Desktop/DAN/dan_updated2209_Folder/validation_images/'
    test_out_location = '/home/uni/Desktop/DAN/dan_updated2209_Folder/test_images/'

    batchSize = 1
    transform = transforms.Compose([transforms.Resize((112, 96)), transforms.ToTensor()])

    # take first split
    # dicriminator ile train edilmis netg
    #change this
    load_model_location = '/home/uni/Desktop/DAN/dan_updated2209_Folder/model_out/updated2209_DanGeneratorYTF-20.pth'
    checkpoint = torch.load(load_model_location)
    netG = g.G_Net()
    netG.load_state_dict(checkpoint)
    netG.train(False)
    netG.cuda()

    for val in range(1, 2):
        train_arr, test_arr, validation_arr = u.read_train_test_val_from_splits(txt_location, 1, 2)
        dl_val = ydl.get_loader(validation_arr, transform, batch_size=1, shuffle=True, num_workers=1)
        dl_test = ydl.get_loader(test_arr, transform, batch_size=1, shuffle=True, num_workers=1)
        print "Validation image creation started"
        create_val_images(dl_val, validation_out_location, netG)
        print "Validation image creation finished"
        print "Test image creation started"
        create_test_images(dl_test, test_out_location, netG)
        print "Test image creation finished"






main()