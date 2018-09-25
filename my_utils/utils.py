import torch.nn as nn
import networks.g_net as g
import networks.d_net as d
import torch.optim as optim
import datetime
from scipy import misc
import torch
import os

def rearrange_data(test_split):
    test_arr = []
    for data in test_split:
        first_celeb = data.split(',')[2]
        second_celeb = data.split(',')[3]
        test_arr.append(first_celeb)
        test_arr.append(second_celeb)
    return test_arr


def read_train_test_val_from_splits(split_loc, test_set_counter, validation_set_counter):
    test_arr = []
    train_arr = []
    validation_arr = []
    with open(split_loc) as f:
        for line in f:
            split_numb = line.split(',')[0]
            # print split_numb
            if int(split_numb) == test_set_counter:
                test_arr.append(line)
            elif int(split_numb) == validation_set_counter:
                validation_arr.append(line)
            else:
                train_arr.append(line)

    return rearrange_data(train_arr), rearrange_data(test_arr), rearrange_data(validation_arr)


def weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)


def load_gnet():
    net_g = g.G_Net()
    net_g.apply(weights_init)
    net_g.cuda()
    L2criterion = nn.MSELoss()
    L2criterion.cuda()
    optimizer_g = optim.Adam(net_g.parameters(), lr=1e-4, betas=(0.9, 0.999))
    return net_g, L2criterion, optimizer_g

def load_dnet():
    net_d = d.D_Net()
    net_d.apply(weights_init)
    net_d.cuda()
    L1Loss = nn.L1Loss()
    L1Loss.cuda()
    optimizer_d = optim.Adam(net_d.parameters(), lr=1e-4, betas=(0.9, 0.999))
    return net_d, L1Loss, optimizer_d


def log(x):
    return torch.log(x + 1e-8)

def get_experiment_folder(root_location, given_exp_name):
    x = datetime.datetime.now()
    today_exp = str(x.day) + '-' + str(x.month) + '-' + str(x.year)
    experiment_name = today_exp + given_exp_name
    experiment_name = os.path.join(root_location, 'experiments', experiment_name)
    return experiment_name


def create_folders(experiment_loc):
    generator_output_location = os.path.join(experiment_loc, 'generator_output')
    disc_gen_output_location = os.path.join(experiment_loc, 'disc_gen_output')
    model_location = os.path.join(experiment_loc, 'model_output')
    test_images_loc = os.path.join(experiment_loc, 'test_images')
    validation_images_loc = os.path.join(experiment_loc, 'validation_images')
    if not os.path.exists(experiment_loc):
        os.makedirs(experiment_loc)
    if not os.path.exists(generator_output_location):
        os.makedirs(generator_output_location)
    if not os.path.exists(disc_gen_output_location):
        os.makedirs(disc_gen_output_location)
    if not os.path.exists(model_location):
        os.makedirs(model_location)
    if not os.path.exists(test_images_loc):
        os.makedirs(test_images_loc)
    if not os.path.exists(validation_images_loc):
        os.makedirs(validation_images_loc)
    return generator_output_location, disc_gen_output_location, model_location, test_images_loc, validation_images_loc


def save_images(fake, output_location, epoch):
    fake_img = fake.data[5, ...].cpu().numpy().transpose((1, 2, 0))
    fake_img = (fake_img * 255).astype(int)
    misc.imsave(os.path.join(output_location, str(epoch) + '.jpg'), fake_img)
