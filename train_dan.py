import my_utils.utils as u
import ytf_dataloader
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import torch
import os

def main():
    split_location = '/home/uni/Desktop/DAN/splits.txt'
    root_location = './'
    batch_size = 16
    training_epoch_count = 1#20
    test_split = 1
    validation_split = 2
    train_first_generator = True
    experiment_location = u.get_experiment_folder(root_location, '_20epoch')
    net_g_output_name = 'net_g_ytf_' + str(training_epoch_count) + '.pth'
    net_d_output_name = 'net_d_ytf_' + str(training_epoch_count) + '.pth'
    generator_output_location, disc_gen_output_location, model_location = u.create_folders(experiment_location)
    train_arr, test_arr, validation_arr, til, vil = u.read_train_test_val_from_splits(split_location, test_split, validation_split)
    transform = transforms.Compose([transforms.Resize((112, 96)), transforms.ToTensor()])
    print "len train arr = ", len(train_arr)
    print "len test arr = ", len(test_arr)
    print "validation arr = ", len(validation_arr)
    dl = ytf_dataloader.get_loader(train_arr, transform, batch_size=batch_size, shuffle=True, num_workers=1)

    net_g, L2criterion, optimizer_g = u.load_gnet()

    #-- generator training started --
    if train_first_generator:
        for epoch in range(training_epoch_count):
            step_loss_arr = []
            for step, (A, read_image_20, image_folder_label) in enumerate(dl):
                #A.shape --> #(16, 3, 112, 96)
                frames = Variable(read_image_20.cuda())
                real_image = Variable(A.cuda())
                #create one fake image using 20 real images
                #feed with 16(batchsize) different people each of them has 20 images
                fake = net_g(frames) #(16, 3, 112, 96)
                loss = L2criterion(fake, real_image)

                net_g.zero_grad()
                loss.backward()
                optimizer_g.step()
                step_loss_arr.append(loss.item())
            print "Epoch:[", epoch, " : ", training_epoch_count, "] average loss = ", str(np.mean(step_loss_arr))
            u.save_images(fake, generator_output_location, epoch)
            if (epoch + 1) == training_epoch_count:
                torch.save(net_g.state_dict(), os.path.join(model_location, net_g_output_name))

    # --generator training ended --



    # -- discriminator generator training started --

    net_d, L1Loss, optimizer_d = u.load_dnet()

    net_g_checkpoint = torch.load(os.path.join(model_location, net_g_output_name))
    net_g.load_state_dict(net_g_checkpoint)
    ganfactor = 0.01

    for epoch in range(training_epoch_count):
        for step, (A, read_image_20, image_folder_label) in enumerate(dl):

            #train with real
            #16 images
            net_d.zero_grad()
            net_g.zero_grad()
            inputv = Variable(A.cuda())
            output = net_d(inputv)

            error_d_real = -ganfactor * torch.mean(u.log(output))
            error_d_real.backward()
            optimizer_d.step()
            d_x = output.data.mean()

            #train with fake
            net_d.zero_grad()
            net_g.zero_grad()
            frames = Variable(read_image_20.cuda())
            fake = net_g(frames)
            output_fake = net_d(fake)
            GD_x = output_fake.data.mean()

            err_d_fake = -ganfactor * torch.mean(u.log(1-output_fake))
            err_d_fake.backward(retain_graph=True)
            optimizer_d.step()


        for step, (A, read_image_20, image_folder_label) in enumerate(dl):
            net_g.zero_grad()
            net_d.zero_grad()

            fake = net_g(Variable(read_image_20.cuda()))
            output = net_d(Variable(fake))

            err_g_fake = -ganfactor * torch.mean(u.log(output))
            L1_recons = L1Loss(fake, Variable(A.cuda()))
            err_g_fake.backward()
            optimizer_g.step()

        print('[%d/%d] D(x): %.4f D(G(x)): %.4f L1_loss: %.4f' % (epoch, training_epoch_count, d_x, GD_x, L1_recons))
        u.save_images(fake, disc_gen_output_location, epoch)
        if (epoch + 1) == training_epoch_count:
            torch.save(net_g.state_dict(), os.path.join(model_location, net_d_output_name))

    # -- discriminator generator training ended --








main()