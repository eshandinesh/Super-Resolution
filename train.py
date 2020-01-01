import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from dataset import Train_Dataset_Prep
from torch.utils.data import DataLoader
from model import Generator, Descriminator

if __name__ == '__main__':

    num_epochs = 100
    batch_size = 8

    crop_size = 128
    upscale_factor = 4

    train_data_set_path = '/content/drive/train'

    train_set = Train_Dataset_Prep(train_data_set_path, crop_size, upscale_factor)
    train_data_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=batch_size, shuffle=True)

    loss = nn.MSELoss()


    loss_desc = nn.MSELoss()

    generator = Generator()

    descriminator = Descriminator()

    optimizerG = optim.Adam(generator.parameters())

    for epoch in range(1, num_epochs + 1):

        train_bar = tqdm(train_data_loader)

        generator.train()

        for lowres, real_img_hr in train_bar:
            generator_generated_image = generator(lowres)

            # Train G
            optimizerG.zero_grad()

            image_loss = loss(generator_generated_image, real_img_hr)

            image_loss.backward()
            optimizerG.step()

            # Print information by tqdm
            train_bar.set_description(f'epoch {epoch}, image_loss {image_loss}')

    # Save model parameters
    torch.save(generator.state_dict(), '/content/drive/generator_epoch_pre_cpu.pth')

    optimizerG = optim.Adam(generator.parameters())
    optimizerD = optim.Adam(descriminator.parameters())

    for epoch in range(1, num_epochs + 1):
        train_bar = tqdm(train_data_loader)

        generator.train()
        descriminator.train()

        for lowres, real_img_hr in train_bar:
            # Train Descriminator

            optimizerD.zero_grad()

            fake = descriminator(generator(lowres))
            real = descriminator(real_img_hr)

            desc_loss = loss_desc(fake, real)

            loss_desc.backward()
            optimizerD.step()

            # Train Generator
            optimizerG.zero_grad()

            gen_image = generator(lowres)
            image_loss = loss(gen_image, real_img_hr)

            desc_res = descriminator(gen_image)
            new_loss = loss_desc(desc_res, real)

            new_loss.backward()
            optimizerG.step()

            # Print information by tqdm
            train_bar.set_description(f'epoch {epoch}, final loss {new_loss}')

    # Save model parameters

    torch.save(generator.state_dict(), '/content/drive/generator_epoch_%d_cpu.pth' % (epoch))
    if epoch % 5 == 0:
        torch.save(descriminator.state_dict(), '/content/drive/descriminator_epoch_%d_cpu.pth' % (epoch))
        torch.save(optimizerG.state_dict(), '/content/drive/optimizerG_epoch_%d_cpu.pth' % (epoch))
        torch.save(optimizerD.state_dict(), '/content/drive/optimizerD_epoch_%d_cpu.pth' % (epoch))


