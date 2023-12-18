import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import data_utils
import sampler_utils
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import conv_model

# Training Function
def train_model(model, dataset_path='output_dataset.h5', input_size=40, 
                batch_size=64, epochs=30, vol_coeff=1.0, 
                iter_sampler=sampler_utils.uniform_sampler(),
                summary_prefix='summary', save_prefix='trained_models'):
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter(log_dir=os.path.join(summary_prefix, f'VOL_COEFF={vol_coeff}'))

    for epoch in range(epochs):
        total_loss = 0
        total_conf_loss = 0
        total_vol_loss = 0
        num_batches = 0

        for x, y in data_utils.DatasetIterator(dataset_path, batch_size, iter_sampler):
            x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
            x = x.permute(0, 3, 1, 2)
            y = y.permute(0, 3, 1, 2)

            optimizer.zero_grad()
            outputs = model(x)
            conf_loss = criterion(outputs, y)
            vol_loss = torch.square(torch.mean(y - torch.sigmoid(outputs)))
            loss = conf_loss + vol_coeff * vol_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_conf_loss += conf_loss.item()
            total_vol_loss += vol_loss.item()
            num_batches += 1

            writer.add_scalar('conf_loss', conf_loss.item(), epoch)
            writer.add_scalar('vol_loss', vol_loss.item(), epoch)
            writer.add_scalar('loss', loss.item(), epoch)

        avg_loss = total_loss / num_batches
        avg_conf_loss = total_conf_loss / num_batches
        avg_vol_loss = total_vol_loss / num_batches

        print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Conf Loss: {avg_conf_loss:.4f}, Vol Loss: {avg_vol_loss:.4f}')

    model_path = os.path.join(save_prefix, f'VOL_COEFF={vol_coeff}')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model.state_dict(), os.path.join(model_path, 'model.pth'))


if __name__ == '__main__':
    parser = ArgumentParser()

    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--dataset-path', type=str, dest='dataset_path', 
                        help='path to `.h5` dataset', required=False)

    parser.add_argument('--input-size', type=int, dest='input_size', 
                        help='size of the input tensor', default=40)

    parser.add_argument('--batch-size', type=int, dest='batch_size', 
                        help='size of a minibatch', default=64)

    parser.add_argument('--epochs', type=int, dest='epochs', 
                        help='number of training epochs', default=30)

    parser.add_argument('--vol-coeff', type=float, dest='vol_coeff', 
                        help='volume constraint coefficient in total loss', 
                        default=1.0)

    parser.add_argument('--iter-sampler', type=str, dest='iter_sampler', 
                        help='iteration sampler. Either "uniform" or "poisson_LAM"\n'
                        'LAM: Lambda parameter in Poisson distribution', 
                        default='uniform')

    parser.add_argument('--summary-prefix', type=str, dest='summary_prefix', 
                        help='root folder to save the summary', 
                        default='summary')

    parser.add_argument('--save-prefix', type=str, dest='save_prefix', 
                        help='root folder to save the model', 
                        default='trained_models')

    
    options = parser.parse_args()

    # Model initialization
    model = conv_model.build()

    # Training
    train_model(model, 
                dataset_path='output_dataset.h5', 
                input_size=options.input_size, 
                batch_size=options.batch_size, 
                epochs=options.epochs, 
                vol_coeff=options.vol_coeff, 
                iter_sampler=sampler_utils.parse_sampler(options.iter_sampler),
                summary_prefix=os.path.join(options.summary_prefix, options.iter_sampler), 
                save_prefix=os.path.join(options.save_prefix, options.iter_sampler))
