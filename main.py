import os
import time

import h5py
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from args import parse_argv
from data import Featurizer
from dataset_pafnucy import DatasetFactory
from model_pafnucy import Pafnucy, val, test

if __name__ == '__main__':
    SHOW_PROCESS_BAR = False
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    args = parse_argv()
    prefix = os.path.abspath(args.output_prefix) + '_' + timestamp
    log_dir = os.path.join(os.path.abspath(args.log_dir), os.path.split(prefix)[1])

    # --------- load data ---------

    featurizer = Featurizer()

    feature_names = {name: i for i, name in enumerate(featurizer.FEATURE_NAMES)}
    print(feature_names)

    phases = ['training', 'validation', 'test']

    ids = {phase_name: [] for phase_name in phases}
    affinity = {phase_name: [] for phase_name in phases}
    coords = {phase_name: [] for phase_name in phases}
    features = {phase_name: [] for phase_name in phases}

    for phase_name in phases:
        dataset_path = os.path.join(args.input_dir, '%s_set.hdf' % phase_name)
        with h5py.File(dataset_path, 'r') as f:
            for pdb_id in f:
                dataset = f[pdb_id]

                coords[phase_name].append(dataset[:, :3])
                features[phase_name].append(dataset[:, 3:])
                affinity[phase_name].append(dataset.attrs['affinity'])
                ids[phase_name].append(pdb_id)

        ids[phase_name] = np.array(ids[phase_name])
        affinity[phase_name] = np.reshape(affinity[phase_name], (-1, 1))

    dataset_factory = DatasetFactory(coords, features, affinity, args.grid_spacing,
                                     args.max_dist, feature_names, args.fp16)
    data_loaders = {phase_name:
                        DataLoader(dataset_factory.get_dataset(phase_name, args.rotations),
                                   batch_size=args.batch_size,
                                   pin_memory=True,
                                   num_workers=4,
                                   shuffle=True)
                    for phase_name in phases}

    # --------- prepare ---------

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f'device = {device}')
    model = Pafnucy(conv_patch=args.conv_patch,
                    pool_patch=args.pool_patch,
                    conv_channels=args.conv_channels,
                    dense_sizes=args.dense_sizes,
                    keep_prob=args.kp)  # default settings

    if args.fp16:
        model.half()

    model = model.to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.lmbda)
    loss_function = nn.MSELoss(reduction='sum')
    writer = SummaryWriter(log_dir)

    # --------- train ---------

    g_step = 0
    for epoch in range(args.num_epochs):
        print(f'epoch {epoch}')
        data_loader = data_loaders['training']
        tbar = tqdm(enumerate(data_loader), total=len(data_loader), disable=not SHOW_PROCESS_BAR)
        for idx, (data, target) in tbar:
            model.train()
            if args.fp16:
                data = torch.HalfTensor(np.vstack(data))
                target = torch.HalfTensor(target)
            else:
                data = torch.Tensor(np.vstack(data))
                target = torch.Tensor(target)

            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output.view(-1), target.view(-1))
            loss.backward()
            optimizer.step()

            tbar.set_description(f' * Train Epoch {epoch} '
                                 f'Loss={loss.item() / (args.batch_size * len(args.rotations)):.3f}')

            if (g_step + 1) % 1000 == 0:
                writer.add_scalar(f'training loss', loss.item() / (args.batch_size * len(args.rotations)),
                                  global_step=g_step)
                writer.add_scalar(f'validation loss',
                                  val(model, data_loaders['validation'], loss_function, device, len(args.rotations)),
                                  global_step=g_step)

                e = test(model, data_loaders['test'], loss_function, device)
                for key in e:
                    writer.add_scalar(key, e[key], global_step=g_step)

            g_step += 1

    # --------- evaluate ---------

    for phase in ['validation', 'test']:
        result_file = os.path.join(log_dir, f'{phase}_result.txt')
        with open(result_file, 'w') as f:
            results = test(model, data_loaders[phase], loss_function, device, len(args.rotations))
            for k, v in results.items():
                f.write(f'{k}: {v}\n')

    torch.save(model.state_dict(), 'model_pafnucy.pt')

    print('test finished')
