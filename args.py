import argparse
import os

phases = ['training', 'validation', 'test']


def parse_argv():
    def input_dir(path):
        """Check if input directory exists and contains all needed files"""

        path = os.path.abspath(path)
        if not os.path.isdir(path):
            raise IOError('Incorrect input_dir specified: no such directory')
        for dataset_name in phases:
            dataset_path = os.path.join(path, '%s_set.hdf' % dataset_name)
            if not os.path.exists(dataset_path):
                raise IOError('Incorrect input_dir specified:'
                              ' %s set file not found' % dataset_path)
        return path

    parser = argparse.ArgumentParser(
        description='Train 3D convolution neural network on affinity data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    io_group = parser.add_argument_group('I/O')
    io_group.add_argument('--input_dir', '-i', required=True, type=input_dir,
                          help='directory with training, validation and test sets')
    io_group.add_argument('--log_dir', '-l', default='./log_dir/',
                          help='directory to store tensorboard summaries')
    io_group.add_argument('--output_prefix', '-o', default='./output',
                          help='prefix for checkpoints, predictions and plots')
    io_group.add_argument('--grid_spacing', '-g', default=1.0, type=float,
                          help='distance between grid points')
    io_group.add_argument('--max_dist', '-d', default=10.0, type=float,
                          help='max distance from complex center')

    arc_group = parser.add_argument_group('Netwrok architecture')
    arc_group.add_argument('--fp16', default=False, type=int,
                           help='use float16')  # todo
    arc_group.add_argument('--conv_patch', default=5, type=int,
                           help='patch size for convolutional layers')
    arc_group.add_argument('--pool_patch', default=2, type=int,
                           help='patch size for pooling layers')
    arc_group.add_argument('--conv_channels', metavar='C', default=[64, 128, 256],
                           type=int, nargs='+',
                           help='number of fileters in convolutional layers')
    arc_group.add_argument('--dense_sizes', metavar='D', default=[1000, 500, 200],
                           type=int, nargs='+',
                           help='number of neurons in dense layers')

    reg_group = parser.add_argument_group('Regularization')
    reg_group.add_argument('--keep_prob', dest='kp', default=0.5, type=float,
                           help='keep probability for dropout')
    reg_group.add_argument('--l2', dest='lmbda', default=0.001, type=float,
                           help='lambda for weight decay')
    reg_group.add_argument('--rotations', metavar='R', default=list(range(24)),
                           type=int, nargs='+',
                           help='rotations to perform')

    tr_group = parser.add_argument_group('Training')
    tr_group.add_argument('--learning_rate', '-lr', default=1e-5, type=float,
                          help='learning rate')
    tr_group.add_argument('--batch_size', default=5, type=int,
                          help='batch size')
    tr_group.add_argument('--num_epochs', default=20, type=int,
                          help='number of epochs')
    # tr_group.add_argument('--num_checkpoints', dest='to_keep', default=10, type=int,
    #                       help='number of checkpoints to keep')

    args = parser.parse_args()

    return args


# licence
"""BSD 3-Clause License

Copyright (c) 2018, cheminfIBB
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""