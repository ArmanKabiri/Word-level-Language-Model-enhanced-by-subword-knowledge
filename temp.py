import argparse

parser = argparse.ArgumentParser(description='LSTM Language Model - Text Generator')
parser.add_argument('--a_str', type=str)
parser.add_argument('--b_int', type=int)
parser.add_argument('--cnn_kernels', type=str, nargs='+', default=['(10,2)', '(30,3)', '(40,4)', '(40,5)'],
                    help="CNN Kernels : (n_kernel,width_kernel). Sample input: (10,2) (30,3) (40,4) (40,5)."
                         "Notice the spaces and parentheses.")
parser.add_argument('--features_level', nargs='+', type=str, default=['word', 'character'],
                    help='Specify the level of features by which you want to represent your words.')
parser.add_argument('--c_str', type=str)

args = parser.parse_args()
args.cnn_kernels = [tuple(map(int, item.replace('(', '').replace(')', '').replace(' ', '').split(','))) for item in
                    args.cnn_kernels]
print('done')
