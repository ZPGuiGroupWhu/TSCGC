import argparse
parser =argparse.ArgumentParser(description='deep clustering', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--name",type=str,default="citeseer")
parser.add_argument("--seed",type=int,default=405)
parser.add_argument("--lr",type=float,default=1e-3)
parser.add_argument("--temperature_f",type=float,default=0.5)
parser.add_argument("--epoch",type=int,default=400)
parser.add_argument("--n_clusters",type=int,default=7)
parser.add_argument("--n_z",type=int,default=10)
parser.add_argument("--n_input",type=int,default=1000)
parser.add_argument("--alpha",type=float,default=0.1)
parser.add_argument("--beta",type=float,default=1.0)
parser.add_argument('--data_path', type=str, default='.txt')
parser.add_argument('--label_path', type=str, default='.txt')
parser.add_argument('--save_path', type=str, default='.txt')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--device', type=str, default='cuda:0')

# clustering performance: acc, nmi, ari, f1
parser.add_argument('--acc', type=float, default=0)
parser.add_argument('--nmi', type=float, default=0)
parser.add_argument('--ari', type=float, default=0)
parser.add_argument('--f1', type=float, default=0)

args =parser.parse_args()



