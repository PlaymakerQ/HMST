import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run HMPRec.")

    # Data
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--data_name', type=str, default='CA', help='Used data name.')
    parser.add_argument('--min_len', type=int, default=3)
    parser.add_argument('--max_len', type=int, default=101)
    parser.add_argument('--num_region', type=int, default=40)

    # Training

    parser.add_argument('--epoch', type=int, default=200, help='Epoch num.')
    parser.add_argument('--batch', type=int, default=64, help='Training batch size.')
    parser.add_argument('--patience', type=float, default=20, help='Initial learning rate.')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Learning rate.')
    parser.add_argument('--workers', type=int, default=0, help='Epoch num.')

    # Model
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--init_size', type=float, default=1e-3)
    parser.add_argument('--n_dim', type=int, default=64, help='Embedding dimension.')
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--scale', type=float, default=0.1)
    parser.add_argument('--network', type=str, default="resSumGCN")
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.6)  # score weight
    parser.add_argument('--beta', type=float, default=0.3)  # loss weight

    # Save
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--save_args', type=bool, default=True)
    parser.add_argument('--log', type=bool, default=True)

    # TEST mode
    parser.add_argument('--TEST_MODE', type=bool, default=True)

    args = parser.parse_args()

    return args
