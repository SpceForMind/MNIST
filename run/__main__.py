from argparse import ArgumentParser

from run.trainer import main as trainer_main
from run.tester import main as tester_main


def parse_args():
    parser = ArgumentParser(description='MNIST Trainer/Tester')
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('trainer', help='MNIST Trainer')
    train_parser.add_argument('--batch_size',
                        action='store',
                        dest='batch_size',
                        default=200,
                        required=False,
                        type=int)
    train_parser.add_argument('--learning_rate',
                        action='store',
                        dest='learning_rate',
                        default=0.01,
                        required=False,
                        type=float
                        )
    train_parser.add_argument('--epochs',
                        action='store',
                        dest='epochs',
                        default=10,
                        required=False,
                        type=int
                        )
    train_parser.add_argument('--log_interval',
                        action='store',
                        dest='log_interval',
                        default=10,
                        required=False,
                        type=int
                        )
    train_parser.add_argument('--path_to_model',
                        action='store',
                        dest='path_to_model',
                        required=False,
                        type=str
                        )
    train_parser.add_argument('--dir_to_save_model',
                        action='store',
                        dest='dir_to_save_model',
                        required=False,
                        type=str
                        )
    train_parser.set_defaults(func=trainer_main)

    test_parser = subparsers.add_parser('tester', help='MNIST Tester')
    test_parser.add_argument('--batch_size',
                              action='store',
                              dest='batch_size',
                              default=200,
                              required=False,
                              type=int)
    test_parser.add_argument('--path_to_model',
                              action='store',
                              dest='path_to_model',
                              required=True,
                              type=str
                              )
    test_parser.set_defaults(func=tester_main)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
