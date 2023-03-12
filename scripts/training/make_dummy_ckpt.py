from argparse import ArgumentParser

from pytorch_lightning import Trainer

from resolve.common.util import yaml_to_dict
from train import load_model


def main():
    parser = ArgumentParser()
    parser.add_argument('--output', type=str, default='dummy.ckpt')
    parser.add_argument('--encoder', type=str, default='bert-base-uncased')
    parser.add_argument('--head_config', type=yaml_to_dict, default="configs/default_mwe_neg_cross_product.yaml",
                        required=False)
    parser.add_argument('--definition_encoder', type=str, required=False)

    # these args don't matter, but let us reuse training script code to load the model
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--load_model')

    args = parser.parse_args()

    model = load_model(args)
    trainer = Trainer()
    trainer.model = model
    print(f'Writing to {args.output}')
    trainer.save_checkpoint(args.output)
    print('Done')


if __name__ == '__main__':
    main()