import argparse

from infosol.evaluate import EvaluateBartEditor, EvaluateBertEditor, EvaluateBartS2S, EvaluateBartLarge, EvaluateNoModel

if __name__ == '__main__':

    meta_parser = argparse.ArgumentParser()
    meta_parser.add_argument('--args_path', type=str)
    meta_parser.add_argument('--cuda_device', type=int)
    meta_args = meta_parser.parse_args()

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_barteditor = subparsers.add_parser('BartEditor')
    parser_barteditor.set_defaults(func=EvaluateBartEditor)
    EvaluateBartEditor.add_args(parser_barteditor)

    parser_berteditor = subparsers.add_parser('BertEditor')
    parser_berteditor.set_defaults(func=EvaluateBertEditor)
    EvaluateBertEditor.add_args(parser_berteditor)

    parser_barts2s = subparsers.add_parser('BartS2S')
    parser_barts2s.set_defaults(func=EvaluateBartS2S)
    EvaluateBartS2S.add_args(parser_barts2s)

    parser_bartlarge = subparsers.add_parser('BartLargeEditor')
    parser_bartlarge.set_defaults(func=EvaluateBartLarge)
    EvaluateBartLarge.add_args(parser_bartlarge)

    parser_baseline = subparsers.add_parser('Baseline')
    parser_baseline.set_defaults(func=EvaluateNoModel)
    EvaluateNoModel.add_args(parser_baseline)

    with open(meta_args.args_path, 'rt') as f:
        for i,line in enumerate(f):
            print(f'#### On job {i} ####')
            args_list = line.strip().split(' ')
            args_list.extend(['--cuda_device', str(meta_args.cuda_device)])
            args = parser.parse_args(args_list)
            eval_instance = args.func()
            eval_instance.setup(args)
            eval_instance.run()
