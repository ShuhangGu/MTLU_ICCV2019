from importlib import import_module
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import default_collate

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:

            module_train = import_module('data.' + args.data_train)
            print(module_train)
            trainset = getattr(module_train, args.data_train)(args)
            print(trainset)
            self.loader_train = dataloader.DataLoader(
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )

        self.loader_test = []

        if args.data_test in ['Set5Y_x4','Set5_x4','DIV2K_valid20_x4','Set6_x4','Set6_x2']:

            module_test = import_module('data.benchmarksr')
            testset = getattr(module_test, 'BenchmarkSR')(args, train=False)

        elif args.data_test in ['DenoiseSet68','DenoiseColorSet68','DenoiseGrayDIV2K','DenoiseColorDIV2K','DenoiseColorDIV2K501','DenoiseColorDIV2K502']:

            module_test = import_module('data.benchmarkdenoise')
            testset = getattr(module_test, 'BenchmarkDenoise')(args, train=False, name = args.data_test)
        else:
            module_test = import_module('data.' +  args.data_test)
            testset = getattr(module_test, args.data_test)(args, train=False)

        self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )
