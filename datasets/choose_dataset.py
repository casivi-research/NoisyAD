from datasets.mvtec import MVTecDataset
from datasets.mpdd import MPDDDataset
from datasets.visa import VisADataset


def choose_datasets(args, train_class, noise_type=None, noisy_p=0):
    if args.data_type == 'mvtec':
        train_dataset = MVTecDataset(dataset_path=args.data_path,
                                     class_name=train_class,
                                     is_train=True,
                                     resize=256,
                                     cropsize=args.size_crops[0],
                                     noise_type=None,
                                     p=0)

        test_dataset = MVTecDataset(dataset_path=args.data_path,
                                    class_name=train_class,
                                    is_train=False,
                                    resize=256,
                                    cropsize=args.size_crops[0],
                                    noise_type=noise_type,
                                    p=noisy_p)
    elif args.data_type == 'mpdd':
        train_dataset = MPDDDataset(dataset_path=args.data_path,
                                     class_name=train_class,
                                     is_train=True,
                                     resize=256,
                                     cropsize=args.size_crops[0],
                                     noise_type=None,
                                     p=0)

        test_dataset = MPDDDataset(dataset_path=args.data_path,
                                    class_name=train_class,
                                    is_train=False,
                                    resize=256,
                                    cropsize=args.size_crops[0],
                                    noise_type=noise_type,
                                    p=noisy_p)
    elif args.data_type == 'visa':
        train_dataset = VisADataset(dataset_path=args.data_path,
                                    class_name=train_class,
                                    is_train=True,
                                    resize=256,
                                    cropsize=args.size_crops[0],
                                    noise_type=None,
                                    p=0)

        test_dataset = VisADataset(dataset_path=args.data_path,
                                   class_name=train_class,
                                   is_train=False,
                                   resize=256,
                                   cropsize=args.size_crops[0],
                                   noise_type=noise_type,
                                   p=noisy_p)
    else:
        raise NotImplementedError('unsupport dataset type')

    return train_dataset, test_dataset
