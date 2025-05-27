import os
import numpy as np
import logging
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from args_get import parameter_parser
from models.model import HMGRec as MODEL
from dataset import POIDataset
from runner import Runner
from utils.tools import MyTool



if __name__ == '__main__':

    # ================
    # initial settings
    # ================

    args = parameter_parser()

    if args.TEST_MODE:
        args.save = False
        args.save_args = False
        args.log = False

    model_name = MODEL.__name__
    args.model_name = model_name
    save_path = MyTool.set_save_path(args.data_name, args.model_name)

    MyTool.set_logging(save_path, args.model_name)

    logging.info(f"seed: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    else:
        args.device = torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ===============================
    # Load Dataset, Runner, and Model
    # ===============================

    dataset_path = 'dataset'
    data = POIDataset(args.data_name, args.min_len, args.max_len)

    train_loader = DataLoader(data.traj_dict['train'], batch_size=args.batch, shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=args.workers, collate_fn=lambda x: x)
    val_loader = DataLoader(data.traj_dict['val'], batch_size=args.batch, shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=args.workers, collate_fn=lambda x: x)
    test_loader = DataLoader(data.traj_dict['test'], batch_size=args.batch, shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=args.workers, collate_fn=lambda x: x)

    args = data.set_nums(args)

    # process auxiliary POI information
    logging.info(f"start splitting regions...")
    region_labels = data.get_region_information(num_region=args.num_region)
    logging.info(f"splitting regions done...")
    cat_labels = data.get_category_information()
    args.num_cat = cat_labels.max().item()

    logging.info("build user-POI and POI-POI edges...")
    up_edges = data.get_user_poi_edges()
    pp_edges = data.get_poi_poi_edges()
    logging.info("user-POI and POI-POI edges done...")

    args.k_list = [1, 5, 10]
    args.required_metrics = ['Acc', 'MRR']
    args.model_params = MyTool.load_model_params(args.data_name)

    model = MODEL(args)
    run = Runner(args, model, region_labels, cat_labels, up_edges, pp_edges)

    # ============
    # Main Process
    # ============
    check_patience = 0
    for epoch in range(args.epoch):

        run.train(train_loader)

        metric_dict = run.valid(val_loader)
        logging.info(f"epoch {epoch + 1:>03d} valid:")
        MyTool.print_metrics(metric_dict)

        metric_dict = run.test(test_loader)
        logging.info(f"epoch {epoch + 1:>03d} test (patience={check_patience}):")
        MyTool.print_metrics(metric_dict)

        # valid result check
        check_result = run.check_best_result(metric_dict)
        if check_result:
            logging.info(f"update valid score at epoch {run.best_epoch}")
            check_patience = 0  # reset
        else:
            check_patience += 1
            if check_patience > args.patience:
                break

        run.current_epoch += 1

    best_result = run.best_metric
    logging.info(f"final result at epoch {run.best_epoch}")
    MyTool.print_metrics(best_result)



