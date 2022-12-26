import argparse
import os
import time
import warnings
import json

import torch
from transformers import *
from datasets import load_dataset, load_metric
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from evaluate import load
# from transformers.src.transformers.bfp_training.util.bfp.bfp_config import BfpConfig

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def set_reproducibility(random_seed):
    # set random number seeds
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # multi-GPU'

    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_project_name(args):
    config = args.config
    if config.use_bfp and config.bfp_M_Bit is None:
        raise ValueError(f"in BFP training, mantissa bits must be set")

    arch = config.architectures[0]
    if config.use_bfp:
        if config.PrecisionFlag == 0:
            data_format = f"fp_m{config.bfp_M_Bit}"
        else:
            data_format = f"bfp_m{config.bfp_M_Bit}"
    else:
        data_format = "fp32"

    model_info = f"{arch}-{data_format}"

    lr = f"lr{args.learning_rate}"
    adam_info = f"adam-{lr}"

    dataset_type = args.dataset_type
    batch_size = f"global_batch{args.batch_size}"
    is_dist = "multi_gpus" if args.multiprocessing_distributed else "single_gpu"
    dataset_info = f"{dataset_type}-{batch_size}-{is_dist}"

    if args.seed is None:
        seed_info = "no_seed"
    else:
        seed_info = f"seed{args.seed}"

    if config.use_bfp:
        if config.use_flex_bfp and config.is_fast:
            raise ValueError("options Flex BFP and FAST cannot be set at the same time")

    if config.use_flex_bfp:
        bfp_info = "-flex-bfp"
    elif config.is_fast:
        bfp_info = "-fast"
    else:
        bfp_info = ""

    st_info = f"st_{'f' if config.f_st else ''}{'w' if config.w_st else ''}{'a' if config.a_st else ''}"

    if not config.use_bfp:
        st_info = "no_st"

    thres_info = f"thres_{'f' if config.f_thres else ''}{'w' if config.w_thres else ''}{'a' if config.a_thres else ''}{config.threshold}"

    if (config.f_thres == False and config.w_thres == False and config.a_thres == False) or not config.use_bfp:
        thres_info = "no_thres"

    if config.use_multi_exp:
        if config.threshold:
            if config.use_shift:
                multi_exp_info = f"multi_exp_shifted_thres{config.threshold}-"
            else:
                multi_exp_info = f"multi_exp_thres{config.threshold}-"
        else:
            raise ValueError(f"must set threshold for supporting multi exponent")
    else:
        multi_exp_info = ""
    group_size = config.group_size

    return f"{multi_exp_info}{st_info}-{thres_info}-{model_info}-{adam_info}-{dataset_info}-{seed_info}{bfp_info}-g{group_size}-epochs{args.training_epochs}"

def train(gpu, ngpus_per_node, args):
    if args.seed is not None:
        set_reproducibility(args.seed) # need to import later

    args.gpu = gpu
    print('Use GPU: {} for training'.format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.distributed, rank=args.rank)
    # device = torch.device('cuda:{:d}'.format(args.rank))

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_name)
    # model = BertForSequenceClassification.from_pretrained(args.pretrained_name, config=args.config_path, num_labels = 2, 
    #                                                     problem_type = "single_label_classification")
    model = BertForSequenceClassification.from_pretrained(args.pretrained_name, config=args.config_path, num_labels = 2,)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per DistributedDataParallel, 
            # we need to divide the batch size ourselves based on the total number of GPUs of the current node.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        print("It is recommended to use GPU for Transformer training!!")
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    dist_but_rank_0 = args.multiprocessing_distributed and args.rank % ngpus_per_node == 0

    # Prepare Dataset & Model
    train_dataset = load_dataset("glue", args.dataset_type, split="train")
    validation_dataset = load_dataset("glue", args.dataset_type, split="validation")
    if args.distributed:
        train_dist_sampler = DistributedSampler(train_data)
    else:
        train_dist_sampler = None
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=(train_dist_sampler is None), 
                                  sampler=train_dist_sampler,
                                  worker_init_fn=seed_worker, 
                                  num_workers=args.workers, 
                                  pin_memory=True, 
                                  drop_last=True,)
    validation_dataloader = DataLoader(validation_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=False,
                                  worker_init_fn=seed_worker, 
                                  num_workers=args.workers, 
                                  pin_memory=True, 
                                  drop_last=True,)
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.learning_rate)
    softmax = torch.nn.Softmax(dim=1)
    loss_func = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    glue_metric = load('glue', args.dataset_type)

    model.train()
    step = 0

    # if not os.path.exists(args.save_path):
    #     os.mkdir(args.save_path)
    # BfpConfig.use_bfp = False if args.precision_flag == 0 else True

    initiated = False
    for epoch in range(args.training_epochs):
        with open(args.log_path, "a") as f:
            f.write("\n=== Epoch {} ===\n".format(epoch))
        
        if args.distributed:
            train_dist_sampler.set_epoch(epoch)

        # end = time.time()
        # max_length = 0
        for i, batch in enumerate(train_dataloader): # batch: 'sentence', 'label', 'idx'
            # Tokenize the sentence and get the model output
            inputs = tokenizer(batch['sentence'], padding=args.pad_option, max_length=args.max_length, truncation=True, return_tensors="pt")
            # print(inputs['input_ids'].shape)
            # if max_length < inputs['input_ids'].shape[1]:
            #     max_length = inputs['input_ids'].shape[1]
            # if step > 0:
            #     continue
            if (args.gpu is not None) or torch.cuda.is_available():
                outputs = model(inputs['input_ids'].cuda(args.gpu, non_blocking=True), inputs['attention_mask'].cuda(args.gpu, non_blocking=True), inputs['token_type_ids'].cuda(args.gpu, non_blocking=True), labels=batch['label'].cuda(args.gpu, non_blocking=True))
            else:
                outputs = model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'], labels=batch['label'])

            if not initiated:
                initiated = True
                continue
            # Update the model
            # loss = loss_func(outputs.logits, batch['label'].cuda(args.gpu, non_blocking=True))
            loss = outputs[0].view(1, 1, -1)
            loss_item = loss.item()
            prediction = torch.argmax(softmax(outputs[1].detach()), dim=1).type(torch.int8).cpu()
            reference = batch['label'].detach().type(torch.int8)
            # count = len(batch['label'])
            # correct = len(batch['label']) - torch.count_nonzero(torch.argmax(softmax(outputs[1]), dim=1).detach().cpu() - batch['label'])            
            TP = torch.sum(torch.bitwise_and(prediction, reference)).detach() # (Pred, Ref) = (1, 1)
            TN = torch.sum(torch.bitwise_and(1 - prediction, 1 - reference)).detach() # (Pred, Ref) = (0, 0)
            FP = torch.sum(torch.bitwise_and(prediction, 1 - reference)).detach() # (Pred, Ref) = (1, 0)
            FN = torch.sum(torch.bitwise_and(1 - prediction, reference)).detach() # (Pred, Ref) = (0, 1)
            prediction = prediction.tolist()
            reference = reference.tolist()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.stdout_interval == 0:
                with open(args.log_path, "a") as f:
                    result = glue_metric.compute(predictions=prediction, references=reference)
                    # f.write("[Step {0:<4}] Train loss: {1}, Metric: {2}, Accuracy: {3}\n".format(step, loss_item, result, correct / count))
                    f.write("[Step {0}] Train loss: {1}, Metric: {2}, Accuracy: {3}, TP: {4}, TN: {5}, FP: {6}, FN: {7}\n".format(step, loss_item, result, (TP+TN)/(TP+TN+FP+FN), TP, TN, FP, FN))
            if step % args.validation_interval == 0: # and step != 0:
                model.eval()
                with torch.no_grad():
                    validation_loss = []
                    predictions = []
                    references = []
                    correct = 0
                    count = 0
                    TP = 0
                    TN = 0
                    FP = 0
                    FN = 0
                    # max_len_val = 0
                    for v_batch in validation_dataloader:
                        inputs = tokenizer(v_batch['sentence'], padding=args.pad_option, max_length=args.max_length, truncation=True, return_tensors="pt")
                        if (args.gpu is not None) or torch.cuda.is_available():
                            outputs = model(inputs['input_ids'].cuda(args.gpu, non_blocking=True), inputs['attention_mask'].cuda(args.gpu, non_blocking=True), inputs['token_type_ids'].cuda(args.gpu, non_blocking=True), labels=v_batch['label'].cuda(args.gpu, non_blocking=True))
                        else:
                            outputs = model(inputs['input_ids'], inputs['attention_mask'],  inputs['token_type_ids'], labels=v_batch['label'])
                        validation_loss.append(outputs[0].item())
                        # if max_len_val < inputs['input_ids'].shape[1]:
                        #     max_len_val = inputs['input_ids'].shape[1]
                        # count += len(v_batch['label'])
                        # correct += len(v_batch['label']) - torch.count_nonzero(torch.argmax(softmax(outputs[1]), dim=1).cpu() - v_batch['label'])
                        # predictions = predictions + torch.argmax(softmax(outputs[1]), dim=1).cpu().tolist()
                        # references = references + v_batch['label'].tolist()
                        prediction = torch.argmax(softmax(outputs[1].detach()), dim=1).type(torch.int8).cpu()
                        reference = v_batch['label'].detach().type(torch.int8)           
                        TP += torch.sum(torch.bitwise_and(prediction, reference)).detach() # (Pred, Ref) = (1, 1)
                        TN += torch.sum(torch.bitwise_and(1 - prediction, 1 - reference)).detach() # (Pred, Ref) = (0, 0)
                        FP += torch.sum(torch.bitwise_and(prediction, 1 - reference)).detach() # (Pred, Ref) = (1, 0)
                        FN += torch.sum(torch.bitwise_and(1 - prediction, reference)).detach() # (Pred, Ref) = (0, 1)
                        predictions = predictions + prediction.tolist()
                        references = references + reference.tolist()
                    # print(max_len_val)
                    with open(args.log_path, "a") as f:
                        results = glue_metric.compute(predictions=predictions, references=references)
                        # f.write("[Step {0:<4}] Validation loss: {1}, Metric: {2}, Accuracy: {3}\n".format(step, sum(validation_loss) / len(validation_loss), results, correct / count))
                        f.write("[Step {0}] Validation loss: {1}, Metric: {2}, Accuracy: {3}, TP: {4}, TN: {5}, FP: {6}, FN: {7}\n".format(step, sum(validation_loss) / len(validation_loss), results, (TP+TN)/(TP+TN+FP+FN), TP, TN, FP, FN))
                model.train()

            step += 1
        # print(max_length)
        # return
    with open(args.log_path, "a") as f:
        f.write("\n\n--- Training Finished ---\n")

if __name__=="__main__":
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    
    # parser.add_argument('--save_path', default='./exp_default')
    # parser.add_argument('--log-path', default='./log.txt')
    parser.add_argument('--pretrained-name', default="bert-base-uncased")
    # parser.add_argument('--pretrained-name', default="bert-large-uncased")
    parser.add_argument('--dataset-type', default='cola') # Current code is supports only CoLA!
    parser.add_argument('--config_path', default='./config.json')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    # parser.add_argument('--learning_rate', default=2e-6, type=int)
    parser.add_argument('--pad-option', default='longest', help="longest or max_length") 
    parser.add_argument('--max-length', default=None, type=int) 
    parser.add_argument('--training_epochs', default=5, type=int)
    parser.add_argument('--stdout_interval', default=50, type=int)
    # parser.add_argument('--checkpoint_interval', default=200, type=int)
    # parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=200, type=int)

    parser.add_argument('--dist-url', default="tcp://127.0.0.1:5712", type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

    args = parser.parse_args()

    with open(args.config_path) as f:
        data = f.read()
        config = json.loads(data)
        config = AttrDict(config)
        args.config = config

    if args.seed is not None:
        set_reproducibility(args.seed) # need to import later
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # print(get_project_name(args))
    args.log_path = "./logs_bert_base_uncased/{}.txt".format(get_project_name(args))
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        mp.spawn(train, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        train(args.gpu, ngpus_per_node, args)