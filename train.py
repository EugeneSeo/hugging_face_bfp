import argparse
import os
import time
import warnings

import torch
from transformers import *
from datasets import load_dataset, load_metric
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# from transformers.src.transformers.bfp_training.util.bfp.bfp_config import BfpConfig

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
    # optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, eps = 1e-8 )
    softmax = torch.nn.Softmax(dim=1)
    loss_func = torch.nn.CrossEntropyLoss().cuda(args.gpu)

    model.train()
    step = 0
    training_time = 0

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    # BfpConfig.use_bfp = False if args.precision_flag == 0 else True

    # max_len = 0
    initiated = False
    for epoch in range(args.training_epochs):
        with open(args.log_path, "a") as f:
            f.write("\n=== Epoch {} ===\n".format(epoch))
        
        if args.distributed:
            train_dist_sampler.set_epoch(epoch)

        end = time.time()
        for i, batch in enumerate(train_dataloader): # batch: 'sentence', 'label', 'idx'
            # Tokenize the sentence and get the model output
            # batch['sentence'] = ["[CLS]" + str(sentence) + "[SEP]" for sentence in batch['sentence']]
            inputs = tokenizer(batch['sentence'], padding=args.pad_option, max_length=args.max_length, truncation=True, return_tensors="pt")
            # print(inputs)
            # print(batch['sentence'])
            # if inputs['input_ids'].shape[1] > max_len:
            #     max_len = inputs['input_ids'].shape[1]
            # continue
            if (args.gpu is not None) or torch.cuda.is_available():
                outputs = model(inputs['input_ids'].cuda(args.gpu, non_blocking=True), inputs['attention_mask'].cuda(args.gpu, non_blocking=True), inputs['token_type_ids'].cuda(args.gpu, non_blocking=True), labels=batch['label'].cuda(args.gpu, non_blocking=True))
                # inputs = inputs.cuda(args.gpu)
                # outputs = model(**inputs)
            else:
                outputs = model(inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'], labels=batch['label'])

            if not initiated:
                initiated = True
                continue
            # Update the model
            # loss = outputs[0].view(1, 1, -1)

            # loss = outputs.loss
            # print(outputs)
            loss = loss_func(outputs.logits, batch['label'].cuda(args.gpu, non_blocking=True))
            loss_item = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # optimizer.zero_grad()
            print("one step done")

            training_time += time.time() - end
            end = time.time()

            if step % args.stdout_interval == 0:
                with open(args.log_path, "a") as f:
                    f.write("[Step {0:<4}] Train loss: {1}\n".format(step, loss_item))
            if step % args.validation_interval == 0: # and step != 0:
                model.eval()
                # val_max_len = 0
                with torch.no_grad():
                    validation_loss = []
                    correct = 0
                    count = 0
                    
                    for v_batch in validation_dataloader:
                        # print(v_batch)
                        inputs = tokenizer(v_batch['sentence'], padding=args.pad_option, max_length=args.max_length, truncation=True, return_tensors="pt")
                        # print("train.py:", inputs['input_ids'].shape, inputs['attention_mask'].shape)
                        # if inputs['input_ids'].shape[1] > val_max_len:
                        #     val_max_len = inputs['input_ids'].shape[1]
                        # continue
                        if (args.gpu is not None) or torch.cuda.is_available():
                            outputs = model(inputs['input_ids'].cuda(args.gpu, non_blocking=True), inputs['attention_mask'].cuda(args.gpu, non_blocking=True), inputs['token_type_ids'].cuda(args.gpu, non_blocking=True), labels=v_batch['label'].cuda(args.gpu, non_blocking=True))
                        else:
                            outputs = model(inputs['input_ids'], inputs['attention_mask'],  inputs['token_type_ids'], labels=v_batch['label'])
                        validation_loss.append(outputs[0].item())
                        count += len(v_batch['label'])
                        correct += len(v_batch['label']) - torch.count_nonzero(torch.argmax(softmax(outputs[1]), dim=1).cpu() - v_batch['label'])
                    with open(args.log_path, "a") as f:
                        f.write("[Step {0:<4}] Validation loss: {1}, Accuracy: {2}\n".format(step, sum(validation_loss) / len(validation_loss), correct / count))
                # if step == 0:
                #     print(val_max_len)
                model.train()
            if step % args.checkpoint_interval == 0 and step != 0:
                checkpoint_path = "{}/model_{:08d}.pt".format(args.save_path, step)
                if args.distributed:
                    torch.save({'model_state_dict': model.module.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
                else:
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)

            step += 1
            # if step > 100:
            #     break
        # if epoch == 0:
        #     print(max_len)
        # scheduler.step()
    with open(args.log_path, "a") as f:
        f.write("\n\n[Training Time] total: {0}, average (time per step): {1}".format(training_time, training_time / step))

if __name__=="__main__":
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--save_path', default='./exp_default')
    parser.add_argument('--log-path', default='./log.txt')
    parser.add_argument('--pretrained-name', default="bert-base-uncased")
    parser.add_argument('--dataset-type', default='cola') # Current code is supports only CoLA!
    parser.add_argument('--config_path', default='./config.json')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    # parser.add_argument('--learning_rate', default=2e-6, type=int)
    parser.add_argument('--pad-option', default='longest', help="longest or max_length") 
    parser.add_argument('--max-length', default=None, type=int) 
    parser.add_argument('--training_epochs', default=5, type=int)
    parser.add_argument('--stdout_interval', default=50, type=int)
    parser.add_argument('--checkpoint_interval', default=200, type=int)
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

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        mp.spawn(train, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        train(args.gpu, ngpus_per_node, args)