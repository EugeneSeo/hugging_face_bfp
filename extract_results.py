import csv 
import re 
import argparse
import os
MAX_EPOCH = 5

def extract(args):
    for log_name in args.log_files:
        f_log = open(os.path.join(args.log_dir, log_name), 'r')
        epoch = -1

        if args.glue_task == "cola" or args.glue_task == "sst2":
            results_dict = {"best_metric1": [-1,-1]}
        elif args.glue_task == "stsb" or args.glue_task == "mnli":
            results_dict = {"best_metric1": [-1,-1,-1], "best_metric2": [-1,-1,-1]}
        else:
            raise Exception("NOT DEFINED!!")

        if args.glue_task == "mnli":
            matched = 0    # placeholder for the previous line's value

        while True:
            line = f_log.readline()
            if not line: break
            line = line.strip() 
            if "[Step" not in line:
                if "Epoch" in line:
                    epoch += 1
                    if epoch == MAX_EPOCH: 
                        break
                continue
            words = line.split(",")
            if words[0][words[0].find(']')+2] == 'T':
                continue
            
            if args.glue_task == "cola":
                cor = float(words[1][34:-1]) 
                if cor > results_dict["best_metric1"][0]:
                    results_dict["best_metric1"] = [cor, epoch]

            elif args.glue_task == "sst2":
                acc = float(words[1][22:-1]) 
                if acc > results_dict["best_metric1"][0]:
                    results_dict["best_metric1"] = [acc, epoch]

            elif args.glue_task == "stsb":
                pearson = float(words[1][21:]) if words[1][21:] != "nan" else -1
                spearmanr = float(words[2][14:-1]) if words[1][21:-1] != "nan" else -1
                if pearson > results_dict["best_metric1"][0]:
                    results_dict["best_metric1"] = [pearson, spearmanr, epoch]
                if spearmanr > results_dict["best_metric2"][1]:
                    results_dict["best_metric2"] = [pearson, spearmanr, epoch]

            elif args.glue_task == "mnli":
                if words[0][words[0].find('(')+2] == 'a': 
                    matched = float(words[1][22:-1]) if words[1][22:-1] != "nan" else 0
                if words[0][words[0].find('(')+2] == 'i': 
                    mismatched = float(words[1][22:-1]) if words[1][22:-1] != "nan" else 0
                    if matched > results_dict["best_metric1"][0]:
                        results_dict["best_metric1"] = [matched, mismatched, epoch]
                    if mismatched > results_dict["best_metric2"][1]:
                        results_dict["best_metric2"] = [matched, mismatched, epoch]
        
        print(results_dict)
        f_log.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--log_dir', default="./logs_bert-base-uncased")
    parser.add_argument('--glue_task', default="cola") # "cola, sst2, stsb, mnli"
    args = parser.parse_args()
    args.glue_task = args.glue_task.lower()
    args.log_files = [
                 "no_st-no_thres-BertForSequenceClassification-fp32-sgd_lr0.0003-cola-global_batch8-single_gpu-seed1024-g16-epochs10.txt",
                 "no_st-no_thres-BertForSequenceClassification-fp32-sgd_lr0.0003-cola-global_batch8-single_gpu-seed2048-g16-epochs5.txt",
                ]

    extract(args)