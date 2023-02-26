import torch
# from transformers import BertTokenizer, BertModel
from transformers import *
from datasets import load_dataset, load_metric
import random
import numpy as np

# _BFP_flag = True
_BFP_flag = False
seed = 2022 
deterministic = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

class Hook():
    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

if __name__=="__main__":
    # pretrained_name = "bert-base-uncased"
    # tokenizer = BertTokenizer.from_pretrained(pretrained_name)
    # model = BertModel.from_pretrained(pretrained_name).cuda()
    pretrained_name = "textattack/bert-base-uncased-CoLA"
    tokenizer = BertTokenizer.from_pretrained(pretrained_name)
    # model = BertForSequenceClassification.from_pretrained(pretrained_name, num_labels = 2, problem_type = "single_label_classification").cuda()
    model = BertForSequenceClassification.from_pretrained(pretrained_name, config="./config.json", num_labels = 2, problem_type = "single_label_classification").cuda()
    # model = BertForSequenceClassification.from_pretrained(pretrained_name, config="./config_bfp.json", num_labels = 2, problem_type = "single_label_classification").cuda()
    
    for i, parameter in enumerate(model.parameters()):
        if _BFP_flag:
            print("parameter_{0:<3}:".format(i), parameter)
            np.save("./parameter_ckpt/{}.npy".format(i), parameter.cpu().detach().numpy())
        else:
            ori_param = parameter.cpu().detach().numpy()
            bfp_param = np.load("./parameter_ckpt/{}.npy".format(i))
            print("parameter_{0:<3}:".format(i), np.sum((ori_param - bfp_param)**2))
    # ========= Simple cases =========
    
    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # outputs = model(inputs['input_ids'].cuda(), inputs['token_type_ids'].cuda(), inputs['attention_mask'].cuda())
    # print(outputs)

    # ========= Test with CoLA dataset =========

    dataset = load_dataset("glue", "cola", split="train")
    optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-5,  eps = 1e-8 )

    print("==="*10)
    for layer in model._modules.items():
        print(layer)

    # print("==="*10)
    # # linear_name = ["query", "key", "value", "dense", "intermediate", "output", "classifier"]
    # linear_name = ["query", "key", "value", "dense", "classifier"]
    # hookB = []
    # def register_hook(module, name):
    #     for layer in module._modules.items():
    #         if layer[0] in linear_name:
    #             hookB.append((name + "." + layer[0], Hook(layer[1], backward=True)))
    #         register_hook(layer[1], name + "." + layer[0])
    # register_hook(model, "model")
    # hookB.reverse()
    # print(hookB)
    
    grad_weight = []
    grad_bias = []
    def tensor_hook_weight(grad):
        grad_weight.append(grad)
        pass
    def tensor_hook_bias(grad):
        grad_bias.append(grad)
        pass
    linear_name = ["query", "key", "value", "dense", "classifier"]
    def register_hook(module, name):
        for layer in module._modules.items():
            if layer[0] in linear_name:
                layer[1].weight.register_hook(tensor_hook_weight)
                layer[1].bias.register_hook(tensor_hook_bias)
            register_hook(layer[1], name + "." + layer[0])

    model.train()
    for example in dataset:
        inputs = tokenizer(example['sentence'], return_tensors="pt")
        print(inputs)
        # outputs = model(**inputs, labels=torch.tensor([example['label']], dtype = torch.float32))
        # outputs = model(**inputs, labels=torch.tensor([example['label']]).cuda())
        outputs = model(inputs['input_ids'].cuda(), inputs['token_type_ids'].cuda(), inputs['attention_mask'].cuda(), labels=torch.tensor([example['label']]).cuda())
        print(outputs)
        loss = outputs[0].view(1, 1, -1)
        ###
        grad_weight = []
        grad_bias = []
        print("==="*10)
        if not _BFP_flag:
            register_hook(model, "model")
        ###
        print(loss.shape)
        loss.backward()
        
        ###
        if not _BFP_flag:
            for layer in range(len(grad_bias)):
                # ori_grad = hookB[layer][1].output[0].cpu().numpy()
                ori_grad_weight = grad_weight[layer].cpu().numpy()
                ori_grad_bias = grad_bias[layer].cpu().numpy()
                bfp_grad_weight = np.load("./gradient_ckpt/Lin_backward_weight_{}.npy".format(layer))
                bfp_grad_bias = np.load("./gradient_ckpt/Lin_backward_bias_{}.npy".format(layer))
                if bfp_grad_bias.ndim > 1: # ctx.needs_input_grad[2] was not set at that time
                    bfp_grad_bias = np.sum(bfp_grad_bias, 0)
                # print("ori_grad: ", ori_grad)
                # print("bfp_grad: ", bfp_grad)
                # print("layer_{}: ".format(layer), np.array_equal(ori_grad, bfp_grad))
                print("layer_{0:<2}: ".format(layer), "weight -", np.average(np.abs(ori_grad_weight - bfp_grad_weight)), ", bias -", np.average(np.abs(ori_grad_bias - bfp_grad_bias)))
                # print("layer_{0:<2}: ".format(layer), "bias -", np.sum((ori_grad_bias - bfp_grad_bias)**2))
                # print(ori_grad_bias)
                # print(bfp_grad_bias)
                # print("layer_{0:<2}: ".format(layer), "weight -", np.sum((ori_grad_weight - bfp_grad_weight)**2))
                # print("layer_{0:<2}{1:<52}: ".format(layer, "(" + hookB[layer][0] + ")"), np.sum((ori_grad - bfp_grad)**2))
                # print("---"*10)
        ###
        optimizer.step()
        break

    