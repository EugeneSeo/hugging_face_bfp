# Transformers with Block Floating Point
This repository explains the steps of applying BFP(Block Floating Point) modules to the transformer-based modules.  

## About huggingface/transformers
The [Transformers](https://github.com/huggingface/transformers) repository developed by 🤗 Hugging Face is selected as a vanilla module.
- How to clone: From the [Transformers](https://github.com/huggingface/transformers) github, you can clone the repository. There are various branches, both the release and the developers' versions. 
- How to install: Enter `pip install -e .` at the transformer/ folder. 
- Code details
    - How does this module load the pre-trained parameters? [_load_state_dict_into_model()](https://github.com/huggingface/transformers/blob/df06fb1f0b0864ca0c2dcd9f7f6aab5de6447c59/src/transformers/modeling_utils.py#L451) function of modeling_utils.py in transformer/src/transformers folder implements this functionality. If you want to change pre-trained parameters for debugging purposes, you may add some codes to this function. 
    - Where are the transformer-based modules implemented? On the [transformers/src/transformers/models](https://github.com/huggingface/transformers/tree/main/src/transformers/models) folder, you can find all the implemented transformer-based models. 

## Adding BFP to the Transformers module
The following steps describe methods of connecting the Transformers repository and the BFP module. It focuses only on BERT, so you should modify it if you are interested in other transformer-based models. To see our implementations, check the region wrapped with the comment lines. 
1. Clone [Transformers](https://github.com/huggingface/transformers).
2. Clone [BFP module](https://github.com/yoonsung-kim/bfp-training) in [./transformers/src/transformers/](https://github.com/huggingface/transformers/tree/main/src/transformers) and follow the initializing steps (E.g. Create .so files using ‘make all’).
3. Modify [modeling_bert.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py).  
    - Import CustomLinear & CustomMatMul.
    - Change nn.linear to the `CustomLinear` function.
        ```python
        if config.use_bfp == False:
            self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        else:
            global GLOBAL_ID
            self.dense = CustomLinear(config.hidden_size, config.intermediate_size, bias=True, precision_flag=config.PrecisionFlag, global_id=GLOBAL_ID, config=config)
            GLOBAL_ID += 1
        ```
    - Change nn.matmul to CustomMatMul: see class BertSelfAttention.
    - For all input tensors of the CustomLayers: apply [.contiguous()](https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html) method. Examples are as follows. 
        ```python
        # line 119 of new_custom_matmul.py
        grad_lhs_mat = bfp_gemm.run(grad_output.contiguous(), rhs_mat.contiguous())
        ```
        
4. Change the BFP code: CustomLinear, CustomMatMul
    - Add the config parameter as the module input.
    - For all input tensors of custom operations: apply [.contiguous()](https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html) method.
5. Install Transformer: Enter `pip install -e .` at the transformer/ folder. 
6. Using the transformer module, train the model using train.py (at .)
    -  train.py code: EugeneSeo/hugging_face_bfp/train.py
    - WARNING: only the single-GPU training script exists!

## Training
To train transformer-based models, enter the following commands. `train.py` supports single-sentence tasks while `train_pair.py` is for sentence-pair tasks. 
```bash
python train.py
python train_pair.py --seed 1024
```
For BFP-based training, the padding option should be set to 'max_length' due to the implementation details of the BFP module. The maximum length of the dataset varies, so the maximum length should be profiled before training. The training script supports CoLA, SST2, STSb, and MNLI of the GLUE Benchmark, but others are not approved yet. 

## Checking the Results
The easiest way to summarize the results saved in the logs directory is using `extract_results.py`. You would have to change the `log_files` arguments on the script. The maximum epoch is set to 5 since we are focusing on the fine-tuning task. Like the training script, this extracting code only supports CoLA, SST2, STSb, and MNLI. 
```bash
python extract_results.py
```
It will generate the best scores of the GLUE metric, as shown below. 
```python
# When there is one metric
{'best_metric1': [//best score of the metric, //epoch]}
# When there are two metrics
{'best_metric1': [//best score of the first metric, //the score of the second metric, //epoch], 
 'best_metric2': [//the score of the first metric, //best score of the second metric, //epoch]}
```
