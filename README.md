# DDP-BASEMODEL-TRAIN

I trained a gpt2 like model with about ~1.3 billion parameter, 768 hidden dimension, gpt2 tokenizer from tiktoken, max seq length=1024, number of multi-head=12 and depth=12. The model uses a learned positional encoding and the weight are not tied b/w embedding layer and prediction head. The training dataset that I used is Fineweb-edu 10BT from hugging face. Training was done using 2xA100 40GB for ~8hrs. 

Here is the brief outline of the repo:
<cfg> : it has cfg.py which has the model configeration parameters
<data> : it has three main folders, <tinyshakespare> , <hellaswag> , <edufineweb10B>. All these 3 folders are created by tiny_shakespare.py, hellaswag.py, fineweb-edu.py. tiny_shakepare.py creates a dataset for overfitting the model while training just to make sure that loss keeps gioing down as we keep on overfitting. fineweb-edu.py creates several *_train_*.bin files and *_val_*.bin files which are files that have been already converted to token ids using tiktoken tokenizer for gpt2. Further, hellaswag.py is used to evaluate the the trained model checkpoint on hellaswag val data which comprises ~10,000 examples where each example has a context and 4 possible completion choices, with correct completion present as label. The hellaswag val split is used to check the model performance on different dataset for the completions task
<model>: it has two folders, <lib>: which has several building blocks for constructing transformer decoder models and GPT2.py is the main file which is used to perform training on multi/single GPU set up.

*How to run the repo*
