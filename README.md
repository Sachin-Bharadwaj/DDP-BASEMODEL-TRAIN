# DDP-BASEMODEL-TRAIN

I trained a gpt2 like model with about ~1.3 billion parameter, 768 hidden dimension, gpt2 tokenizer from tiktoken, max seq length=1024, number of multi-head=12 and depth=12. The model uses a learned positional encoding and the weight are not tied b/w embedding layer and prediction head. The training dataset that I used is Fineweb-edu 10BT from hugging face. Training was done using 2xA100 40GB for ~8hrs. 

Here is the brief outline of the repo: </br>
|cfg| : it has cfg.py which has the model configeration parameters
|data| : it has three main folders, |tinyshakespare| , |hellaswag| , |edufineweb10B|. All these 3 folders are created by tiny_shakespare.py, hellaswag.py, fineweb-edu.py. tiny_shakepare.py creates a dataset for overfitting the model while training just to make sure that loss keeps gioing down as we keep on overfitting. fineweb-edu.py creates several *_train_*.bin files and *_val_*.bin files which are files that have been already converted to token ids using tiktoken tokenizer for gpt2. Further, hellaswag.py is used to evaluate the the trained model checkpoint on hellaswag val data which comprises ~10,000 examples where each example has a context and 4 possible completion choices, with correct completion present as label. The hellaswag val split is used to check the model performance on different dataset for the completions task
|model|: it has two folders, <lib>: which has several building blocks for constructing transformer decoder models and GPT2.py is the main file which is used to perform training on multi/single GPU set up.

*How to run the repo*
- conda create -n <env_name> python=3.11
- conda activate <env_name>
- install pytorch for GPU from torch website
- pip install -r requirements-gpu-cuda.txt (remove the pytorch related installations form this file because you already installed those in above steps) </br>

Now, before you start training, ensure that the training data and val data is available. Simply run python data/fineweb-edu.py from the root folder of the repo, this will dump the .bin shards in the appropraite folder. After that, run python data/tiny_shakespare.py from the root folder. This will dump a small training data in the appropriate folder which we will use to overfit our model before training on large scale data to ensure that everything is fine, the training loss should keep on decreasing as we keep on overfitting which is a good sign. (If this doesn't happen - time for debugging our code for bugs). Here is the command to run to start overfitting <br>
torchrun --standalone --nproc_per_node=2 model/GPT2.py --num_iterations=200 --sequence_length=1024 --batch_size=16 --total_batch_size=65536  --compile=1 --tensorcores=1 --dtype=bfloat16 --flash=1 --input_train_bin="data/tinyshakespare/*_train.bin"  --output_dir="./outdir"

After we ensure that the model has been set up correctly, run the following command to start the training on 10BT train dataset. <br>
torchrun --standalone --nproc_per_node=2 model/GPT2.py --num_iterations=38146 --sequence_length=1024 --batch_size=16 --total_batch_size=262144  --compile=1 --tensorcores=1 --dtype=bfloat16 --flash=1 --input_train_bin="data/edu_fineweb10B/*_train_*.bin"  --input_val_bin="data/edu_fineweb10B/*_val_*.bin"  --output_dir="./outdir/edufineweb10B" --learning_rate=6e-4 --weight_decay=0.1 --learning_rate_decay_frac=0.1 --warmup_iters=700 --overfit_single_batch=0 --val_loss_every=100 --sample_every=1000 --zero_stage=0 --save_every_niter=2000 --wandb_log_iters=1 --wandb_project_id="gpt2-10BT"

Now, wait for 8 hours for the training to finish (if you are using 2xA100-40GB). **NOTE** --zero_stage=0 , do not change it from anything else than 0 because for some reason I was unable to make the sync the sharded optimizer states if you set it to 1. The checkpoint is saved in the path provided above by --output_dir.
Let me explain some of the flags in the above command </br>
