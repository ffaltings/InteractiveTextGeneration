#Makefile for training
CUDA_DEVICE ?= 0

# CNN BASE
base_run_dir := #/path/to/your/run_dir/
base-cnn-bart_editor_large:
	python infosol/train.py --cuda_device $(CUDA_DEVICE) --run_dir $(base_run_dir) --run_name cnn-bart_editor_large base --n_epochs 6 --max_train_edits 1000000 --max_val_edits 2000 --model_name bart-large --track_gradient_norm True
base-cnn-bart_editor_large-clip_grad_norm:
	python infosol/train.py --cuda_device $(CUDA_DEVICE) --run_dir $(base_run_dir) --run_name cnn-bart_editor_large base --n_epochs 6 --max_train_edits 1000000 --max_val_edits 2000 --model_name bart-large --track_gradient_norm True --clip_grad_norm True


# CNN DAGGER
run_dir := #/path/to/your/run_dir/
dagger_args_ := --use_timestamp False --run_dir $(run_dir) --cuda_device $(CUDA_DEVICE) dagger --n_epochs 600 --max_train_edits 1000000 --max_val_edits 2000 --n_warmup_epochs 300  --dagger_sampling_rate 1.0 --sample_batch_size 10000 --val_sample_batch_size 1000 
dagger_args := $(dagger_args_) --sampling_annealing_rate 0.9

# main
cnn-bart_editor:
	python infosol/train.py --run_name cnn-bart_editor $(dagger_args) 
cnn-bart_editor_large:
	python infosol/train.py --run_name cnn-bart_editor_large $(dagger_args) --model_name bart-large --track_gradient_norm True --clip_grad_norm True
cnn-bart_s2s:
	python infosol/train.py --run_name cnn-bart_s2s $(dagger_args) --model_name barts2s 
# different oracles
cnn-bart_editor-adj_edits:
	python infosol/train.py --run_name cnn-bart_editor-adj_edits $(dagger_args) --adjacent_ops True 
cnn-bart_editor-contig_edits:
	python infosol/train.py --run_name cnn-bart_editor-contig_edits $(dagger_args) --contiguous_edits True 
cnn-bart_editor-adj_edits-contig_edits:
	python infosol/train.py --run_name cnn-bart_editor-adj_edits-contig_edits $(dagger_args) --adjacent_ops True --contiguous_edits True 

# noise fractions
cnn-bart_editor-noise_0.0:
	python infosol/train.py --run_name cnn-bart_editor-noise_0.0 $(dagger_args) --noise_frac 0 
cnn-bart_editor-noise_0.1:
	python infosol/train.py --run_name cnn-bart_editor-noise_0.1 $(dagger_args) --noise_frac 0.1 
cnn-bart_editor-noise_0.2:
	python infosol/train.py --run_name cnn-bart_editor-noise_0.2 $(dagger_args) --noise_frac 0.2
# annealing rate
cnn-bart_editor-anneal_0.85:
	python infosol/train.py --run_name cnn-bart_editor-anneal_0.85 $(dagger_args_) --sampling_annealing_rate 0.85 
cnn-bart_editor-anneal_0.80:
	python infosol/train.py --run_name cnn-bart_editor-anneal_0.80 $(dagger_args_) --sampling_annealing_rate 0.80 




