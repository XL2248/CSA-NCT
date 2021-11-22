code_dir=thumt_stage2_code
work_dir=$PWD
ctx_ind=3
kl_steps1=5000
kl_steps2=5000
bb=1 #bottom_block
vocab_data_dir=path_to_vocab_file
data_dir=path_to_data_file
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $work_dir/models/$model_name \
  --input $train_data/train_bpe.32k.${src} $train_data/train_bpe.32k.${tgt} \
  --vocabulary $vocab_data_dir/ende.bpe32k.vocab4.txt $vocab_data_dir/ende.bpe32k.vocab4.txt $vocab_data_dir/position.txt \
  --validation $data_dir/dev_bpe.32k.${src} \
  --references $data_dir/dev.tok.${tgt} \
  --context_source $data_dir/train_ctx_src_bpe.32k.${src} \
  --dialog_src_context $train_data/train_ctx_bpe.32k.${src} \
  --dialog_tgt_context $train_data/train_ctx_bpe.32k.${tgt} \
  --style_src_context $train_data/train_ctx_bpe.32k.${src} \
  --style_tgt_context $train_data/train_ctx_bpe.32k.${tgt} \
  --sample $train_data/train_bpe.32k.${tgt} \
  --dev_context_source $data_dir/dev_ctx_src_bpe.32k.${src} \
  --dev_dialog_src_context $data_dir/dev_ctx_bpe.32k.${src} \
  --dev_dialog_tgt_context $data_dir/dev_ctx_bpe.32k.${tgt} \
  --dev_style_src_context $data_dir/dev_ctx_bpe.32k.${src} \
  --dev_style_tgt_context $data_dir/dev_ctx_bpe.32k.${tgt} \
  --dev_sample $data_dir/dev_bpe.32k.${tgt} \
  --parameters=device_list=[0,1,2,3],update_cycle=1,eval_steps=50,train_steps=1,batch_size=4096,max_length=128,constant_batch_size=False,residual_dropout=0.1,attention_dropout=0.1,relu_dropout=0.1,shared_source_target_embedding=True,learning_rate=1.0,start_steps=1,bottom_block=$bb,use_crg=True,use_mrg=True,use_speaker=True,use_coherence=True,kl_annealing_steps=$kl_steps1,kl_annealing_steps2=$kl_steps2
