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
  --output $work_dir/train_en_de_base_model_bottom${bb}_ctx${ctx_ind}_end2end \
  --input $data_dir/train.tok.bpe.32000.en $data_dir/train.tok.bpe.32000.de \
  --vocabulary $vocab_data_dir/ende.bpe32k.vocab4.txt $vocab_data_dir/ende.bpe32k.vocab4.txt \
  --validation $data_dir/dev.tok.bpe.32000.en \
  --references $data_dir/dev.tok.de \
  --dialog_src_context $data_dir/train_ctx.tok.bpe.32000.en \
  --dialog_tgt_context $data_dir/train_ctx.tok.bpe.32000.de \
  --style_src_context $data_dir/train_enper_ctx.tok.bpe.32000.en \
  --style_tgt_context $data_dir/train_deper_ctx.tok.bpe.32000.de \
  --language_src_context $data_dir/train_ctx.tok.bpe.32000.en \
  --language_tgt_context $data_dir/train_ctx.tok.bpe.32000.de \
  --dev_dialog_src_context $data_dir/dev_ctx.tok.bpe.32000.en \
  --dev_dialog_tgt_context $data_dir/dev_ctx.tok.bpe.32000.de \
  --dev_style_src_context $data_dir/dev_enper_ctx.tok.bpe.32000.en \
  --dev_style_tgt_context $data_dir/dev_deper_ctx.tok.bpe.32000.de \
  --dev_language_src_context $data_dir/dev_ctx.tok.bpe.32000.en \
  --dev_language_tgt_context $data_dir/dev_ctx.tok.bpe.32000.de \
  --parameters=device_list=[0,1,2,3],update_cycle=1,eval_steps=50,train_steps=1,batch_size=4096,max_length=128,constant_batch_size=False,residual_dropout=0.1,attention_dropout=0.1,relu_dropout=0.1,shared_source_target_embedding=True,learning_rate=1.0,start_steps=1,bottom_block=$bb,use_crg=True,use_mrg=True,use_speaker=True,use_coherence=True,kl_annealing_steps=$kl_steps1,kl_annealing_steps2=$kl_steps2
