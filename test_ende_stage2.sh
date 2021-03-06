code_dir=thumt_stage2_code
work_dir=$PWD
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
data_name=test
kl_steps1=5000
kl_steps2=5000
vocab_data_dir=path_to_vocab_file
data_dir=path_to_data_file
checkpoint_dir=path_to_checkpoint_file

Step="xxxxx"
for idx in $Step
do
    chmod 777 -R $work_dir/$checkpoint_dir
    echo model_checkpoint_path: \"model.ckpt-$idx\" > $work_dir/$checkpoint_dir/checkpoint
    cat $work_dir/$checkpoint_dir/checkpoint
    echo decoding with checkpoint-$idx
    CUDA_VISIBLE_DEVICES=0 python $work_dir/$code_dir/thumt/bin/translator.py \
        --models transformer \
        --checkpoints $work_dir/$checkpoint_dir \
        --input $data_dir/"$data_name".tok.bpe.32000.en \
        --output $data_dir/"$data_name".out.de.${idx} \
        --vocabulary $vocab_data_dir/ende.bpe32k.vocab4.txt $vocab_data_dir/ende.bpe32k.vocab4.txt $vocab_dir/position.txt \
        --dev_dialog_src_context $data_dir/"$data_name"_ctx.tok.bpe.32000.en \
        --dev_dialog_tgt_context $data_dir/"$data_name"_ctx.tok.bpe.32000.de \
        --dev_style_src_context $data_dir/"$data_name"_enper_ctx.tok.bpe.32000.en \
        --dev_style_tgt_context $data_dir/"$data_name"_deper_ctx.tok.bpe.32000.de \
        --dev_sample $data_dir/"$data_name"_bpe.32k.en \
        --parameters=decode_batch_size=64
    echo evaluating with checkpoint-$idx
    chmod 777 -R $data_dir
    sed -r "s/(@@ )|(@@ ?$)//g" $data_dir/"$data_name".out.de.${idx} > $data_dir/${data_name}.out.de.delbpe.${idx}
    echo finished of checkpoint-$idx
done
