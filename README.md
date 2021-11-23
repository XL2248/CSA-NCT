# CSA-NCT
Code for EMNLP21 main conference paper: [Towards Making the Most of Dialogue Characteristics for Neural Chat Translation](https://aclanthology.org/2021.emnlp-main.6/)

# Training (Taking En->De as an example)
Our code is basically based on the publicly available toolkit: [THUMT-Tensorflow](https://github.com/THUNLP-MT/THUMT) (our python version 3.6).
The following steps are training our model and then test its performance in terms of BLEU, TER, and Sentence Similarity.

## Data Preprocessing
Please refer to the "data_preprocess_code" file.

## Two-stage Training

+ The first stage

```
1) bash train_ende_base_stage1.sh # Suppose the generated checkpoint file is located in path1
```
+ The second stage (i.e., fine-tuning on the chat translation data)

```
2) bash train_ende_base_stage2.sh # Here, set the training_step=1; Suppose the generated checkpoint file is located in path2
3) python thumt_stage1_code/thumt/scripts/combine_add.py --model path2 --part path1 --output path3  # copy the weight of the first stage to the second stage.
4) bash train_ende_base_stage2.sh # Here, set the --output=path3 and the training_step=first_stage_step + 5,000; Suppose the generated checkpoint file is path4
```
+ Test by multi-blue.perl

```
5) bash test_ende_stage2.sh # set the checkpoint file path to path4 in this script. # Suppose the predicted file is located in path5 at checkpoint step xxxxx
```
+ Test by SacreBLEU and TER
Required TER: v0.7.25; Sacre-BLEU: version.1.4.13 (BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.13)

```
6) python SacreBLEU_TER_Coherence_Evaluation_code/cal_bleu_ter4ende.py # Please correctly set the golden file and predicted file in this file and in sacrebleu_ende.py, respectively.
```

+ Coherence Evaluation by Sentence Similarity
Required: gensim; MosesTokenizer

```
7) python SacreBLEU_TER_Coherence_Evaluation_code/train_word2vec.py # firstly downloading the corpus in [2] and then training the word2vec.
8) python SacreBLEU_TER_Coherence_Evaluation_code/eval_coherence.py # putting the file containing three precoding utterances and the predicted file in corresponding location and then running it.
```

# Citation
If you find this project helps, please cite our paper :)

```
@inproceedings{liang-etal-2021-towards,
    title = "Towards Making the Most of Dialogue Characteristics for Neural Chat Translation",
    author = "Liang, Yunlong  and
      Zhou, Chulun  and
      Meng, Fandong  and
      Xu, Jinan  and
      Chen, Yufeng  and
      Su, Jinsong  and
      Zhou, Jie",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.6",
    pages = "67--79",
}
```
