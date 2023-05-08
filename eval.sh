#!/bin/bash
eval_files=("character_bert_st_dltypo_typo_rank3.txt.trec" "character_bert_st_dltypo_typo_rank4.txt.trec" "character_bert_st_dltypo_typo_rank5.txt.trec" "character_bert_st_dltypo_typo_rank6.txt.trec" "character_bert_st_dltypo_typo_rank7.txt.trec" "character_bert_st_dltypo_typo_rank8.txt.trec")

for i in ${!eval_files[@]}; do
trec_eval -l 2 -m ndcg_cut.10 -m map -m recip_rank data/dl-typo/qrels.txt eval_files/${eval_files[$i]}

done  