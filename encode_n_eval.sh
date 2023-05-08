#!/bin/bash
trained_folder=("model_msmarco_characterbert_st_3" "model_msmarco_characterbert_st_4" "model_msmarco_characterbert_st_5" "model_msmarco_characterbert_st_6" "model_msmarco_characterbert_st_7" "model_msmarco_characterbert_st_8")

encoded_folder=("model_msmarco_characterbert_st_3_embs" "model_msmarco_characterbert_st_4_embs" "model_msmarco_characterbert_st_5_embs" "model_msmarco_characterbert_st_6_embs" "model_msmarco_characterbert_st_7_embs" "model_msmarco_characterbert_st_8_embs")

for name in ${encoded_folder[@]}; do
mkdir $name
done

for i in ${!encoded_folder[@]}; do
# encode query
python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path ${trained_folder[$i]}/checkpoint-final \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --encode_in_path data/dl-typo/query.typo.tsv \
  --encoded_save_path ${encoded_folder[$i]}/query_dltypo_typo_emb.pkl \
  --q_max_len 32 \
  --encode_is_qry \
  --character_query_encoder True


# encode corpus
for s in $(seq -f "%02g" 0 5)
do
python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path ${trained_folder[$i]}/checkpoint-final \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --p_max_len 128 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --encoded_save_path ${encoded_folder[$i]}/corpus_emb.${s}.pkl \
  --encode_num_shard 20 \
  --encode_shard_index ${s} \
  --cache_dir cache \
  --character_query_encoder True \
  --passage_field_separator [SEP]
done

#retrieval
python -m tevatron.faiss_retriever \
--query_reps ${encoded_folder[$i]}/query_dltypo_typo_emb.pkl \
--passage_reps ${encoded_folder[$i]}/'corpus_emb.*.pkl' \
--depth 1000 \
--batch_size -1 \
--save_text \
--save_ranking_to ${encoded_folder[$i]}/character_bert_st_dltypo_typo_rank.txt


python -m tevatron.utils.format.convert_result_to_trec \
              --input ${encoded_folder[$i]}/character_bert_st_dltypo_typo_rank.txt \
              --output ${encoded_folder[$i]}/character_bert_st_dltypo_typo_rank.txt.trec


done

