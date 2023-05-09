#!/bin/bash
#used to create the DR models used for experiment
trained_folder=("model_msmarco_characterbert_st_3" "model_msmarco_characterbert_st_4" "model_msmarco_characterbert_st_5" "model_msmarco_characterbert_st_6" "model_msmarco_characterbert_st_7" "model_msmarco_characterbert_st_8")
training_rate=(5e-3 5e-4 5e-5 5e-6 5e-7 5e-8)

for name in ${trained_folder[@]}; do
mkdir $name
done

for i in ${!trained_folder[@]}; do
python -m tevatron.driver.train \
--model_name_or_path bert-base-uncased \
--character_bert_path ./general_character_bert \
--output_dir trained_folder \
--passage_field_separator [SEP] \
--save_steps 40000 \
--dataset_name Tevatron/msmarco-passage \
--fp16 \
#NOTE: reduced from 16 to 4 be able to run on ICL6 machine
--per_device_train_batch_size 4 \ 
--learning_rate ${training_rate[$i]} \
#NOTE: reduced from 150,000 to 3,000 in order to run in reasonable time 
--max_steps 3000 \ 
--dataloader_num_workers 10 \
--cache_dir ./cache \
--logging_steps 150 \
--character_query_encoder True \
--self_teaching True
done
