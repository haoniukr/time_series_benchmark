seq_len=96
root_path_name=/mnt/data1/python/time_series_benchmark/TOTEM-master/
data_path_name=../../datasets/bigdatasets/NYTM-0919.csv
data_name=custom
random_seed=2021
pred_len=96
gpu=0

for year_start in 2017 2015 2013 2011 2009
do

python -u forecasting/save_revin_data.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 67 \
  --gpu $gpu\
  --date_split ${year_start}-01-01S00:00:00D2018-01-01S00:00:00D2019-01-01S00:00:00 \
  --save_path "forecasting/data/NYTM-0919-"${year_start}

done



for year_start in 2017 2015 2013 2011 2009
do
gpu=0
python forecasting/train_vqvae.py \
  --config_path forecasting/scripts/NYTM-0919.json \
  --model_init_num_gpus $gpu \
  --data_init_cpu_or_gpu cpu \
  --comet_log \
  --comet_tag pipeline \
  --comet_name vqvae_weather \
  --save_path "forecasting/saved_models/NYTM-0919-"${year_start}"/"\
  --base_path "forecasting/data/NYTM-0919-"${year_start}\
  --batchsize 4096
done





gpu=0
random_seed=2021
root_path_name=/mnt/data1/python/time_series_benchmark/TOTEM-master/
data_path_name=../../datasets/bigdatasets/NYTM-0919.csv
model_id_name=NYTM-0919
data_name=custom
seq_len=336

for year_start in 2017 2015 2013 2011 2009
do

for pred_len in 96
do
python -u forecasting/extract_forecasting_data.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 67 \
  --gpu $gpu\
  --save_path "forecasting/data/NYTM-0919-"${year_start}"/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path "forecasting/saved_models/NYTM-0919-"${year_start}"/CD64_CW256_CF4_BS4096_ITR15000/checkpoints/final_model.pth"\
  --compression_factor 4 \
  --date_split ${year_start}-01-01S00:00:00D2018-01-01S00:00:00D2019-01-01S00:00:00 \
  --batch_size 128 \
  --classifiy_or_forecast "forecast"
done
done



gpu=0
Tin=336
datatype=NYTM-0919

for year_start in 2017 2015 2013 2011 2009
do

for seed in 2021
do
for Tout in 96
do
python forecasting/train_forecaster.py \
  --data-type $datatype \
  --Tin $Tin \
  --Tout $Tout \
  --cuda-id $gpu \
  --seed $seed \
  --data_path "forecasting/data/"$datatype"-"${year_start}"/Tin"$Tin"_Tout"$Tout"" \
  --codebook_size 256 \
  --checkpoint \
  --epochs 100\
  --checkpoint_path "forecasting/saved_models/"$datatype"-"${year_start}"/forecaster_checkpoints/"$datatype"_Tin"$Tin"_Tout"$Tout"_seed"$seed""\
  --date_split ${year_start}-01-01S00:00:00D2018-01-01S00:00:00D2019-01-01S00:00:00 \
  --file_save_path "forecasting/results/"$datatype"-"${year_start}"/"
done
done
done