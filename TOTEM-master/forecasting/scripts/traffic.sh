seq_len=96
root_path_name=/mnt/data1/python/time_series_benchmark/TOTEM-master/
data_path_name=../../datasets/traffic.csv
data_name=custom
random_seed=2021
pred_len=96
gpu=0

python -u forecasting/save_revin_data.py\
  --random_seed $random_seed \
  --data $data_name \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 862 \
  --batch_size 1024 \
  --gpu $gpu\
  --save_path "forecasting/data/traffic"

gpu=0
python forecasting/train_vqvae.py \
  --config_path forecasting/scripts/traffic.json \
  --model_init_num_gpus $gpu \
  --data_init_cpu_or_gpu cpu \
  --comet_log \
  --comet_tag pipeline \
  --comet_name vqvae_traffic \
  --save_path "forecasting/saved_models/traffic/"\
  --base_path "forecasting/data"\
  --batchsize 4096
  
  

random_seed=2021
root_path_name=/mnt/data1/python/time_series_benchmark/TOTEM-master/
data_path_name=../../datasets/traffic.csv
model_id_name=traffic
data_name=custom
seq_len=336
gpu=0
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
  --enc_in 862 \
  --gpu $gpu\
  --save_path "forecasting/data/traffic/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path 'forecasting/saved_models/traffic/CD64_CW256_CF4_BS4096_ITR15000/checkpoints/final_model.pth'\
  --compression_factor 4 \
  --batch_size 24 \
  --classifiy_or_forecast "forecast"
done

gpu=0
Tin=336
datatype=traffic
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
  --data_path "forecasting/data/"$datatype"/Tin"$Tin"_Tout"$Tout"" \
  --codebook_size 256 \
  --checkpoint \
  --checkpoint_path "forecasting/saved_models/"$datatype"/forecaster_checkpoints/"$datatype"_Tin"$Tin"_Tout"$Tout"_seed"$seed""\
  --file_save_path "forecasting/results/"$datatype"/"
done
done


random_seed=2021
root_path_name=/mnt/data1/python/time_series_benchmark/TOTEM-master/
data_path_name=../../datasets/traffic.csv
model_id_name=traffic
data_name=custom
seq_len=12
gpu=0
for pred_len in 12
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
  --enc_in 862 \
  --gpu $gpu\
  --save_path "forecasting/data/traffic/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path 'forecasting/saved_models/traffic/CD64_CW256_CF4_BS4096_ITR15000/checkpoints/final_model.pth'\
  --compression_factor 4 \
  --batch_size 64 \
  --classifiy_or_forecast "forecast"
done

gpu=0
Tin=12
datatype=traffic
for seed in 2021
do
for Tout in 12
do
python forecasting/train_forecaster.py \
  --data-type $datatype \
  --Tin $Tin \
  --Tout $Tout \
  --cuda-id $gpu \
  --seed $seed \
  --data_path "forecasting/data/"$datatype"/Tin"$Tin"_Tout"$Tout"" \
  --codebook_size 256 \
  --checkpoint \
  --loss_type mae \
  --loss_inverse \
  --checkpoint_path "forecasting/saved_models/"$datatype"/forecaster_checkpoints/"$datatype"_Tin"$Tin"_Tout"$Tout"_seed"$seed""\
  --file_save_path "forecasting/results/"$datatype"/"
done
done




##############################################

random_seed=2021
root_path_name=/mnt/data1/python/time_series_benchmark/TOTEM-master/
data_path_name=../../datasets/traffic.csv
model_id_name=traffic
data_name=custom
seq_len=336
gpu=0
for pred_len in 720
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
  --enc_in 862 \
  --gpu $gpu\
  --save_path "forecasting/data/traffic/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path 'forecasting/saved_models/traffic/CD64_CW256_CF4_BS4096_ITR15000/checkpoints/final_model.pth'\
  --compression_factor 4 \
  --batch_size 24 \
  --classifiy_or_forecast "forecast"
done

gpu=0
Tin=336
datatype=traffic
for seed in 2021
do
for Tout in 720
do
python forecasting/train_forecaster.py \
  --data-type $datatype \
  --Tin $Tin \
  --Tout $Tout \
  --cuda-id $gpu \
  --seed $seed \
  --data_path "forecasting/data/"$datatype"/Tin"$Tin"_Tout"$Tout"" \
  --codebook_size 256 \
  --checkpoint \
  --checkpoint_path "forecasting/saved_models/"$datatype"/forecaster_checkpoints/"$datatype"_Tin"$Tin"_Tout"$Tout"_seed"$seed""\
  --file_save_path "forecasting/results/"$datatype"/"
done
done
