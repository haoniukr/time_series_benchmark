if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./logs/Forecasting" ]; then
    mkdir ./logs/Forecasting
fi

for random_seed in 2021
do

for seq_len in 336
do

for pred_len in 96
do

if [ $pred_len -eq 12 ]
then
  label_len=6
else
  label_len=48
fi

for year_start in 1979
do

model_name=TimesNet
model_id=$model_name'_'${year_start}

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ../datasets/bigdatasets/ \
  --data_path 2m-temperature-NA.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 126 \
  --dec_in 126 \
  --c_out 126 \
  --des 'Exp' \
  --d_model 192\
  --d_ff 192\
  --top_k 5 \
  --learning_rate 0.0001\
  --batch_size 64\
  --train_epochs 1 \
  --patience 3 \
  --date_split ${year_start}-01-01S00:00:00D2016-01-01S00:00:00D2017-01-01S00:00:00 \
  --itr 1 >logs/Forecasting/2m-temperature-NA_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log  

done
done
done
done