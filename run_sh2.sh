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

if [ $seq_len==12 ]
then
  label_len=6
else
  label_len=48
fi

for pred_len in 96
do

model_name=MegaCRN
model_id=$model_name

#P100
python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ../datasets/ \
  --data_path weather.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom_extension \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --learning_rate 0.01\
  --batch_size 128\
  --train_epochs 200 \
  --patience 10 \
  --itr 1 >logs/Forecasting/weather_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log  

done
done
done