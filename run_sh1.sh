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

model_name=DLinear
model_id=$model_name

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ../datasets/ \
  --data_path electricity.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --learning_rate 0.001\
  --batch_size 16\
  --train_epochs 10 \
  --patience 3 \
  --itr 1 >logs/Forecasting/electricity_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log  

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ../datasets/ \
  --data_path weather.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom \
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
  --learning_rate 0.0001\
  --batch_size 16\
  --train_epochs 10 \
  --patience 3 \
  --itr 1 >logs/Forecasting/weather_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log  
  
python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ../datasets/ \
  --data_path traffic.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --learning_rate 0.05\
  --batch_size 16\
  --train_epochs 10 \
  --patience 3 \
  --itr 1 >logs/Forecasting/traffic_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log  

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ../datasets/ \
  --data_path metr-la.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 207 \
  --dec_in 207 \
  --c_out 207 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --learning_rate 0.001\
  --batch_size 16\
  --train_epochs 10 \
  --patience 3 \
  --data_missing \
  --itr 1 >logs/Forecasting/metr-la_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log
  
python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ../datasets/ \
  --data_path pems-bay.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 325 \
  --dec_in 325 \
  --c_out 325 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --learning_rate 0.001\
  --batch_size 16\
  --train_epochs 10 \
  --patience 3 \
  --data_missing \
  --itr 1 >logs/Forecasting/pems-bay_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log   

done
done
done





if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./logs/Forecasting" ]; then
    mkdir ./logs/Forecasting
fi

for random_seed in 2021
do

for seq_len in 12
do

if [ $seq_len==12 ]
then
  label_len=6
else
  label_len=48
fi

for pred_len in 12
do

model_name=DLinear
model_id=$model_name

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ../datasets/ \
  --data_path electricity.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --learning_rate 0.001\
  --batch_size 16\
  --train_epochs 10 \
  --patience 3 \
  --loss_type mae \
  --loss_inverse \
  --itr 1 >logs/Forecasting/electricity_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log  

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ../datasets/ \
  --data_path weather.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom \
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
  --learning_rate 0.0001\
  --batch_size 16\
  --train_epochs 10 \
  --patience 3 \
  --loss_type mae \
  --loss_inverse \
  --itr 1 >logs/Forecasting/weather_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log  
  
python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ../datasets/ \
  --data_path traffic.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --learning_rate 0.05\
  --batch_size 16\
  --train_epochs 10 \
  --patience 3 \
  --loss_type mae \
  --loss_inverse \
  --itr 1 >logs/Forecasting/traffic_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log  

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ../datasets/ \
  --data_path metr-la.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 207 \
  --dec_in 207 \
  --c_out 207 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --learning_rate 0.001\
  --batch_size 16\
  --train_epochs 10 \
  --patience 3 \
  --data_missing \
  --loss_type mae \
  --loss_inverse \
  --itr 1 >logs/Forecasting/metr-la_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log
  
python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ../datasets/ \
  --data_path pems-bay.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 325 \
  --dec_in 325 \
  --c_out 325 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --learning_rate 0.001\
  --batch_size 16\
  --train_epochs 10 \
  --patience 3 \
  --data_missing \
  --loss_type mae \
  --loss_inverse \
  --itr 1 >logs/Forecasting/pems-bay_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log   

done
done
done