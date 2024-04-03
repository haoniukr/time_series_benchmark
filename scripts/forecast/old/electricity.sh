if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./logs/Forecasting" ]; then
    mkdir ./logs/Forecasting
fi

for random_seed in 2021
do

for seq_len in 12 96
do

if [ $seq_len==12 ]
then
  label_len=6
else
  label_len=48
fi

for pred_len in 12 96
do

model_name=iTransformer
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
  --itr 1 >logs/Forecasting/electricity_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log


############## the data of MegaCRN is different
model_name=MegaCRN
model_id=$model_name

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ../datasets/ \
  --data_path electricity.csv \
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
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1 >logs/Forecasting/electricity_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log
  
  
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
  --itr 1 >logs/Forecasting/electricity_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log  
 
 
model_name=Autoformer
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
  --itr 1 >logs/Forecasting/electricity_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log  


model_name=Transformer
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
  --itr 1 >logs/Forecasting/electricity_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log  

done
done
done










#########################################MegaCRN
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./logs/Forecasting" ]; then
    mkdir ./logs/Forecasting
fi

for random_seed in 2021
do

for seq_len in 12 96
do

if [ $seq_len==12 ]
then
  label_len=6
else
  label_len=48
fi

for pred_len in 96
do

############## the data of MegaCRN is different
model_name=MegaCRN
model_id=$model_name

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ../datasets/ \
  --data_path electricity.csv \
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
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --batch_size 16\
  --itr 1 >logs/Forecasting/electricity_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log

done
done
done