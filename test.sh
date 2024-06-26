# if [ ! -d "./logs" ]; then
#     mkdir ./logs
# fi
# if [ ! -d "./logs/Forecasting" ]; then
#     mkdir ./logs/Forecasting
# fi

# for random_seed in 2021
# do

# for seq_len in 12
# do

# if [ $seq_len==12 ]
# then
#   label_len=6
# else
#   label_len=48
# fi

# for pred_len in 12
# do

# model_name=DLinear
# model_id=test

# python -u run.py \
#   --task_name forecast \
#   --is_training 1 \
#   --root_path ../datasets/ \
#   --data_path weather.csv \
#   --model_id $model_id \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len $label_len \
#   --pred_len $pred_len \
#   --e_layers 3 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --des 'Exp' \
#   --d_model 512\
#   --d_ff 512\
#   --learning_rate 0.0001\
#   --batch_size 16\
#   --train_epochs 10 \
#   --patience 3 \
#   --loss_type mae \
#   --loss_inverse \
#   --itr 1 >logs/Forecasting/weather_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log  
  

# done
# done
# done















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

model_name=iTransformer
model_id=$model_name

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ../datasets/bigdatasets/ \
  --data_path NYTM-0919.csv \
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
  --enc_in 67 \
  --dec_in 67 \
  --c_out 67 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --learning_rate 0.0001\
  --batch_size 128\
  --train_epochs 10 \
  --patience 3 \
  --date_split 2017-01-01S00:00:00D2018-01-01S00:00:00D2019-01-01S00:00:00 \
  --itr 1 
  
  #>logs/Forecasting/NYTM-0919_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len_2017.log  
  
done
done
done

































# if [ ! -d "./logs" ]; then
#     mkdir ./logs
# fi
# if [ ! -d "./logs/Forecasting" ]; then
#     mkdir ./logs/Forecasting
# fi

# for random_seed in 2021
# do

# for seq_len in 336
# do

# if [ $seq_len==12 ]
# then
#   label_len=6
# else
#   label_len=48
# fi

# for pred_len in 96
# do

# model_name=DLinear
# model_id=$model_name

# python -u run.py \
#   --task_name forecast \
#   --is_training 1 \
#   --root_path ../datasets/ \
#   --data_path weather.csv \
#   --model_id $model_id \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len $label_len \
#   --pred_len $pred_len \
#   --e_layers 3 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --des 'Exp' \
#   --d_model 512\
#   --d_ff 512\
#   --learning_rate 0.001\
#   --batch_size 16\
#   --itr 1 >logs/Forecasting/weather_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log  

# done
# done
# done

# if [ ! -d "./logs" ]; then
#     mkdir ./logs
# fi
# if [ ! -d "./logs/Forecasting" ]; then
#     mkdir ./logs/Forecasting
# fi

# for random_seed in 2021
# do

# for seq_len in 12
# do

# if [ $seq_len==12 ]
# then
#   label_len=6
# else
#   label_len=48
# fi

# for pred_len in 12
# do

# model_name=DLinear
# model_id=$model_name

# python -u run.py \
#   --task_name forecast \
#   --is_training 1 \
#   --root_path ../datasets/ \
#   --data_path weather.csv \
#   --model_id $model_id \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len $label_len \
#   --pred_len $pred_len \
#   --e_layers 3 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --des 'Exp' \
#   --d_model 512\
#   --d_ff 512\
#   --inverse \
#   --learning_rate 0.001\
#   --batch_size 16\
#   --loss_type mae \
#   --itr 1 >logs/Forecasting/weather_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log  

# done
# done
# done



# if [ ! -d "./logs" ]; then
#     mkdir ./logs
# fi
# if [ ! -d "./logs/Forecasting" ]; then
#     mkdir ./logs/Forecasting
# fi

# for random_seed in 2021
# do

# for seq_len in 12
# do

# if [ $seq_len==12 ]
# then
#   label_len=6
# else
#   label_len=48
# fi

# for pred_len in 12
# do

# model_name=MegaCRN
# model_id=$model_name

# python -u run.py \
#   --task_name forecast \
#   --is_training 1 \
#   --root_path ../datasets/ \
#   --data_path weather.csv \
#   --model_id $model_id \
#   --model $model_name \
#   --data custom_extension \
#   --features M \
#   --seq_len $seq_len \
#   --label_len $label_len \
#   --pred_len $pred_len \
#   --e_layers 3 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --des 'Exp' \
#   --d_model 512\
#   --d_ff 512\
#   --inverse \
#   --learning_rate 0.01\
#   --batch_size 64\
#   --loss_type mae \
#   --train_epochs 200 \
#   --patience 10 \
#   --data_missing \
#   --itr 1 >logs/Forecasting/weather_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log  

# done
# done
# done





# if [ ! -d "./logs" ]; then
#     mkdir ./logs
# fi
# if [ ! -d "./logs/Forecasting" ]; then
#     mkdir ./logs/Forecasting
# fi

# for random_seed in 2021
# do

# for seq_len in 12
# do

# if [ $seq_len==12 ]
# then
#   label_len=6
# else
#   label_len=48
# fi

# for pred_len in 12
# do

# model_name=MegaCRN
# model_id=$model_name

# python -u run.py \
#   --task_name forecast \
#   --is_training 1 \
#   --root_path ../datasets/ \
#   --data_path metr-la.csv \
#   --model_id $model_id \
#   --model $model_name \
#   --data custom_extension \
#   --features M \
#   --seq_len $seq_len \
#   --label_len $label_len \
#   --pred_len $pred_len \
#   --e_layers 3 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 207 \
#   --dec_in 207 \
#   --c_out 207 \
#   --des 'Exp' \
#   --d_model 512\
#   --d_ff 512\
#   --inverse \
#   --learning_rate 0.01\
#   --batch_size 64\
#   --loss_type mae \
#   --train_epochs 200 \
#   --patience 10 \
#   --data_missing \
#   --itr 1 >logs/Forecasting/metr-la_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log  

# done
# done
# done



