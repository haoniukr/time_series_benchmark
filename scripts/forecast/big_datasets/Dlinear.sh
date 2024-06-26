#Same as the Transformer except model_name
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

for year_start in 2017 2015 2013 2011 2009
do

model_name=DLinear
model_id=$model_name'_'${year_start}

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
  --date_split ${year_start}-01-01S00:00:00D2018-01-01S00:00:00D2019-01-01S00:00:00 \
  --itr 1 >logs/Forecasting/NYTM-0919_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log  

done
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

for year_start in 2009 1999 1989 1979
do

model_name=DLinear
model_id=$model_name'_'${year_start}

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ../datasets/bigdatasets/ \a
  --data_path 2m-temperature-NA.csv \
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
  --enc_in 126 \
  --dec_in 126 \
  --c_out 126 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --learning_rate 0.0001\
  --batch_size 64\
  --train_epochs 10 \
  --patience 3 \
  --date_split ${year_start}-01-01S00:00:00D2016-01-01S00:00:00D2017-01-01S00:00:00 \
  --itr 1 >logs/Forecasting/2m-temperature-NA_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len.log  

done
done
done
done




# #Same as the Transformer except model_name
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
#   --root_path ../datasets/bigdatasets/ \
#   --data_path NYTM-0919.csv \
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
#   --enc_in 67 \
#   --dec_in 67 \
#   --c_out 67 \
#   --des 'Exp' \
#   --d_model 512\
#   --d_ff 512\
#   --learning_rate 0.0001\
#   --batch_size 128\
#   --train_epochs 10 \
#   --patience 3 \
#   --date_split 2017-01-01S00:00:00D2018-01-01S00:00:00D2019-01-01S00:00:00 \
#   --itr 1 >logs/Forecasting/NYTM-0919_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len'_'2017.log  


# python -u run.py \
#   --task_name forecast \
#   --is_training 1 \
#   --root_path ../datasets/bigdatasets/ \
#   --data_path NYTM-0919.csv \
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
#   --enc_in 67 \
#   --dec_in 67 \
#   --c_out 67 \
#   --des 'Exp' \
#   --d_model 512\
#   --d_ff 512\
#   --learning_rate 0.0001\
#   --batch_size 128\
#   --train_epochs 10 \
#   --patience 3 \
#   --date_split 2015-01-01S00:00:00D2018-01-01S00:00:00D2019-01-01S00:00:00 \
#   --itr 1 >logs/Forecasting/NYTM-0919_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len'_'2015.log 

# python -u run.py \
#   --task_name forecast \
#   --is_training 1 \
#   --root_path ../datasets/bigdatasets/ \
#   --data_path NYTM-0919.csv \
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
#   --enc_in 67 \
#   --dec_in 67 \
#   --c_out 67 \
#   --des 'Exp' \
#   --d_model 512\
#   --d_ff 512\
#   --learning_rate 0.0001\
#   --batch_size 128\
#   --train_epochs 10 \
#   --patience 3 \
#   --date_split 2013-01-01S00:00:00D2018-01-01S00:00:00D2019-01-01S00:00:00 \
#   --itr 1 >logs/Forecasting/NYTM-0919_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len'_'2013.log 

# python -u run.py \
#   --task_name forecast \
#   --is_training 1 \
#   --root_path ../datasets/bigdatasets/ \
#   --data_path NYTM-0919.csv \
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
#   --enc_in 67 \
#   --dec_in 67 \
#   --c_out 67 \
#   --des 'Exp' \
#   --d_model 512\
#   --d_ff 512\
#   --learning_rate 0.0001\
#   --batch_size 128\
#   --train_epochs 10 \
#   --patience 3 \
#   --date_split 2011-01-01S00:00:00D2018-01-01S00:00:00D2019-01-01S00:00:00 \
#   --itr 1 >logs/Forecasting/NYTM-0919_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len'_'2011.log 
  
# python -u run.py \
#   --task_name forecast \
#   --is_training 1 \
#   --root_path ../datasets/bigdatasets/ \
#   --data_path NYTM-0919.csv \
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
#   --enc_in 67 \
#   --dec_in 67 \
#   --c_out 67 \
#   --des 'Exp' \
#   --d_model 512\
#   --d_ff 512\
#   --learning_rate 0.0001\
#   --batch_size 128\
#   --train_epochs 10 \
#   --patience 3 \
#   --date_split 2009-01-01S00:00:00D2018-01-01S00:00:00D2019-01-01S00:00:00 \
#   --itr 1 >logs/Forecasting/NYTM-0919_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len'_'2009.log   
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
#   --root_path ../datasets/bigdatasets/ \a
#   --data_path 2m-temperature-NA.csv \
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
#   --enc_in 126 \
#   --dec_in 126 \
#   --c_out 126 \
#   --des 'Exp' \
#   --d_model 512\
#   --d_ff 512\
#   --learning_rate 0.0001\
#   --batch_size 64\
#   --train_epochs 10 \
#   --patience 3 \
#   --date_split 2009-01-01S00:00:00D2016-01-01S00:00:00D2017-01-01S00:00:00 \
#   --itr 1 >logs/Forecasting/2m-temperature-NA_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len'_'2009.log  


# python -u run.py \
#   --task_name forecast \
#   --is_training 1 \
#   --root_path ../datasets/bigdatasets/ \
#   --data_path 2m-temperature-NA.csv \
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
#   --enc_in 126 \
#   --dec_in 126 \
#   --c_out 126 \
#   --des 'Exp' \
#   --d_model 512\
#   --d_ff 512\
#   --learning_rate 0.0001\
#   --batch_size 64\
#   --train_epochs 10 \
#   --patience 3 \
#   --date_split 1999-01-01S00:00:00D2016-01-01S00:00:00D2017-01-01S00:00:00 \
#   --itr 1 >logs/Forecasting/2m-temperature-NA_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len'_'1999.log 

# python -u run.py \
#   --task_name forecast \
#   --is_training 1 \
#   --root_path ../datasets/bigdatasets/ \
#   --data_path 2m-temperature-NA.csv \
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
#   --enc_in 126 \
#   --dec_in 126 \
#   --c_out 126 \
#   --des 'Exp' \
#   --d_model 512\
#   --d_ff 512\
#   --learning_rate 0.0001\
#   --batch_size 64\
#   --train_epochs 10 \
#   --patience 3 \
#   --date_split 1989-01-01S00:00:00D2016-01-01S00:00:00D2017-01-01S00:00:00 \
#   --itr 1 >logs/Forecasting/2m-temperature-NA_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len'_'1989.log 

# python -u run.py \
#   --task_name forecast \
#   --is_training 1 \
#   --root_path ../datasets/bigdatasets/ \
#   --data_path 2m-temperature-NA.csv \
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
#   --enc_in 126 \
#   --dec_in 126 \
#   --c_out 126 \
#   --des 'Exp' \
#   --d_model 512\
#   --d_ff 512\
#   --learning_rate 0.0001\
#   --batch_size 64\
#   --train_epochs 10 \
#   --patience 3 \
#   --date_split 1979-01-01S00:00:00D2016-01-01S00:00:00D2017-01-01S00:00:00 \
#   --itr 1 >logs/Forecasting/2m-temperature-NA_$model_id'_'1'_'$random_seed'_'$seq_len'_'$pred_len'_'1979.log 
  
# done
# done
# done