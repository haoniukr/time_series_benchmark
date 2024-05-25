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
  --checkpoint_path "forecasting/saved_models/"$datatype"/forecaster_checkpoints/"$datatype"_Tin"$Tin"_Tout"$Tout"_seed"$seed""\
  --file_save_path "forecasting/results/"$datatype"/"
done
done