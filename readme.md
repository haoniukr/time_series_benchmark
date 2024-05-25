Existing models:
-add in model.py    or self.task_name == 'forecast'
-revise in exp_basic.py
 


MegaCRN
-data format: [B,L,N,2] 2:[value,time_in_day]
 --Dataset_Custom_Extension

-loss: loss1+loss2+loss3



Long-term:
 train val : normalized values
 test: both
 loss: mse
 
short-term:
 train val : original values (args: --inverse)
 test: both
 loss: mae (masked)
 
 
 
 A100:
 TOTEM expect weather
 Some MegaCRN long-term predicitons
 
