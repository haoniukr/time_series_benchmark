
MegaCRN
-data format: [B,L,N,2] 2:[value,time_in_day]
 --Dataset_Custom_Extension

-loss: loss1+loss2+loss3



Long-term:
 train val : normalized values
 test: both
 loss: mse
 
Long-term:
 train val : original values (args: --inverse)
 test: both
 loss: mae (masked)
 
