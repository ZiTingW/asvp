#！/bin/bash


dataset='imagenet'
dataset_path='./data/imagenet/'
selfmodel_path='./pretrain/res50_byol_eman_200ep.pth.tar'
feas_path='./pretrain/'
isamp='margin'
outpath_base='./res/'


for iexp in '1' '2' '3'
do	
	
	python train.py --expid $iexp --sampling_strategy $isamp --outpath_base $outpath_base --dataset_name $dataset --dataset_path $dataset_path --al_budget "[10000]*10+[50000]*6" --train_eps 130 --lr 0.1 --cls_lr 0.1 --weight_decay 0.0 --milestone '60,80' --early_stop 130 --batchsize_train 64 --grad_accu 4 --batchsize_al_forward 128 --batchsize_evaluation 128 --training_mode 2 --classifier_type "Linear" --classifier_dim '2048,2048,10' --network "res50" --selfmodel_path $selfmodel_path --totfeas_path $feas_path'totfeas.npy' --totlabel_path $feas_path'totlabel_img.npy' --totfeas_test_path $feas_path'totfeas_test.npy' --totlabel_test_path $feas_path'totlabel_img_test.npy'

done
		
python res_array.py --path_base $outpath_base --dataset_name $dataset --prefix1 $dataset"_"$isamp"_exp" --prefix2 "_training_strategy2" --num_exp "[i for i in range(1,4)]" --res_name "totacc_"$isamp"_ft.npy"