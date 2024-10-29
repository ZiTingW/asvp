#！/bin/bash



dataset='imagenet'
dataset_path='./data/imagenet/'
selfmodel_path='./pretrain/res50_byol_eman_200ep.pth.tar'
feas_path='./pretrain/'
isamp='margin'
outpath_base='./res/'

ipt='15000'

for iexp in '1' '2' '3'
do	
	python /home/auv/Documents/ziting/code/fullcover_new/extract_ftfeas.py --outpath_base $outpath_base --dataset_name $dataset --dataset_path $dataset_path --checkpoints_path $outpath_base$dataset"/"$dataset"_"$isamp"_exp"$iexp"_svpp_ft_training_strategy2/checkpoint_"$ipt"_.pth.tar" --batchsize 128 --network 'res50'
	
	python train.py --expid $iexp"_asvp_proxy" --sampling_strategy $isamp --outpath_base $outpath_base --dataset_name 'feas' --al_budget "[10000]*10+[50000]*6" --train_eps 50 --lr 0.1 --cls_lr 0.1 --weight_decay 0 --milestone '30,40' --early_stop 50 --batchsize_train 2048 --grad_accu 1 --batchsize_al_forward 4096 --batchsize_evaluation 4096 --training_mode 0 --classifier_type "MLP" --classifier_dim '2048,2048,1000' --network "res50" --selfmodel_path $selfmodel_path --totfeas_path $outpath_base'ftfeas.npy' --totlabel_path $feas_path'totlabel_img.npy' --totfeas_test_path $outpath_base'ftfeas_test.npy' --totlabel_test_path $feas_path'totlabel_img_test.npy'
	
	python ft_mlpproxy.py --expid $iexp"_asvp_ft" --sampling_strategy $isamp --outpath_base $outpath_base --dataset_name $dataset --dataset_path $dataset_path --train_eps 50 --lr 0.01 --cls_lr 0.1 --weight_decay 0.0001 --milestone '30,40' --early_stop 50 --batchsize_train 128 --grad_accu 1 --batchsize_al_forward 512 --batchsize_evaluation 512 --training_mode 2 --classifier_type "Linear" --classifier_dim '2048,2048,1000' --al_budget "[20000] + [10000]*8 + [50000]*6" --network "res50" --selfmodel_path $selfmodel_path --alidxpath $outpath_base"feas/feas_"$isamp"_exp"$iexp"_asvp_proxy_training_strategy0/alidx.npy"

done
		
python res_array.py --path_base $outpath_base --dataset_name $dataset --prefix1 $dataset"_"$isamp"_exp" --prefix2 "_asvp_ft_training_strategy2" --num_exp "[i for i in range(1,4)]" --res_name "totacc_"$isamp"_asvp.npy"

