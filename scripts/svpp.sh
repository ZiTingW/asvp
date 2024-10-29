#！/bin/bash



python /root/autodl-tmp/mlpproxy/res_array.py --path_base "/root/autodl-tmp/mlpproxy/res/" --dataset_name "cifar10" --prefix1 "cifar10_ActiveFT(al)_exp" --prefix2 "_r50byoleman2_mlpproxy_ft_training_strategy2" --num_exp "[i for i in range(1,4)]" --res_name "totacc_ActiveFT_al_r50byoleman2_mlpproxy_ft.npy"

for iexp in '1' '2' '3' 
do	
	python /root/autodl-tmp/mlpproxy/train.py --expid $iexp"_r50byoleman2_ft" --sampling_strategy "ActiveFT(al)" --outpath_base '/root/autodl-tmp/mlpproxy/res/' --dataset_name "cifar10" --dataset_path '/root/autodl-tmp/data/cifar10/' --al_budget "[200]*10" --train_eps 130 --lr 0.1 --cls_lr 0.1 --weight_decay 0 --milestone '60,80' --early_stop 200 --batchsize_train 256 --grad_accu 1 --batchsize_al_forward 256 --batchsize_evaluation 256 --training_mode 2 --classifier_type "Linear" --classifier_dim '2048,2048,10' --network "res50" --selfmodel_path "/root/autodl-nas/res50_byol_eman_200ep.pth.tar" --totfeas_path '/root/autodl-tmp/mlpproxy/selfsup_feas/ftfeas.npy' 
	
done

python /root/autodl-tmp/mlpproxy/res_array.py --path_base "/root/autodl-tmp/mlpproxy/res/" --dataset_name "cifar10" --prefix1 "cifar10_ActiveFT(al)_exp" --prefix2 "_r50byoleman2_ft_training_strategy2" --num_exp "[i for i in range(1,4)]" --res_name "totacc_ActiveFT_al_r50byoleman2_ft.npy"


for iexp in '1' '2' '3'
do	

	python /root/autodl-tmp/mlpproxy/ft_mlpproxy.py --expid $iexp"_r50byoleman2_mlpproxy_ft" --sampling_strategy 'ActiveFT(self)' --outpath_base '/root/autodl-tmp/mlpproxy/res/' --dataset_name "cifar10" --dataset_path '/root/autodl-tmp/data/cifar10/' --train_eps 130 --lr 0.1 --cls_lr 0.1 --weight_decay 0 --milestone '60,80' --early_stop 200 --batchsize_train 256 --grad_accu 1 --batchsize_al_forward 256 --batchsize_evaluation 256 --training_mode 2 --classifier_type "Linear" --classifier_dim '2048,2048,10' --al_budget "[200]*10" --network "res50" --selfmodel_path "/root/autodl-nas/res50_byol_eman_200ep.pth.tar" --alidxpath "/root/autodl-tmp/mlpproxy/alidx/activeft_self/alidx"$iexp".npy"

done

python /root/autodl-tmp/mlpproxy/res_array.py --path_base "/root/autodl-tmp/mlpproxy/res/" --dataset_name "cifar10" --prefix1 "cifar10_ActiveFT(self)_exp" --prefix2 "_r50byoleman2_ft_training_strategy2" --num_exp "[i for i in range(1,4)]" --res_name "totacc_ActiveFT_self_r50byoleman2_ft.npy"
