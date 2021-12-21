CUDA_VISIBLE_DEVICES=0,1  python3 main.py --data_root "/home/student/workspace_Yufei/CityScapes/leftImg8bit/"\
                        --dataset "cityscapes" \
                        --model "deeplabv3plus_resnet50"\
                        --output_stride 8 \
                        --batch_size 12 \
                        --crop_size 768 \
                        --gpu_id 0,1 \
                        --lr 0.1 \
			--ckpt './pre_models/model_best_rms_7.55890_depth_deeplab.pth'\
			> cityscapes_best_obow_resnet_imagenet_depth_train_extra_p9.out
