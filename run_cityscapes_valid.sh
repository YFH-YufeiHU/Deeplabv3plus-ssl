CUDA_VISIBLE_DEVICES=0  python3 valid.py --data_root "/home/student/workspace_Yufei/CityScapes/leftImg8bit/"\
                        --dataset "cityscapes" \
                        --model "deeplabv3plus_resnet50"\
                        --output_stride 8 \
                        --batch_size 12 \
                        --crop_size 68 \
                        --gpu_id 0 \
                        --lr 0.1 \
			                  --test_only \
			                  --ckpt './checkpoints_obow_deeplabv3plus_s15_PC/best_deeplabv3plus_resnet50_cityscapes_os8.pth' \
			                  --val_batch_size 1\
                        > cityscapes_GAN_valid.out

