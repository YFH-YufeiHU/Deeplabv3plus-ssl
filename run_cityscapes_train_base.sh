CUDA_VISIBLE_DEVICES=0,1  python3 main.py --data_root "/home/student/workspace_Yufei/CityScapes/leftImg8bit/"\
                        --dataset "cityscapes" \
                        --model "deeplabv3plus_resnet50"\
                        --output_stride 8 \
                        --batch_size 12 \
                        --crop_size 768 \
                        --gpu_id 0,1 \
                        --lr 0.1 \
                        --checkpoint_root './pre_models/best_ckpt_base.pth'\
                        > cityscapes_GAN_Descriptor_base.out
