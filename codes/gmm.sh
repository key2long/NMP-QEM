# CUDA_VISIBLE_DEVICES=3 python main.py --cuda --do_train --do_test \
#   --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 -g 42 \
#   -lr 0.00007 --max_steps 800001 --cpu_num 4 --geo gmm --valid_steps 15000 \
#   --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up.2in.3in.inp.pin.pni" --exp_info "FB15k-237"


# CUDA_VISIBLE_DEVICES=1 python main.py --cuda --do_train --do_test \
#   --data_path data/NELL-betae -n 128 -b 512 -d 400 -g 30 \
#   -lr 0.00005 --max_steps 500001 --cpu_num 4 --geo gmm --valid_steps 15000 \
#   --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up.2in.3in.inp.pin.pni" --exp_info "NELL995"


# CUDA_VISIBLE_DEVICES=1 python main.py --cuda --do_train --do_test \
#   --data_path data/wn18rr -n 128 -b 512 -d 400 -g 24 \
#   -lr 0.00005 --max_steps 500001 --cpu_num 4 --geo gmm --valid_steps 15000 \
#   --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up.2in.3in.inp.pin.pni" --exp_info "wn18rr"


################################ evaluation ##########################################
# CUDA_VISIBLE_DEVICES=7 python main.py --cuda --do_test \
#   --data_path data/NELL-betae -n 128 -b 512 -d 400 -g 42 \
#   -lr 0.00005 --max_steps 600001 --cpu_num 4 --geo gmm --valid_steps 15000 \
#   --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up.2in.3in.inp.pin.pni" --checkpoint_path ......
######################################################################################
