

export CUDA_VISIBLE_DEVICES=0
{ nohup python3 main_semi_cps.py --Yoe --fraction=0.2 --ulb_w=0.0005 > logs/main_semi_CPS_ULB_02_5e4.log 2>&1 ;} &
export CUDA_VISIBLE_DEVICES=3
{ nohup python3 main_semi_cps.py --Yoe --fraction=0.25 --ulb_w=0.0005 > logs/main_semi_CPS_ULB_5e4.log 2>&1 ;} &
export CUDA_VISIBLE_DEVICES=2
{ nohup python3 main_semi_cps.py --Yoe --fraction=0.33 --ulb_w=0.0005 > logs/main_semi_CPS_ULB_033_5e4.log 2>&1 ;} &
export CUDA_VISIBLE_DEVICES=3
{ nohup python3 main_semi_cps.py --Yoe --fraction=0.5 --ulb_w=0.0005 > logs/main_semi_CPS_ULB_05_5e4.log 2>&1 ;} &




export CUDA_VISIBLE_DEVICES=1
{ nohup python3 main_semi_ucvme.py --Yoe --fraction=0.25 --NonLinear --ulb_w=0.0005 > logs/main_semi_ucvme_ULB_025_5e4.log 2>&1 ;} &
export CUDA_VISIBLE_DEVICES=0
{ nohup python3 main_semi_ucvme.py --Yoe --fraction=0.2 --NonLinear --ulb_w=0.0005 > logs/main_semi_ucvme_ULB_02_5e4.log 2>&1 ;} &
export CUDA_VISIBLE_DEVICES=2
{ nohup python3 main_semi_ucvme.py --Yoe --fraction=0.33 --NonLinear --ulb_w=0.0005 > logs/main_semi_ucvme_ULB_033_5e4.log 2>&1 ;} &
export CUDA_VISIBLE_DEVICES=3
{ nohup python3 main_semi_ucvme.py --Yoe --fraction=0.5 --NonLinear --ulb_w=0.0005 > logs/main_semi_ucvme_ULB_05_5e4.log 2>&1 ;} &




export CUDA_VISIBLE_DEVICES=0
{ nohup python3 main_semi_psrank_ftrank.py --Yoe --fraction=0.2 > logs/main_semi_Yoe_02_NLin_pslftrnk.log 2>&1 ;} &
export CUDA_VISIBLE_DEVICES=4
{ nohup python3 main_semi_psrank_ftrank.py --Yoe --fraction=0.25 > logs/main_semi_Yoe_025_NLin_pslftrnk.log 2>&1 ;} &
export CUDA_VISIBLE_DEVICES=2
{ nohup python3 main_semi_psrank_ftrank.py --Yoe --fraction=0.33 > logs/main_semi_Yoe_033_NLin_pslftrnk.log 2>&1 ;} &
export CUDA_VISIBLE_DEVICES=6
{ nohup python3 main_semi_psrank_ftrank.py --Yoe --fraction=0.5 > logs/main_semi_Yoe_05_NLin_pslftrnk.log 2>&1 ;} &





