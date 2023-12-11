
# 20%

export CUDA_VISIBLE_DEVICES=4,5
{ nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb20pc_lr1e3_s1 --num_epochs=30 --reduced_set --rd_label=84 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=5 --val_samp=1 --lr=0.001 --seed=1 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb20pc_lr1e3_s1.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb20pc_lr1e3_s2 --num_epochs=30 --reduced_set --rd_label=84 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=5 --val_samp=1 --lr=0.001 --seed=2 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb20pc_lr1e3_s2.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb20pc_lr1e3_s3 --num_epochs=30 --reduced_set --rd_label=84 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=5 --val_samp=1 --lr=0.001 --seed=3 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb20pc_lr1e3_s3.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb20pc_lr1e3_s4 --num_epochs=30 --reduced_set --rd_label=84 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=5 --val_samp=1 --lr=0.001 --seed=4 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb20pc_lr1e3_s4.log 2>&1 ;} &

export CUDA_VISIBLE_DEVICES=6,7
{ nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb20pc_lr1e3_s5 --num_epochs=30 --reduced_set --rd_label=84 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=5 --val_samp=1 --lr=0.001 --seed=5 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb20pc_lr1e3_s5.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb20pc_lr1e3_s6 --num_epochs=30 --reduced_set --rd_label=84 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=5 --val_samp=1 --lr=0.001 --seed=6 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb20pc_lr1e3_s6.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb20pc_lr1e3_s7 --num_epochs=30 --reduced_set --rd_label=84 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=5 --val_samp=1 --lr=0.001 --seed=7 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb20pc_lr1e3_s7.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb20pc_lr1e3_s8 --num_epochs=30 --reduced_set --rd_label=84 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=5 --val_samp=1 --lr=0.001 --seed=8 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb20pc_lr1e3_s8.log 2>&1 ;} &



# 25%


export CUDA_VISIBLE_DEVICES=4,5
{ nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb25pc_lr1e3_s1 --num_epochs=30 --reduced_set --rd_label=105 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=4 --val_samp=1 --lr=0.001 --seed=1 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb25pc_lr1e3_s1.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb25pc_lr1e3_s2 --num_epochs=30 --reduced_set --rd_label=105 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=4 --val_samp=1 --lr=0.001 --seed=2 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb25pc_lr1e3_s2.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb25pc_lr1e3_s3 --num_epochs=30 --reduced_set --rd_label=105 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=4 --val_samp=1 --lr=0.001 --seed=3 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb25pc_lr1e3_s3.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb25pc_lr1e3_s4 --num_epochs=30 --reduced_set --rd_label=105 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=4 --val_samp=1 --lr=0.001 --seed=4 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb25pc_lr1e3_s4.log 2>&1 ;} &

export CUDA_VISIBLE_DEVICES=6,7
{ nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb25pc_lr1e3_s5 --num_epochs=30 --reduced_set --rd_label=105 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=4 --val_samp=1 --lr=0.001 --seed=5 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb25pc_lr1e3_s5.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb25pc_lr1e3_s6 --num_epochs=30 --reduced_set --rd_label=105 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=4 --val_samp=1 --lr=0.001 --seed=6 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb25pc_lr1e3_s6.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb25pc_lr1e3_s7 --num_epochs=30 --reduced_set --rd_label=105 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=4 --val_samp=1 --lr=0.001 --seed=7 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb25pc_lr1e3_s7.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb25pc_lr1e3_s8 --num_epochs=30 --reduced_set --rd_label=105 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=4 --val_samp=1 --lr=0.001 --seed=8 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb25pc_lr1e3_s8.log 2>&1 ;} &



# 33%


export CUDA_VISIBLE_DEVICES=4,5
{ nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb33pc_lr1e3_s1 --num_epochs=30 --reduced_set --rd_label=140 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=3 --val_samp=1 --lr=0.001 --seed=1 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb33pc_lr1e3_s1.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb33pc_lr1e3_s2 --num_epochs=30 --reduced_set --rd_label=140 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=3 --val_samp=1 --lr=0.001 --seed=2 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb33pc_lr1e3_s2.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb33pc_lr1e3_s3 --num_epochs=30 --reduced_set --rd_label=140 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=3 --val_samp=1 --lr=0.001 --seed=3 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb33pc_lr1e3_s3.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb33pc_lr1e3_s4 --num_epochs=30 --reduced_set --rd_label=140 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=3 --val_samp=1 --lr=0.001 --seed=4 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb33pc_lr1e3_s4.log 2>&1 ;} &

export CUDA_VISIBLE_DEVICES=6,7
{ nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb33pc_lr1e3_s5 --num_epochs=30 --reduced_set --rd_label=140 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=3 --val_samp=1 --lr=0.001 --seed=5 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb33pc_lr1e3_s5.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb33pc_lr1e3_s6 --num_epochs=30 --reduced_set --rd_label=140 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=3 --val_samp=1 --lr=0.001 --seed=6 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb33pc_lr1e3_s6.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb33pc_lr1e3_s7 --num_epochs=30 --reduced_set --rd_label=140 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=3 --val_samp=1 --lr=0.001 --seed=7 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb33pc_lr1e3_s7.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb33pc_lr1e3_s8 --num_epochs=30 --reduced_set --rd_label=140 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=3 --val_samp=1 --lr=0.001 --seed=8 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb33pc_lr1e3_s8.log 2>&1 ;} &






# 50%

export CUDA_VISIBLE_DEVICES=4,5
{ nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb50pc_lr1e3_s1 --num_epochs=30 --reduced_set --rd_label=210 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=2 --val_samp=1 --lr=0.001 --seed=1 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb50pc_lr1e3_s1.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb50pc_lr1e3_s2 --num_epochs=30 --reduced_set --rd_label=210 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=2 --val_samp=1 --lr=0.001 --seed=2 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb50pc_lr1e3_s2.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb50pc_lr1e3_s3 --num_epochs=30 --reduced_set --rd_label=210 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=2 --val_samp=1 --lr=0.001 --seed=3 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb50pc_lr1e3_s3.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb50pc_lr1e3_s4 --num_epochs=30 --reduced_set --rd_label=210 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=2 --val_samp=1 --lr=0.001 --seed=4 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb50pc_lr1e3_s4.log 2>&1 ;} &

export CUDA_VISIBLE_DEVICES=6,7
{ nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb50pc_lr1e3_s5 --num_epochs=30 --reduced_set --rd_label=210 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=2 --val_samp=1 --lr=0.001 --seed=5 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb50pc_lr1e3_s5.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb50pc_lr1e3_s6 --num_epochs=30 --reduced_set --rd_label=210 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=2 --val_samp=1 --lr=0.001 --seed=6 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb50pc_lr1e3_s6.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb50pc_lr1e3_s7 --num_epochs=30 --reduced_set --rd_label=210 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=2 --val_samp=1 --lr=0.001 --seed=7 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb50pc_lr1e3_s7.log 2>&1 ; \
nohup python3 vol_reg_oedNP_semi_ulbsep_ranksim_cps.py --run_test --quick_test --batch_size=16 --output=output/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb50pc_lr1e3_s8 --num_epochs=30 --reduced_set --rd_label=210 --rd_unlabel=0 --allsplt --w_cps=1 --ssl_mult=2 --val_samp=1 --lr=0.001 --seed=8 --w_ctr=0 --w_ctrulb_0=0 --w_ctrulb_1=0 --lambda_val=2  > logs/220513_odeNPCPS_wcps1_wctr0_wct0u0_wct1u0_lmb2_lb50pc_lr1e3_s8.log 2>&1 ;} &




