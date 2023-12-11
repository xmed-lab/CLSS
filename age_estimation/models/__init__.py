from .resnet50 import resnet50, \
                    resnet50_CTR, resnet50_CTRNP, resnet50_UCVME,\
                    SupConLoss_admargin, \
                        SupConLoss_admargin_semi, \
                            SupConLoss_ctrv2, \
                                SupConLoss_ctrv2_semi, \
                                    SupConLoss_admargin_val, \
                                        SupConLoss_ctrv2_val, \
                                            SupConLoss_admargin_semi_lbulbsplit, \
                                                SupConLoss_admargin_semi_lbulbsplit_ranksim, \
                                                    SupConLoss_admargin_semi_lbulbsplit_ranksim_pslb, \
                                                        SupConLoss_admargin_semi_lbulbsplit_ranksim_pslb_sng


from .OrdinalEntropy import ordinal_entropy