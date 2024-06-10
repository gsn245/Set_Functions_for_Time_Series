CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.5 --balance --log_dir default_settings_HAR.5_seft DeepSetAttentionModel

CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.5 --balance --log_dir default_settings_HAR.5_ipnets InterpolationPredictionModel

CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.5 --balance --log_dir default_settings_HAR.5_grud GRUDModel

CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.5 --balance --log_dir default_settings_HAR.5_transformer TransformerModel
