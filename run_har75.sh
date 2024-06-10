CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.75 --balance --log_dir default_settings_HAR.75_seft DeepSetAttentionModel

CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.75 --balance --log_dir default_settings_HAR.75_ipnets InterpolationPredictionModel

CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.75 --balance --log_dir default_settings_HAR.75_grud GRUDModel

CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.75 --balance --log_dir default_settings_HAR.75_transformer TransformerModel
