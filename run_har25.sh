CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.25 --balance --log_dir default_settings_HAR.25_seft DeepSetAttentionModel

CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.25 --balance --log_dir default_settings_HAR.25_ipnets InterpolationPredictionModel

CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.25 --balance --log_dir default_settings_HAR.25_grud GRUDModel

CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.25 --balance --log_dir default_settings_HAR.25_transformer TransformerModel
