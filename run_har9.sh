CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.9 --balance --log_dir default_settings_HAR.9_seft DeepSetAttentionModel

CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.9 --balance --log_dir default_settings_HAR.9_ipnets InterpolationPredictionModel

CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.9 --balance --log_dir default_settings_HAR.9_grud GRUDModel

CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.9 --balance --log_dir default_settings_HAR.9_transformer TransformerModel
