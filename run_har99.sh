CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.99 --balance --log_dir default_settings_HAR.99_seft DeepSetAttentionModel

CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.99 --balance --log_dir default_settings_HAR.99_ipnets InterpolationPredictionModel

CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.99 --balance --log_dir default_settings_HAR.99_grud GRUDModel

CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.99 --balance --log_dir default_settings_HAR.99_transformer TransformerModel
