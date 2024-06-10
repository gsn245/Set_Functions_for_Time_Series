CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata --balance --log_dir default_settings_HAR_seft DeepSetAttentionModel

CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata --balance --log_dir default_settings_HAR_ipnets InterpolationPredictionModel

CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata --balance --log_dir default_settings_HAR_grud GRUDModel

CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata --balance --log_dir default_settings_HAR_transformer TransformerModel
