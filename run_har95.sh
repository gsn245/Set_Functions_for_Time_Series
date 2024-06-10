CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.95 --balance --log_dir default_settings_HAR.95_seft DeepSetAttentionModel

CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.95 --balance --log_dir default_settings_HAR.95_ipnets InterpolationPredictionModel

CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.95 --balance --log_dir default_settings_HAR.95_grud GRUDModel

CUDA_VISIBLE_DEVICES=-1 poetry run seft_fit_model --dataset HARdata_.95 --balance --log_dir default_settings_HAR.95_transformer TransformerModel
