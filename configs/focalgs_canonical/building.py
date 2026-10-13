from configs.focalgs_canonical._shared import build_model_params, clone_common


ModelHiddenParams, OptimizationParams, PipelineParams = clone_common()
ModelParams = build_model_params(
    voxel_size=0.005,
    use_deblur=True,
)
