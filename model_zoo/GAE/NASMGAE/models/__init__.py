from .edcoder import PreModel


def build_model(param):
    num_heads       = param.num_heads
    num_out_heads   = param.num_out_heads
    num_hidden      = param.num_hidden
    num_layers      = param.num_layers
    residual        = param.residual
    attn_drop       = param.attn_drop
    in_drop         = param.in_drop
    norm            = param.norm
    negative_slope  = param.negative_slope
    encoder_type    = param.encoder
    decoder_type    = param.decoder
    mask_rate       = param.mask_rate
    drop_edge_rate  = param.drop_edge_rate
    replace_rate    = param.replace_rate


    activation      = param.activation
    loss_fn         = param.loss_fn
    alpha_l         = param.alpha_l
    concat_hidden   = param.concat_hidden
    num_features    = param.num_features


    model = PreModel(
        in_dim=num_features,
        num_hidden=num_hidden,
        num_layers=num_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        mask_rate=mask_rate,
        norm=norm,
        loss_fn=loss_fn,
        drop_edge_rate=drop_edge_rate,
        replace_rate=replace_rate,
        alpha_l=alpha_l,
        concat_hidden=concat_hidden,
    )
    return model