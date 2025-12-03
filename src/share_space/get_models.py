import torch.optim as optim
from share_space.models.sim2exp_model import Sim2ExpModel, StyleTransferModel
from share_space.train import TeacherStudentTrainer
from share_space.loss import Sim2ExpLoss
from share_space.evaluation import MetricsEvaluator
from share_space.diffusion import DiffusionModel
from share_space.train_style import StyleTransferTrainer
from share_space.models.decoders import ConvDecoder, TransformerDecoder
from share_space.loss import StyleTransferLoss
from share_space.adain_histo import HistoAdaIN


def get_stage_1_model(config, device, stage_name):
    # Model configuration
    encoder_config = {
        'img_size': config['model']['img_size'],
        'patch_size': config['model']['patch_size'],
        'sim_chans': config['model']['sim_chans'],
        'exp_chans': config['model']['exp_chans'],
        'embed_dim': config['model']['embed_dim'],
        'depth': config['model']['depth'],
        'num_heads': config['model']['num_heads'],
        'mlp_ratio': 4.0,
        'qkv_bias': True,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0
    }

    decoder_config = {
        'embed_dim': config['model']['embed_dim'],
        'img_size': config['model']['img_size'],
        'patch_size': config['model']['patch_size'],
        'sim_chans': config['model']['sim_chans'],
        'exp_chans': config['model']['exp_chans']
    }
    # Create model
    print("Creating model...")
    model = Sim2ExpModel(
        encoder_config=encoder_config,
        decoder_type=config['model']['decoder_type'],
        decoder_config=decoder_config
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Create trainer
    trainer = TeacherStudentTrainer(
        student_model=model,
        device=device,
        teacher_momentum=config['training'][stage_name]['teacher_momentum'],
        use_diffusion=config['training'][stage_name]['use_diffusion'],
        diffusion_model=DiffusionModel(
            unet_config=config['diffusion']['unet_config'],
            timesteps=config['diffusion']['timesteps'],
            beta_start=config['diffusion']['beta_start'],
            beta_end=config['diffusion']['beta_end']
        )
    )
    
    # Create loss function
    loss_fn = Sim2ExpLoss(
        use_diffusion=config['training'][stage_name]['use_diffusion'],
        recon_weight=config['loss']['recon_weight'],
        distill_weight=config['loss']['distill_weight'],
        mask_weight=config['loss']['mask_weight'],
        diffusion_weight=config['loss']['diffusion_weight']
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.decoder.parameters(), # if use_diffusion, also include diffusion model parameters
        lr=config['training'][stage_name]['learning_rate'],
        weight_decay=config['training'][stage_name]['weight_decay']
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training'][stage_name]['epochs'],
        eta_min=config['training'][stage_name]['learning_rate'] * 0.01
    )
    evaluator = MetricsEvaluator(device=device)
    return model, trainer, optimizer, scheduler, loss_fn, evaluator

def get_stage_2_model(config, device, stage_name, feature_extractor):
    decoder_config = {
        'embed_dim': config['model']['embed_dim'],
        'img_size': config['model']['img_size'],
        'patch_size': config['model']['patch_size'],
        'out_chans': config['model']['exp_chans']
    }
    loss_fn = StyleTransferLoss(
        use_diffusion=config['training'][stage_name]['use_diffusion'],
        content_weight=config['loss']['content_weight'],
        style_weight=config['loss']['style_weight'],
        element_weight=config['loss']['element_weight'],
        diffusion_weight=config['loss']['diffusion_weight'],
        image_weight=config['loss']['image_weight'],
        fused_patches_weight=config['loss']['fused_patches_weight']
    )
    style_model = StyleTransferModel(
        feature_extractor=feature_extractor,
        #decoder=feature_extractor.decoder,
        decoder=ConvDecoder(**decoder_config) if config['model']['decoder_type'] == 'conv' else TransformerDecoder(**decoder_config),
        adain=HistoAdaIN(embed_dim=config['model']['embed_dim'], mode='vanilla')
        #adain=FusionModule(embed_dim=config['model']['embed_dim'], mode='cross_attn')
        #adain=WeightedAdaIN(embed_dim=config['model']['embed_dim'])
        #adain=LearnableAdaIN(embed_dim=config['model']['embed_dim'])
        #adain=HistoAdaINSpatialAware(embed_dim=config['model']['embed_dim'], use_content_residual=config['training'][stage_name]['use_content_residual'])
        #adain=HistoAdaINHybrid(patch_size=config['model']['patch_size'])
    )
    if config['training'][stage_name]['use_diffusion']:
        diffusion_model = DiffusionModel(
            unet_config=config['diffusion']['unet_config'],
            timesteps=config['diffusion']['timesteps'],
            beta_start=config['diffusion']['beta_start'],
            beta_end=config['diffusion']['beta_end']
        )
    else:
        diffusion_model = None
    trainer_style = StyleTransferTrainer(
        model=style_model,
        device=device,
        use_diffusion=config['training'][stage_name]['use_diffusion'],
        diffusion_model=diffusion_model,
        use_all_pairs=config['training'][stage_name]['use_all_pairs'],
    )
    param_groups = [
        {
            'params': style_model.decoder.parameters(),
            'lr': config['training'][stage_name]['learning_rate'] * 0.01,
            'weight_decay': config['training'][stage_name]['weight_decay'],
        },
        {
            'params': style_model.adain.parameters(),
            'lr': config['training'][stage_name]['learning_rate'],
            'weight_decay': config['training'][stage_name]['weight_decay'],
        }
    ]
    if config['training'][stage_name]['use_diffusion'] and diffusion_model is not None:
        param_groups.append({
            'params': diffusion_model.parameters(),
            'lr': config['training'][stage_name]['learning_rate'],
            'weight_decay': config['training'][stage_name]['weight_decay'],
        })
    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training'][stage_name]['epochs'],
        eta_min=config['training'][stage_name]['learning_rate'] * 0.01
    )

    evaluator = MetricsEvaluator(device=device)
    return style_model, trainer_style, optimizer, scheduler, loss_fn, evaluator
