# import argparse
# import importlib

# def execute_function(method, mode):
#     if method == 'vae':
#         mode = 'train'
#     mode = 'main' if mode == 'train' else 'sample'

#     if method == 'vae':
#         module_name = f"tabsyn.vae.main"
#     elif method == 'tabsyn':
#         module_name = f"tabsyn.{mode}"
#     elif method == 'tabddpm':
#         module_name = f"baselines.tabddpm.main_train" if mode == 'main' else f"baselines.tabddpm.main_sample"
#     else:
#         module_name = f"baselines.{method}.{mode}"

#     try:
#         train_module = importlib.import_module(module_name)
#         train_function = getattr(train_module, 'main')
#     except ModuleNotFoundError:
#         print(f"Module {module_name} not found.")
#         exit(1)
#     except AttributeError:
#         print(f"Function 'main' not found in module {module_name}.")
#         exit(1)
#     return train_function

# def get_args():
#     parser = argparse.ArgumentParser(description='Pipeline')

#     # General configs
#     parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
#     parser.add_argument('--mode', type=str, default='train', help='Mode: train or sample.')
#     parser.add_argument('--method', type=str, default='tabsyn', help='Method: tabsyn or baseline.')
#     parser.add_argument('--gpu', type=int, default=0, help='GPU index.')


#     ''' configs for CTGAN '''

#     parser.add_argument('-e', '--epochs', default=1000, type=int,
#                         help='Number of training epochs')
#     parser.add_argument('--no-header', dest='header', action='store_false',
#                         help='The CSV file has no header. Discrete columns will be indices.')

#     parser.add_argument('-m', '--metadata', help='Path to the metadata')
#     parser.add_argument('-d', '--discrete',
#                         help='Comma separated list of discrete columns without whitespaces.')
#     parser.add_argument('-n', '--num-samples', type=int,
#                         help='Number of rows to sample. Defaults to the training data size')

#     parser.add_argument('--generator_lr', type=float, default=2e-4,
#                         help='Learning rate for the generator.')
#     parser.add_argument('--discriminator_lr', type=float, default=2e-4,
#                         help='Learning rate for the discriminator.')

#     parser.add_argument('--generator_decay', type=float, default=1e-6,
#                         help='Weight decay for the generator.')
#     parser.add_argument('--discriminator_decay', type=float, default=0,
#                         help='Weight decay for the discriminator.')

#     parser.add_argument('--embedding_dim', type=int, default=1024,
#                         help='Dimension of input z to the generator.')
#     parser.add_argument('--generator_dim', type=str, default='1024,2048,2048,1024',
#                         help='Dimension of each generator layer. '
#                         'Comma separated integers with no whitespaces.')
#     parser.add_argument('--discriminator_dim', type=str, default='1024,2048,2048,1024',
#                         help='Dimension of each discriminator layer. '
#                         'Comma separated integers with no whitespaces.')

#     parser.add_argument('--batch_size', type=int, default=500,
#                         help='Batch size. Must be an even number.')
#     parser.add_argument('--save', default=None, type=str,
#                         help='A filename to save the trained synthesizer.')
#     parser.add_argument('--load', default=None, type=str,
#                         help='A filename to load a trained synthesizer.')

#     parser.add_argument('--sample_condition_column', default=None, type=str,
#                         help='Select a discrete column name.')
#     parser.add_argument('--sample_condition_column_value', default=None, type=str,
#                         help='Specify the value of the selected discrete column.')

#     ''' configs for GReaT '''

#     parser.add_argument('--bs', type=int, default=16, help='(Maximum) batch size')

#     ''' configs for CoDi '''

#     # General Options
#     parser.add_argument('--logdir', type=str, default='./codi_exp', help='log directory')
#     parser.add_argument('--train', action='store_true', help='train from scratch')
#     parser.add_argument('--eval', action='store_true', help='load ckpt.pt and evaluate')

#     # Network Architecture
#     parser.add_argument('--encoder_dim', nargs='+', type=int, help='encoder_dim')
#     parser.add_argument('--encoder_dim_con', type=str, default="512,1024,1024,512", help='encoder_dim_con')
#     parser.add_argument('--encoder_dim_dis', type=str, default="512,1024,1024,512", help='encoder_dim_dis')
#     parser.add_argument('--nf', type=int, help='nf')
#     parser.add_argument('--nf_con', type=int, default=16, help='nf_con')
#     parser.add_argument('--nf_dis', type=int, default=64, help='nf_dis')
#     parser.add_argument('--input_size', type=int, help='input_size')
#     parser.add_argument('--cond_size', type=int, help='cond_size')
#     parser.add_argument('--output_size', type=int, help='output_size')
#     parser.add_argument('--activation', type=str, default='relu', help='activation')

#     # Training
#     parser.add_argument('--training_batch_size', type=int, default=4096, help='batch size')
#     parser.add_argument('--eval_batch_size', type=int, default=2100, help='batch size')
#     parser.add_argument('--T', type=int, default=50, help='total diffusion steps')
#     parser.add_argument('--beta_1', type=float, default=0.00001, help='start beta value')
#     parser.add_argument('--beta_T', type=float, default=0.02, help='end beta value')
#     parser.add_argument('--lr_con', type=float, default=2e-03, help='target learning rate')
#     parser.add_argument('--lr_dis', type=float, default=2e-03, help='target learning rate')
#     parser.add_argument('--total_epochs_both', type=int, default=20000, help='total training steps')
#     parser.add_argument('--grad_clip', type=float, default=1., help="gradient norm clipping")
#     parser.add_argument('--parallel', action='store_true', help='multi gpu training')

#     # Sampling
#     parser.add_argument('--sample_step', type=int, default=2000, help='frequency of sampling')

#     # Continuous diffusion model
#     parser.add_argument('--mean_type', type=str, default='epsilon', choices=['xprev', 'xstart', 'epsilon'], help='predict variable')
#     parser.add_argument('--var_type', type=str, default='fixedsmall', choices=['fixedlarge', 'fixedsmall'], help='variance type')

#     # Contrastive Learning
#     parser.add_argument('--ns_method', type=int, default=0, help='negative condition method')
#     parser.add_argument('--lambda_con', type=float, default=0.2, help='lambda_con')
#     parser.add_argument('--lambda_dis', type=float, default=0.2, help='lambda_dis')
#     ################    


#     # configs for TabDDPM
#     parser.add_argument('--ddim', action = 'store_true', default=False, help='Whether use DDIM sampler')

#     # configs for SMOTE
#     parser.add_argument('--cat_encoding', type=str, default='one-hot', help='Encoding method for categorical features')


#     # configs for traing TabSyn's VAE
#     parser.add_argument('--max_beta', type=float, default=1e-2, help='Maximum beta')
#     parser.add_argument('--min_beta', type=float, default=1e-5, help='Minimum beta.')
#     parser.add_argument('--lambd', type=float, default=0.7, help='Batch size.')
    
#     # Experiment 2 (Exp 2)
#     parser.add_argument('--denoiser_type', type=str, default='token_hybrid', choices=['mlp', 'token_hybrid'], help='Diffusion denoiser architecture.')
#     parser.add_argument('--dim_t', type=int, default=1024, help='Time embedding width.')
#     parser.add_argument('--hybrid_model_dim', type=int, default=128, help='Hidden token width for the hybrid denoiser.')
#     parser.add_argument('--hybrid_num_heads', type=int, default=4, help='Number of attention heads in the hybrid denoiser.')
#     parser.add_argument('--hybrid_mlp_hidden_factor', type=float, default=4.0, help='Expansion factor for the token-wise MLP.')
#     parser.add_argument('--hybrid_dropout', type=float, default=0.0, help='Dropout in the hybrid denoiser.')


#     # configs for sampling
#     parser.add_argument('--save_path', type=str, default=None, help='Path to save synthetic data.')
#     parser.add_argument('--steps', type=int, default=50, help='NFEs.')
    
#     args = parser.parse_args()

#     return args



import argparse
import importlib

def execute_function(method, mode):
    if method == 'vae':
        mode = 'train'
    mode = 'main' if mode == 'train' else 'sample'

    if method == 'vae':
        module_name = f"tabsyn.vae.main"
    elif method == 'tabsyn':
        module_name = f"tabsyn.{mode}"
    elif method == 'tabddpm':
        module_name = f"baselines.tabddpm.main_train" if mode == 'main' else f"baselines.tabddpm.main_sample"
    else:
        module_name = f"baselines.{method}.{mode}"

    try:
        train_module = importlib.import_module(module_name)
        train_function = getattr(train_module, 'main')
    except ModuleNotFoundError:
        print(f"Module {module_name} not found.")
        exit(1)
    except AttributeError:
        print(f"Function 'main' not found in module {module_name}.")
        exit(1)
    return train_function

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    # General configs
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--mode', type=str, default='train', help='Mode: train or sample.')
    parser.add_argument('--method', type=str, default='tabsyn', help='Method: tabsyn or baseline.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')


    ''' configs for CTGAN '''

    parser.add_argument('-e', '--epochs', default=1000, type=int,
                        help='Number of training epochs')
    parser.add_argument('--no-header', dest='header', action='store_false',
                        help='The CSV file has no header. Discrete columns will be indices.')

    parser.add_argument('-m', '--metadata', help='Path to the metadata')
    parser.add_argument('-d', '--discrete',
                        help='Comma separated list of discrete columns without whitespaces.')
    parser.add_argument('-n', '--num-samples', type=int,
                        help='Number of rows to sample. Defaults to the training data size')

    parser.add_argument('--generator_lr', type=float, default=2e-4,
                        help='Learning rate for the generator.')
    parser.add_argument('--discriminator_lr', type=float, default=2e-4,
                        help='Learning rate for the discriminator.')

    parser.add_argument('--generator_decay', type=float, default=1e-6,
                        help='Weight decay for the generator.')
    parser.add_argument('--discriminator_decay', type=float, default=0,
                        help='Weight decay for the discriminator.')

    parser.add_argument('--embedding_dim', type=int, default=1024,
                        help='Dimension of input z to the generator.')
    parser.add_argument('--generator_dim', type=str, default='1024,2048,2048,1024',
                        help='Dimension of each generator layer. '
                        'Comma separated integers with no whitespaces.')
    parser.add_argument('--discriminator_dim', type=str, default='1024,2048,2048,1024',
                        help='Dimension of each discriminator layer. '
                        'Comma separated integers with no whitespaces.')

    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size. Must be an even number.')
    parser.add_argument('--save', default=None, type=str,
                        help='A filename to save the trained synthesizer.')
    parser.add_argument('--load', default=None, type=str,
                        help='A filename to load a trained synthesizer.')

    parser.add_argument('--sample_condition_column', default=None, type=str,
                        help='Select a discrete column name.')
    parser.add_argument('--sample_condition_column_value', default=None, type=str,
                        help='Specify the value of the selected discrete column.')

    ''' configs for GReaT '''

    parser.add_argument('--bs', type=int, default=16, help='(Maximum) batch size')

    ''' configs for CoDi '''

    # General Options
    parser.add_argument('--logdir', type=str, default='./codi_exp', help='log directory')
    parser.add_argument('--train', action='store_true', help='train from scratch')
    parser.add_argument('--eval', action='store_true', help='load ckpt.pt and evaluate')

    # Network Architecture
    parser.add_argument('--encoder_dim', nargs='+', type=int, help='encoder_dim')
    parser.add_argument('--encoder_dim_con', type=str, default="512,1024,1024,512", help='encoder_dim_con')
    parser.add_argument('--encoder_dim_dis', type=str, default="512,1024,1024,512", help='encoder_dim_dis')
    parser.add_argument('--nf', type=int, help='nf')
    parser.add_argument('--nf_con', type=int, default=16, help='nf_con')
    parser.add_argument('--nf_dis', type=int, default=64, help='nf_dis')
    parser.add_argument('--input_size', type=int, help='input_size')
    parser.add_argument('--cond_size', type=int, help='cond_size')
    parser.add_argument('--output_size', type=int, help='output_size')
    parser.add_argument('--activation', type=str, default='relu', help='activation')

    # Training
    parser.add_argument('--training_batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=2100, help='batch size')
    parser.add_argument('--T', type=int, default=50, help='total diffusion steps')
    parser.add_argument('--beta_1', type=float, default=0.00001, help='start beta value')
    parser.add_argument('--beta_T', type=float, default=0.02, help='end beta value')
    parser.add_argument('--lr_con', type=float, default=2e-03, help='target learning rate')
    parser.add_argument('--lr_dis', type=float, default=2e-03, help='target learning rate')
    parser.add_argument('--total_epochs_both', type=int, default=20000, help='total training steps')
    parser.add_argument('--grad_clip', type=float, default=1., help="gradient norm clipping")
    parser.add_argument('--parallel', action='store_true', help='multi gpu training')

    # Sampling
    parser.add_argument('--sample_step', type=int, default=2000, help='frequency of sampling')

    # Continuous diffusion model
    parser.add_argument('--mean_type', type=str, default='epsilon', choices=['xprev', 'xstart', 'epsilon'], help='predict variable')
    parser.add_argument('--var_type', type=str, default='fixedsmall', choices=['fixedlarge', 'fixedsmall'], help='variance type')

    # Contrastive Learning
    parser.add_argument('--ns_method', type=int, default=0, help='negative condition method')
    parser.add_argument('--lambda_con', type=float, default=0.2, help='lambda_con')
    parser.add_argument('--lambda_dis', type=float, default=0.2, help='lambda_dis')
    ################    


    # configs for TabDDPM
    parser.add_argument('--ddim', action = 'store_true', default=False, help='Whether use DDIM sampler')

    # configs for SMOTE
    parser.add_argument('--cat_encoding', type=str, default='one-hot', help='Encoding method for categorical features')


    # configs for traing TabSyn's VAE
    parser.add_argument('--max_beta', type=float, default=1e-2, help='Maximum beta')
    parser.add_argument('--min_beta', type=float, default=1e-5, help='Minimum beta.')
    parser.add_argument('--lambd', type=float, default=0.7, help='Batch size.')
    
    parser.add_argument('--cat_decoder_type', type=str, default='mlp', choices=['linear', 'mlp'],
                        help='Categorical decoder head. Use mlp to strengthen categorical reconstruction.')
    parser.add_argument('--cat_decoder_hidden_factor', type=int, default=4,
                        help='Hidden size multiplier for the categorical MLP head.')
    parser.add_argument('--cat_decoder_dropout', type=float, default=0.1,
                        help='Dropout inside the categorical MLP head.')


    # configs for sampling
    parser.add_argument('--save_path', type=str, default=None, help='Path to save synthetic data.')
    parser.add_argument('--steps', type=int, default=50, help='NFEs.')

    # blockwise latent noise for TabSyn
    parser.add_argument('--block_noise_mode', type=str, default='data', choices=['none', 'data', 'learned'], help='Blockwise latent noise scaling for TabSyn: none, data-driven fixed, or data-initialized learned.')
    parser.add_argument('--block_noise_min', type=float, default=0.5, help='Lower clamp for blockwise latent noise amplitudes.')
    parser.add_argument('--block_noise_max', type=float, default=1.5, help='Upper clamp for blockwise latent noise amplitudes.')
    
    args = parser.parse_args()

    return args