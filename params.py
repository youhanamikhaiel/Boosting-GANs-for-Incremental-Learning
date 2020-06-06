''' Parameters
   General architecture used for the GAN models '''

params = {'dataset': 'C10',
          'augment': False,
          'num_workers': 8,
          'pin_memory': True,
          'shuffle': True,
          'load_in_mem': False,
          'use_multiepoch_sampler': False,
          'G_param': 'SN',
          'D_param': 'SN',
          'G_ch': 64,
          'D_ch': 64,
          'G_depth': 1,
          'D_depth': 1,
          'D_wide': True,
          'G_shared': False,
          'shared_dim': 0,
          'dim_z': 128,
          'z_var': 1.0,
          'hier': False,
          'cross_replica': False,
          'mybn': False,
          'G_nl': 'relu',
          'D_nl': 'relu',
          'G_attn': '0',
          'D_attn': '0',
          'norm_style': 'bn',
          'seed': 0,
          'G_init': 'N02',
          'D_init': 'N02',
          'skip_init': False,
          'G_lr': 0.0002,
          'D_lr': 0.0002,
          'G_B1': 0.0,
          'D_B1': 0.0,
          'G_B2': 0.999,
          'D_B2': 0.999,
          'batch_size': 50,
          'G_batch_size': 0,
          'num_G_accumulations': 1,
          'num_D_steps': 4,
          'num_D_accumulations': 1,
          'split_D': False,
          'num_epochs': 1000,
          'parallel': True,
          'G_fp16': False,
          'D_fp16': False,
          'D_mixed_precision': False,
          'G_mixed_precision': False,
          'accumulate_stats': False,
          'num_standing_accumulations': 16,
          'G_eval_mode': False,
          'save_every': 500,
          'num_save_copies': 2,
          'num_best_copies': 5,
          'which_best': 'IS',
          'no_fid': False,
          'test_every': 500,
          'num_inception_images': 50000,
          'hashname': False,
          'base_root': '',
          'data_root': 'data',
          'weights_root': 'weights',
          'logs_root': 'logs',
          'samples_root': 'samples',
          'pbar': 'mine',
          'name_suffix': '',
          'experiment_name': '',
          'config_from_name': False,
          'ema': True,
          'ema_decay': 0.9999,
          'use_ema': True,
          'ema_start': 1000,
          'adam_eps': 1e-08,
          'BN_eps': 1e-05,
          'SN_eps': 1e-08,
          'num_G_SVs': 1,
          'num_D_SVs': 1,
          'num_G_SV_itrs': 1,
          'num_D_SV_itrs': 1,
          'G_ortho': 0.0,
          'D_ortho': 0.0,
          'toggle_grads': True,
          'which_train_fn': 'GAN',
          'load_weights': '',
          'resume': False,
          'logstyle': '%3.3e',
          'log_G_spectra': False,
          'log_D_spectra': False,
          'sv_log_interval': 10,
          'sample_npz': True,
          'sample_num_npz': 1,
          'sample_sheets': False,
          'sample_interps': False,
          'sample_sheet_folder_num': -1,
          'sample_random': False,
          'sample_trunc_curves': '0.05_0.05_1.0',
          'sample_inception_metrics': False}
