import pytest, os
from DeepClassifier.entity.config_entity import PrepareCallbacksConfig
from DeepClassifier.components.prepare_callback import PrepareCallback
from pathlib import Path
from DeepClassifier.constants import *
from DeepClassifier.utils import read_yaml

class Test_Prepare_Callback_Tensorboard_Log_Dir:

    def test_get_tb_log_dir(self):
        config_fp = read_yaml(CONFIG_FILE_PATH)
        prepare_callback_config = PrepareCallbacksConfig(
            root_dir=config_fp.prepare_callbacks.root_dir,
            tensorboard_root_log_dir=config_fp.prepare_callbacks.tensorboard_root_log_dir,
            checkpoint_model_filepath=config_fp.prepare_callbacks.checkpoint_model_filepath)

        prepare_callbacks=PrepareCallback(config=prepare_callback_config)
        prepare_callbacks.get_tb_ckpt_callbacks()

        assert os.path.exists(prepare_callback_config.tensorboard_root_log_dir)

class Test_Prepare_Callback_CheckpointDir:

    def test_get_tb_ckpt_callbacks(self):
        config_fp = read_yaml(CONFIG_FILE_PATH)
        prepare_callback_config = PrepareCallbacksConfig(
            root_dir=config_fp.prepare_callbacks.root_dir,
            tensorboard_root_log_dir=config_fp.prepare_callbacks.tensorboard_root_log_dir,
            checkpoint_model_filepath=config_fp.prepare_callbacks.checkpoint_model_filepath)

        prepare_callbacks=PrepareCallback(config=prepare_callback_config)
        prepare_callbacks.get_tb_ckpt_callbacks()

        checkpoint_path = prepare_callback_config.checkpoint_model_filepath
        head,tail = os.path.split(checkpoint_path)
        assert os.path.exists(head)