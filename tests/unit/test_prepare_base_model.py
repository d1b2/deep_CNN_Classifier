import pytest, os
from DeepClassifier.components import prepare_base_model
from DeepClassifier.entity.config_entity import PrepareBaseModelConfig
from DeepClassifier.components.prepare_base_model import PrepareBaseModel
from pathlib import Path
from DeepClassifier.constants import *
from DeepClassifier.utils import read_yaml, create_directories

class Test_Prepare_BaseModel:
    def test_base_model(self):
        config_fp = read_yaml(CONFIG_FILE_PATH)
        params_fp = read_yaml(PARAMS_FILE_PATH)

        prepare_basemodel_config = PrepareBaseModelConfig(
            root_dir=config_fp.prepare_base_model.root_dir,
            base_model_path=config_fp.prepare_base_model.base_model_path,
            updated_base_model_path=config_fp.prepare_base_model.updated_base_model_path,
            params_learning_rate=params_fp.LEARNING_RATE,
            params_include_top=params_fp.INCLUDE_TOP,
            params_weights=params_fp.WEIGHTS,
            params_image_size=params_fp.IMAGE_SIZE,
            params_classes=params_fp.CLASSES
        )

        base_model=PrepareBaseModel(config=prepare_basemodel_config)
        base_model.get_base_model()

        assert os.path.exists(prepare_basemodel_config.base_model_path)

class Test_Updated_Base_Model:
    def test_get_updated_base_model(self):
        config_fp = read_yaml(CONFIG_FILE_PATH)
        params_fp = read_yaml(PARAMS_FILE_PATH)

        prepare_basemodel_config = PrepareBaseModelConfig(
            root_dir=config_fp.prepare_base_model.root_dir,
            base_model_path=config_fp.prepare_base_model.base_model_path,
            updated_base_model_path=config_fp.prepare_base_model.updated_base_model_path,
            params_learning_rate=params_fp.LEARNING_RATE,
            params_include_top=params_fp.INCLUDE_TOP,
            params_weights=params_fp.WEIGHTS,
            params_image_size=params_fp.IMAGE_SIZE,
            params_classes=params_fp.CLASSES
        )

        base_model_updated=PrepareBaseModel(config=prepare_basemodel_config)
        base_model_updated.get_base_model()
        base_model_updated.update_base_model()

        assert os.path.exists(prepare_basemodel_config.updated_base_model_path)