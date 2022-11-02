import pytest
from DeepClassifier.entity import DataIngestionConfig
from DeepClassifier.components import DataIngestion
from DeepClassifier.constants import *
from DeepClassifier.utils import read_yaml, create_directories
from pathlib import Path
import os

class Test_DataIngestion_download:
    data_ingestion_config = DataIngestionConfig(
        root_dir="tests/data/", 
        source_URL="https://raw.githubusercontent.com/d1b2/deep_CNN_Classifier/master/sample_data_for_tests/sample_data.zip", 
        local_data_file="tests/data/data.zip", 
        unzip_dir="tests/data/")

    def test_download(self):
        data_ingestion = DataIngestion(config=self.data_ingestion_config)
        data_ingestion.download_file()
        assert os.path.exists(self.data_ingestion_config.local_data_file)


class Test_DataIngestion_unzip:
    data_ingestion_config = DataIngestionConfig(
        root_dir="tests/data/", 
        source_URL="", 
        local_data_file="tests/data/data.zip", 
        unzip_dir="tests/data/")

    def test_unzip(self):
        data_ingestion = DataIngestion(config=self.data_ingestion_config)
        data_ingestion.unzip_and_clean()
        assert os.path.isdir(Path("tests/data/PetImages"))
        assert os.path.isdir(Path("tests/data/PetImages/Cat"))
        assert os.path.isdir(Path("tests/data/PetImages/Dog"))