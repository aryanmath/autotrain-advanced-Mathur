import os
from typing import Dict, List, Optional

import librosa
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import Wav2Vec2Processor

from autotrain import logger
from autotrain.preprocessor import AutoTrainPreprocessor


class AutomaticSpeechRecognitionPreprocessor(AutoTrainPreprocessor):
    def __init__(
        self,
        train_data: str,
        token: str,
        project_name: str,
        username: str,
        column_mapping: Dict[str, str],
        valid_data: Optional[str] = None,
        test_data: Optional[str] = None,
        text_column: str = "transcription",
        audio_column: str = "audio",
        max_duration: float = 30.0,
        sampling_rate: int = 16000,
    ):
        super().__init__(
            train_data=train_data,
            token=token,
            project_name=project_name,
            username=username,
            column_mapping=column_mapping,
            valid_data=valid_data,
            test_data=test_data,
        )
        self.text_column = text_column
        self.audio_column = audio_column
        self.max_duration = max_duration
        self.sampling_rate = sampling_rate
        self.processor = None

    def prepare(self):
        """Prepare the data for ASR training."""
        logger.info("Preparing data for ASR training...")
        
        # Load and process training data
        train_df = self._load_data(self.train_data)
        train_df = self._process_data(train_df)
        
        # Load and process validation data if provided
        valid_df = None
        if self.valid_data:
            valid_df = self._load_data(self.valid_data)
            valid_df = self._process_data(valid_df)
        
        # Load and process test data if provided
        test_df = None
        if self.test_data:
            test_df = self._load_data(self.test_data)
            test_df = self._process_data(test_df)
        
        # Create datasets
        train_dataset = Dataset.from_pandas(train_df)
        valid_dataset = Dataset.from_pandas(valid_df) if valid_df is not None else None
        test_dataset = Dataset.from_pandas(test_df) if test_df is not None else None
        
        # Save datasets
        self._save_datasets(train_dataset, valid_dataset, test_dataset)
        
        logger.info("Data preparation completed!")

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from various formats."""
        if data_path.endswith(".csv"):
            return pd.read_csv(data_path)
        elif data_path.endswith(".json"):
            return pd.read_json(data_path)
        elif data_path.endswith(".jsonl"):
            return pd.read_json(data_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the data for ASR training."""
        # Ensure required columns exist
        required_columns = [self.text_column, self.audio_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Process audio files
        df[self.audio_column] = df[self.audio_column].apply(self._process_audio)
        
        # Clean text
        df[self.text_column] = df[self.text_column].apply(self._clean_text)
        
        return df

    def _process_audio(self, audio_path: str) -> np.ndarray:
        """Process audio file to numpy array."""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
            # Trim silence
            audio = librosa.effects.trim(audio)[0]
            # Normalize
            audio = librosa.util.normalize(audio)
            return audio
        except Exception as e:
            logger.error(f"Error processing audio file {audio_path}: {e}")
            return np.array([])

    def _clean_text(self, text: str) -> str:
        """Clean text for ASR training."""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = " ".join(text.split())
        return text

    def _save_datasets(self, train_dataset: Dataset, valid_dataset: Optional[Dataset], test_dataset: Optional[Dataset]):
        """Save processed datasets."""
        output_dir = f"{self.project_name}/autotrain-data"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save datasets
        train_dataset.save_to_disk(f"{output_dir}/train")
        if valid_dataset:
            valid_dataset.save_to_disk(f"{output_dir}/validation")
        if test_dataset:
            test_dataset.save_to_disk(f"{output_dir}/test") 