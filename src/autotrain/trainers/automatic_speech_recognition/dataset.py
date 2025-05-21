import torch
from datasets import Dataset
from transformers import Wav2Vec2Processor

from autotrain.trainers.automatic_speech_recognition.params import AutomaticSpeechRecognitionParams


class AutomaticSpeechRecognitionDataset:
    def __init__(self, data: Dataset, processor: Wav2Vec2Processor, config: AutomaticSpeechRecognitionParams):
        self.data = data
        self.processor = processor
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio = item[self.config.audio_column]
        text = item[self.config.text_column]

        # Process audio
        if isinstance(audio, str):
            # Load audio file
            import librosa
            audio_array, _ = librosa.load(audio, sr=self.config.sampling_rate)
        else:
            audio_array = audio

        # Process text
        if isinstance(text, str):
            text = text.lower()

        # Prepare inputs
        inputs = self.processor(
            audio_array,
            sampling_rate=self.config.sampling_rate,
            return_tensors="pt",
            padding=True,
            max_length=int(self.config.max_duration * self.config.sampling_rate),
            truncation=True,
        )

        # Prepare labels
        with self.processor.as_target_processor():
            labels = self.processor(text).input_ids

        return {
            "input_values": inputs.input_values.squeeze(),
            "labels": torch.tensor(labels),
        } 