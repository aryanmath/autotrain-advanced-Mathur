import argparse
import json
import os
from typing import Optional

from autotrain import logger
from autotrain.cli.run import AutoTrainCLI
from autotrain.trainers.automatic_speech_recognition import __main__ as trainer


class RunAutoTrainAutomaticSpeechRecognitionCommand(AutoTrainCLI):
    @staticmethod
    def register_subcommand(parser: argparse._SubParsersAction):
        run_parser = parser.add_parser(
            "automatic-speech-recognition",
            epilog="For more information about a command, run: `autotrain automatic-speech-recognition <command> --help`",
        )
        run_parser.set_defaults(func=RunAutoTrainAutomaticSpeechRecognitionCommand)
        run_parser.add_argument(
            "--train_data",
            type=str,
            required=True,
            help="Path to training data",
        )
        run_parser.add_argument(
            "--valid_data",
            type=str,
            required=False,
            help="Path to validation data",
        )
        run_parser.add_argument(
            "--project_name",
            type=str,
            required=True,
            help="Name of the project",
        )
        run_parser.add_argument(
            "--model",
            type=str,
            required=True,
            help="Model to use",
        )
        run_parser.add_argument(
            "--text_column",
            type=str,
            default="transcription",
            help="Name of the text column",
        )
        run_parser.add_argument(
            "--audio_column",
            type=str,
            default="audio",
            help="Name of the audio column",
        )
        run_parser.add_argument(
            "--max_duration",
            type=float,
            default=30.0,
            help="Maximum duration of audio in seconds",
        )
        run_parser.add_argument(
            "--sampling_rate",
            type=int,
            default=16000,
            help="Sampling rate for audio",
        )
        run_parser.add_argument(
            "--batch_size",
            type=int,
            default=8,
            help="Batch size",
        )
        run_parser.add_argument(
            "--epochs",
            type=int,
            default=3,
            help="Number of epochs",
        )
        run_parser.add_argument(
            "--learning_rate",
            type=float,
            default=3e-4,
            help="Learning rate",
        )
        run_parser.add_argument(
            "--optimizer",
            type=str,
            default="adamw",
            help="Optimizer to use",
        )
        run_parser.add_argument(
            "--scheduler",
            type=str,
            default="linear",
            help="Scheduler to use",
        )
        run_parser.add_argument(
            "--mixed_precision",
            type=str,
            default="fp16",
            help="Mixed precision to use",
        )
        run_parser.add_argument(
            "--weight_decay",
            type=float,
            default=0.01,
            help="Weight decay",
        )
        run_parser.add_argument(
            "--warmup_ratio",
            type=float,
            default=0.1,
            help="Warmup ratio",
        )
        run_parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=3,
            help="Early stopping patience",
        )
        run_parser.add_argument(
            "--early_stopping_threshold",
            type=float,
            default=0.01,
            help="Early stopping threshold",
        )
        run_parser.add_argument(
            "--eval_strategy",
            type=str,
            default="epoch",
            help="Evaluation strategy",
        )
        run_parser.add_argument(
            "--save_total_limit",
            type=int,
            default=1,
            help="Total number of checkpoints to save",
        )
        run_parser.add_argument(
            "--auto_find_batch_size",
            action="store_true",
            help="Automatically find batch size",
        )
        run_parser.add_argument(
            "--logging_steps",
            type=int,
            default=-1,
            help="Logging steps",
        )

    def run(self):
        """Run the ASR training."""
        logger.info("Starting ASR training...")
        
        # Create training config
        training_config = {
            "task": "automatic-speech-recognition",
            "model": self.args.model,
            "train_data": self.args.train_data,
            "valid_data": self.args.valid_data,
            "project_name": self.args.project_name,
            "text_column": self.args.text_column,
            "audio_column": self.args.audio_column,
            "max_duration": self.args.max_duration,
            "sampling_rate": self.args.sampling_rate,
            "batch_size": self.args.batch_size,
            "epochs": self.args.epochs,
            "learning_rate": self.args.learning_rate,
            "optimizer": self.args.optimizer,
            "scheduler": self.args.scheduler,
            "mixed_precision": self.args.mixed_precision,
            "weight_decay": self.args.weight_decay,
            "warmup_ratio": self.args.warmup_ratio,
            "early_stopping_patience": self.args.early_stopping_patience,
            "early_stopping_threshold": self.args.early_stopping_threshold,
            "eval_strategy": self.args.eval_strategy,
            "save_total_limit": self.args.save_total_limit,
            "auto_find_batch_size": self.args.auto_find_batch_size,
            "logging_steps": self.args.logging_steps,
        }
        
        # Save config
        config_path = f"{self.args.project_name}/training_config.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(training_config, f, indent=2)
        
        # Run training
        trainer.train(training_config) 