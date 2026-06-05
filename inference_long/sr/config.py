from dataclasses import dataclass, field, asdict
import os
import json
import types
import importlib.util
import inspect
from pathlib import Path
from typing import Type, List, Callable
import os


def fliter_fps_func(json_data, is_image, is_video, item):
    if is_image:
        return False
    fps = json_data.get("fps", 24)
    if fps < 20 or fps > 240:
        return True
    return False

FILTER_FPS = fliter_fps_func


def print_rank0(message: str) -> None:
    if int(os.getenv("RANK", "0")) == 0:
        print(message)


def func_to_struct(f):
    if isinstance(f, (types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
        src = inspect.getsource(f)
        return {
            "__type__": "function",
            "module": getattr(f, "__module__", None),
            "qualname": getattr(f, "__qualname__", getattr(f, "__name__", str(f))),
            "name": getattr(f, "__name__", None),
            "file": inspect.getsourcefile(f),
            "source": src,
        }

    return {
        "__type__": "callable",
        "repr": repr(f),
        "class_module": f.__class__.__module__,
        "class_qualname": f.__class__.__qualname__,
    }


def make_json_safe(x):
    if callable(x):
        return func_to_struct(x)
    if isinstance(x, (list, tuple)):
        return [make_json_safe(v) for v in x]
    if isinstance(x, dict):
        return {k: make_json_safe(v) for k, v in x.items()}
    return x

@dataclass
class ExpConfig:
    seed: int = 42
    val_seed: int = 42
    exp_name: str = "test"
    output_dir: str = "./output"

    # Resume
    resume_from_checkpoint: str = None
    resume_optimizer: bool = True
    resume_dataloader: bool = True
    resume_random_state: bool = True
    auto_resume: bool = False

    # DIT
    dit_ckpt: str = None
    dit_ckpt_type: str = "pt"  # "safetensor" or "pt"
    dit_arch_config: str = None
    dit_precision: str = "bf16"
    is_repa: bool = False
    repa_layer: int = 20
    repa_lambda: float = 0.5
    repa_aligh: str = 'patch'

    # VAE
    vae_ckpt: str = None
    vae_precision: str = "fp16"

    # Text Encoder
    text_encoder_arch_config: str = None
    text_encoder_precision: str = "bf16"
    text_token_max_length: int = 512

    # Data
    train_image_data_files: str = None
    train_video_data_files: str = None
    train_image_caption_keys: list[str] = None
    train_image_caption_sampling_prob: list[float] = None
    train_video_caption_keys: list[str] = None
    train_video_caption_sampling_prob: list[float] = None
    video_sampling_prob: float = 1
    bucket_configs: list[tuple[int, int, int, int]] = None
    prioritize_frame_matching: bool = True
    ensure_divisible_shards: bool = True
    shuffle: bool = True
    num_workers: int = 2
    fps: int = -1
    buffer_size: int = 1000
    filter_lambdas: List[Callable] = field(default_factory=lambda: None)

    # Training
    weighting_scheme: str = "lognorm"
    train_image_flow_shift: int = 1.0
    train_video_flow_shift: int = 1.0
    cfg_rate: float = 0.1

    # Multi-task
    enable_multi_task_training: bool = False
    condition_inject_mode: str = "token_replace"
    multi_task_types: list[str] = field(
        default_factory=lambda: ["T2V", "I2V"])
    multi_task_sampling_prob: list[float] = field(
        default_factory=lambda: [0.8, 0.2])

    micro_batch_size: int = 1
    max_train_steps: int = 10000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    data_check_interval: int = 100
    validation_steps: list[int] = field(default_factory=lambda: [1, 10, 100])
    validation_interval: int = 100
    checkpoint_interval: int = 1000
    grad_check_interval: int = 1000
    gc_interval: int = 1000
    log_interval: int = 10
    log_every_nth_rank: int = 1

    optimizer_name: str = 'adamw'
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_weight_decay: float = 0
    adam_epsilon: float = 1e-10
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 100

    enable_activation_checkpointing: bool = False
    activation_checkpointing_type: str = "full"  # "full" or "block_skip"
    # for "block_skip", checkpoint every N-th
    activation_checkpointing_skip_interval: int = 2

    # Validation
    val_data_files: str = None
    val_noise_scheduler: str = "flow_euler_discrete"
    val_batch_size: int = 1
    val_width: int = 256
    val_height: int = 256
    val_num_frames: int = 33
    val_num_inference_steps: int = 30
    val_guidance_scale: float = 7.5
    val_max_samples: int = None

    # Parallelism
    sp_size: int = 1

    # FSDP2
    training_mode: bool = True
    hsdp_shard_dim: int = 1
    reshard_after_forward: bool = False  # zero2=False, zero3=True
    use_fsdp_inference: bool = False
    cpu_offload: bool = False
    pin_cpu_memory: bool = False
    enable_torch_compile: bool = False

    # FA3
    enable_flash_attention_3: bool = False

    def __post_init__(self):
        self._validate()

        if self.enable_flash_attention_3:
            os.environ["ENABLE_FLASH_ATTENTION_3"] = "1"

    def _validate(self):
        if self.resume_from_checkpoint and self.dit_ckpt:
            raise ValueError(
                "Cannot specify both 'resume_from_checkpoint' and 'dit_ckpt'. Choose one.")

    def to_json_string(self) -> str:
        d = asdict(self)
        d = make_json_safe(d)
        return json.dumps(d, indent=2)

def load_config_class_from_pyfile(file_path: str) -> Type[ExpConfig]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for '{file_path}'.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for _, obj in inspect.getmembers(module, inspect.isclass):
        # The condition ensures we find a subclass, not ExpConfig itself.
        if issubclass(obj, ExpConfig) and obj is not ExpConfig:
            print_rank0(
                f"Dynamically loaded config class: '{obj.__name__}' from '{file_path}'")
            return obj

    raise ValueError(
        f"No class inheriting from 'ExpConfig' was found in '{file_path}'.")
