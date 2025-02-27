from transformers import Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser
from dataclasses import dataclass
from PIL import Image
import re
import os


@dataclass
class ScriptArguments:
    dataset_id_or_path: str = "./data/sample"
    

old_forward = Qwen2_5_VLForConditionalGeneration.forward
def custom_forward(self, *args, **kwargs):
    if "logits_to_keep" in kwargs:
        kwargs.pop("logits_to_keep")
    return old_forward(self, *args, **kwargs)
Qwen2_5_VLForConditionalGeneration.forward = custom_forward  


def format_reward(completions, **kwargs):
    # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern = r"\s*<think>.*?</think>\s*<answer>.*?</answer>\s*"
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completions]
    return [1.0 if match else 0.0 for match in matches]


def result_reward(completions, labels, **kwargs):
    result, pattern = [], r"\s*<think>.*?</think>\s*<answer>(.*?)</answer>\s*"
    for completion, label in zip(completions, labels):
        matched = re.search(pattern, completion, re.DOTALL)
        if not matched:
            result.append(0.0)
            continue
        
        pred = matched.group(1).strip()
        result.append(1.0 if pred == label else 0.0)
        
        os.makedirs("completion_samples", exist_ok=True)
        log_file = os.path.join("completion_samples", "completion_samples_vlm.txt")
        with open(log_file, "a") as f:
            f.write(f"\n\n====================================[label={label}] [pred={pred}]================================================\n")
            f.write(completion)
        
    return result


def model_factory(model_id, **kwargs):
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)


def ref_model_factory(model_id, **kwargs):
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)


def processing_class_factory(model_id, min_pixels=3136, max_pixels=12845056):
    processing_class = AutoProcessor.from_pretrained(model_id)
    processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
    processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
    processing_class.image_processor.max_pixels = max_pixels
    processing_class.image_processor.min_pixels = min_pixels
    return processing_class


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def main(model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig):
    print(f"Model parameters {model_args}")
    print(f"Training/evaluation parameters {training_args}")

    # 1. load data
    train_dataset = load_dataset("imagefolder", data_dir=script_args.dataset_id_or_path, split="train")
    train_dataset = train_dataset.shuffle(seed=training_args.seed)
    
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    def generate_prompt(image: Image.Image, label):
        prompt = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "你是一个优秀的图像鉴赏专家。你首先在脑海中进行分析和推理，然后为用户提供答案。"},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": "请从多个角度分析这张图是否是一个优秀的图片，你需要一步一步的进行推理和分析，最终给出答案，你的答案只能是“是”或者“否”。在<think></think>标签内展示你的分析推理过程。并在<answer></answer>标签中返回最终的答案，例如<answer>是</answer>。在<think>标签内逐步的进行分析和推理。"},
                ],
            }
        ]
        
        image = image.resize((image.width // 2, image.height // 2))
        
        return {"prompt": processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) , "image": image, "labels": "是" if str(label) == "1" else "否"}

    train_dataset = train_dataset.map(lambda x: generate_prompt(x["image"], x["label"]))
        
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[format_reward, result_reward],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=get_peft_config(model_args),
        factory={"model": model_factory, "ref_model": ref_model_factory, "processing_class": processing_class_factory},
    )

    train_result = trainer.train()

    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("*** Training complete ***")

    print("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    processor.save_pretrained(training_args.output_dir)
    print(f"Tokenizer saved to {training_args.output_dir}")

    
if __name__ == "__main__":
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    main(model_args, script_args, training_args)
