from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset
import pandas as pd

TestModel = "hf-internal-testing/tiny-random-LlamaForCausalLM"

class SupervisedFineTuner: 
    def __init__(self,data_path, instruction, device='cpu', model_path=TestModel):
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False
        )
        self.model_path = model_path
        self.data_path = data_path
        self.instruction = instruction
        self.encoder = None
        self.model = None
        self.device = device
    

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token


        model = AutoModelForCausalLM.from_pretrained(self.model_path)
        model.to(self.device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")
        self.model = get_peft_model(model, self.lora_config)
        self.model.print_trainable_parameters()

    def load_training_data(self): 
        df = pd.read_json(self.data_path)
        ds = Dataset.from_pandas(df)
        self.tokenized_id = ds.map(self.process_one, remove_columns=ds.column_names)

    def process_one(self, data):
        MAX_LENGTH = 384

        prompt = data['instruction']
        messages = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": prompt}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = self.tokenizer(prompt, add_special_tokens=False)
        response = self.tokenizer(f"{data['output']}<|eot_id|>", add_special_tokens=False)
        input_ids = prompt["input_ids"] + response["input_ids"]
        attention_mask = prompt["attention_mask"] + response["attention_mask"] 
        labels = [-100] * len(prompt["input_ids"]) + response["input_ids"]
        
        if len(input_ids) > MAX_LENGTH:  
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def train(self):
        args = TrainingArguments(
            output_dir=f"./output/{self.model_path.split('/')[-1]}",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            logging_steps=10,
            num_train_epochs=3,
            save_steps=100, 
            learning_rate=1e-4,
            save_on_each_node=True,
            gradient_checkpointing=True
        )
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.tokenized_id,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer, padding=True),
        )
        trainer.train()

    def generate(self, prompt): 
        self.model.eval()
        messages = [
                {"role": "system", "content": self.instruction},
                {"role": "user", "content": prompt}
        ]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([input_ids], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(model_inputs.input_ids,max_new_tokens=512)

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print('A：', prompt)
        print('B：',response)