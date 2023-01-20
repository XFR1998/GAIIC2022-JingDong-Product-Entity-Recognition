import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import warnings
import pandas as pd
from transformers import (AutoModelForMaskedLM,
                          AutoTokenizer, LineByLineTextDataset,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

from transformers import BertTokenizer



model_name = './data/pretrain_model/nezha-cn-base/'

from configuration_nezha import NeZhaConfig
from modeling_nezha import NeZhaForMaskedLM
vocab_file_dir = model_name+'vocab.txt'
tokenizer = BertTokenizer.from_pretrained(vocab_file_dir)
tokenizer.save_pretrained('../data/pretrain_model/my_nezha_cn_base')
config = NeZhaConfig(
    vocab_size=len(tokenizer),
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    max_position_embeddings=512,
)

model = NeZhaForMaskedLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))


train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="../data/tmp_data/pretrain_data_105.txt",  # mention train text file here
    block_size=256)
print('train data over')
valid_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="../data/tmp_data/pretrain_data_105.txt",  # mention valid text file here
    block_size=256)
print('valid data over')
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="../data/pretrain_model/my_pretrain_nezha_chk",  # select model path for checkpoint
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    evaluation_strategy='steps',
    save_total_limit=2,
    eval_steps=500,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    load_best_model_at_end=True,
    prediction_loss_only=True,
    report_to="none")

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset)

print('training start .... ')
trainer.train()
trainer.save_model(f'../data/pretrain_model/my_nezha_cn_base')