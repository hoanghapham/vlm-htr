#%%
import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, load_from_disk,  Dataset
import random
import numpy as np
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--batch-size", default=4)
parser.add_argument("--take", default=100)
args = parser.parse_args()

BATCH_SIZE = int(args.batch_size)
TAKE = int(args.take)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
#%%

torch.cuda.empty_cache()

# train_subset = Dataset.load_from_disk('./finetuning_data/train_subset')
# eval_subset= Dataset.load_from_disk('./finetuning_data/eval_subset')

from datasets import load_dataset

train_subset_stream = load_dataset("oscar-corpus/OSCAR-2201",
                        # use_auth_token=True, # required
                        language="sv", 
                        streaming=True, # optional
                        split="train",
                        trust_remote_code=True) # optional, but the dataset only has a train split

train_subset = train_subset_stream.take(TAKE)
eval_subset = train_subset_stream.shuffle().take(TAKE)

#%%
# Tokenizer
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

# Apply tokenization
train_subset = train_subset.map(tokenize_function, batched=True)
eval_subset = eval_subset.map(tokenize_function, batched=True)

train_subset = train_subset.remove_columns(["text"])
eval_subset = eval_subset.remove_columns(["text"])

# Randomly sampling to test
#train_subset = train_subset.shuffle(seed=42).select(range(30000))
#print(f"Number of rows in the reduced train dataset: {len(train_subset)}")
#eval_subset = eval_subset.shuffle(seed=42).select(range(3000))
#print(f"Number of rows in the reduced eval dataset: {len(eval_subset)}")

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
# out = data_collator([train_subset[i] for i in range(5)])
# for key in out:
#     print(f"{key} shape: {out[key].shape}")




#%%
# Random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# Training arguments
training_args = TrainingArguments(
    output_dir='./checkpoints',
    # evaluation_strategy='steps',
    # eval_steps=100,  # Evaluate after every 1000 steps. 
    save_strategy='steps',
    save_steps=50, # must be a round multiple of the evaluation steps for early stopping to work??
    num_train_epochs=3,
    load_best_model_at_end=True, # Necessary for early stopping
    per_device_train_batch_size=BATCH_SIZE,
    # per_device_eval_batch_size=BATCH_SIZE,
    logging_dir='./logs',
    logging_steps=50,
    seed=seed,
    learning_rate=2e-5, # default is 5e-5? 
    weight_decay=0.01,
    gradient_accumulation_steps=2,
    # metric_for_best_model="eval_loss",  # Track evaluation loss
    max_steps=200
    #fp16=False,
    #bf16=True,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    loss_fct = torch.nn.CrossEntropyLoss()
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return {"eval_loss": loss.item(), "perplexity": torch.exp(loss).item()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_subset,
    # eval_dataset=eval_subset,
    data_collator=data_collator,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Stop if no improvement after 3 evaluations
    # compute_metrics=compute_metrics,
)

torch.cuda.empty_cache()

# Training the model
trainer.train()

model_output_dir = 'models/gpt2'
model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)
# %%
