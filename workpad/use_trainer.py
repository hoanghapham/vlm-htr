from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    num_train_epochs=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    optim="adamw_torch",
    learning_rate=1e-6,
    # lr_scheduler_type="linear",
    # lr_scheduler_kwargs={"optimizer": optimizer, "num_warmup_steps": 0, "num_training_steps":  num_training_steps}
    lr_scheduler_type="linear",
    warmup_steps=0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor,
    data_collator=collate_fn,
    compute_metrics=model.loss_function
)

# Very bad on mac CPU
trainer.train()