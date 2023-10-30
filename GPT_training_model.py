# import necessary packages
import transformers
import pandas as pd
from datasets import Dataset, load_metric
import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer

# load dataset as pandas Dataframe
df = pd.read_csv("Training Data.csv")

# check
df

# load the dataframe in a hugging face compatible format
dataset = Dataset.from_pandas(df)

# check the type
type(dataset)

# encode the dataset labels as integers
dataset = dataset.class_encode_column('label')

# view a sample of the dataset
dataset[2]

# verify the dataset features
dataset.features

# declare the checkpoint
checkpoint = "bert-base-uncased"

# call the tokenizer for training
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# create a function for tokenizing the sample_mails
def tokenize_function(example):
    return tokenizer(example["Email Text"], truncation=True)

# tokenize the dataset with the map function
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets

# apply dynamic padding -- pad all the sample_mails to the length of the longest element when we batch elements together
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

fields_to_remove = ["Email Text"]

# Remove the specified fields
tokenized_datasets = tokenized_datasets.remove_columns(fields_to_remove)

# we can convert the tokenized dataset back to text as follows
tokenizer.convert_ids_to_tokens(tokenized_datasets['input_ids'][-1])

# define a metric to monitor during training
metric = load_metric("accuracy")

# create a function that helps compute the specified metric


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# define the training arguments
training_args = TrainingArguments('training_args',
                                  num_train_epochs=20)


model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=2)

# define trainer object
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)



# train the model
trainer.train()

# save the trained model together with the tokenizer in a directory
trainer.save_model('custom_model')
