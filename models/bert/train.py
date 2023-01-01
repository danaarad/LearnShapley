import random
import torch
from torch.utils.data.dataloader import default_collate

from transformers import BertTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from models.bert.dataset import QueriesDataset


def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)

    return {"mse": mse, "mae": mae, "r2": r2}


class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = torch.squeeze(outputs.logits)
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train(args):
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    args.ntokens = len(tokenizer.vocab)

    print("creating datasets")
    train_data = QueriesDataset(args.max_results_for_train, "train", data_path=args.data)
    dev_data = QueriesDataset(args.max_results_for_eval, "dev", data_path=args.data)
    print(f"loaded {len(train_data)} train triplets, {len(dev_data)} dev triplets")

    ###############################################################################
    # Build the model
    ###############################################################################

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
    training_args = TrainingArguments(
        output_dir=args.save,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=10,
        metric_for_best_model="mse",
        load_best_model_at_end=True,
        weight_decay=0.01,
        gradient_accumulation_steps=args.grad_accumulation
        
    )
    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        compute_metrics=compute_metrics_for_regression,
        data_collator=default_collate
    )

    ###############################################################################
    # Training code
    ###############################################################################

    print("training")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
