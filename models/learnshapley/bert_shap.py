import random
import torch
from glob import glob
from torch import nn
from torch.utils.data.dataloader import default_collate

from transformers import BertTokenizer
from transformers import BertModel
from transformers import TrainingArguments, Trainer

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from models.learnshapley.dataset import QueriesDataset
from models.learnshapley.bert_sim import BertSimModel


class BertShapModel(nn.Module):
    def __init__(self, args):
        super(BertShapModel, self).__init__()
        self.args = args
        self.src_mask = None
        self.bert_sim_model = BertSimModel(args)
        checkpoint = [fname for fname in glob(f"{args.sim_checkpoint}/*") if "pytorch_model.bin" in fname][0] 
        self.bert_sim_model.load_state_dict(torch.load(checkpoint))
        
        self.shap = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(args.nhid, 1))

        self.init_weights()

    def init_weights(self):
        self.shap.apply(init_weights)

    def forward(self, input_ids, attention_mask, labels=None):
        cls_hidden_states, _ = self.bert_sim_model(input_ids=input_ids, attention_mask=attention_mask)
        shap = self.shap(cls_hidden_states)
        return shap


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        

class ShapTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = torch.squeeze(outputs)
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
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    args.ntokens = len(tokenizer.vocab)

    print("creating datasets")
    train_data = QueriesDataset(args.max_results_for_train, "train", data_path=args.data, percent=args.queries_percent_for_train)
    dev_data = QueriesDataset(args.max_results_for_eval, "dev", data_path=args.data, percent=args.queries_percent_for_train)
    print(f"loaded {len(train_data)} train triplets, {len(dev_data)} dev triplets")

    ###############################################################################
    # Build the model
    ###############################################################################

    model = BertShapModel(args)
    training_args = TrainingArguments(
        output_dir=args.save,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        weight_decay=0.01,
        gradient_accumulation_steps=args.grad_accumulation
        
    )
    trainer = ShapTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
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
