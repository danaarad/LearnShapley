import random
import torch
from torch import nn
from torch.utils.data.dataloader import default_collate

from transformers import BertTokenizer
from transformers import BertModel
from transformers import TrainingArguments, Trainer

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from models.learnshapley.dataset import SimilarityDataset

LOSS_WEIGHTS = dict()


class BertSimModel(nn.Module):
    def __init__(self, args):
        super(BertSimModel, self).__init__()
        self.args = args
        self.src_mask = None
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.use_sim_s = args.use_sim_s
        self.use_sim_r = args.use_sim_r
        self.use_sim_w = args.use_sim_w

        if self.use_sim_s:
            self.sim_s = nn.Sequential(
                nn.Dropout(args.dropout),
                nn.Linear(args.nhid, 1))
        
        if self.use_sim_r:
            self.sim_r = nn.Sequential(
                nn.Dropout(args.dropout),
                nn.Linear(args.nhid, 1))

        if self.use_sim_w:
            self.sim_w = nn.Sequential(
                nn.Dropout(args.dropout),
                nn.Linear(args.nhid, 1))

        self.init_weights()

    def init_weights(self):
        if self.use_sim_s:
            self.sim_s.apply(init_weights)
        if self.use_sim_r:
            self.sim_r.apply(init_weights)
        if self.use_sim_w:
            self.sim_w.apply(init_weights)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs[0] # from https://huggingface.co/transformers/v3.0.2/model_doc/bert.html#transformers.BertModel
        cls_hidden_states = last_hidden_states[:, 0, :]

        sim_ouput = dict()
        if self.use_sim_s:
            sim_s = self.sim_s(cls_hidden_states)
            sim_ouput["sim_s"] = sim_s
        if self.use_sim_r:
            sim_r = self.sim_r(cls_hidden_states)
            sim_ouput["sim_r"] = sim_r
        if self.use_sim_w:
            sim_w = self.sim_w(cls_hidden_states)
            sim_ouput["sim_w"] = sim_w

        return cls_hidden_states, sim_ouput


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        

class SimilarityTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")        
        outputs = model(**inputs)
        _, sims_output = outputs
        loss = 0
        for sim_name, sim_output in sims_output.items():
            if sim_output.shape == (1, 1):
                sim_output = torch.squeeze(sim_output, dim=1)
            else:
                sim_output = torch.squeeze(sim_output)
            sim_gold = labels[sim_name]

            if sim_output.shape != sim_gold.shape:
                print("hi")
            
            mask =  sim_gold != -1
            sim_gold = sim_gold[mask]
            sim_output = sim_output[mask]
            
            if sim_output.device != sim_gold.device:
                # print("Not on the same device: ", sim_output.device, sim_gold.device)
                sim_gold = sim_gold.to(sim_output.device)
                # print(sim_output.device, sim_gold.device)

            sim_loss = torch.nn.functional.mse_loss(sim_output, sim_gold)
            sim_weight = LOSS_WEIGHTS[sim_name]

            loss += sim_weight * sim_loss

        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(
        self, model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
                loss = loss.mean().detach()
                if prediction_loss_only:
                    return (loss, None, None)


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

    print(f"creating datasets (use_sim_s={args.use_sim_s}, use_sim_w={args.use_sim_w}, use_sim_r={args.use_sim_r})")
    train_data = SimilarityDataset(split="train", use_sim_s=args.use_sim_s, use_sim_w=args.use_sim_w, use_sim_r=args.use_sim_r, args=args)
    dev_data = SimilarityDataset(split="dev", use_sim_s=args.use_sim_s, use_sim_w=args.use_sim_w, use_sim_r=args.use_sim_r, args=args)
    print(f"loaded {len(train_data)} train triplets, {len(dev_data)} dev triplets")

    if args.use_sim_s:
        LOSS_WEIGHTS["sim_s"] = args.sim_s_weight
    if args.use_sim_r:
        LOSS_WEIGHTS["sim_r"] = args.sim_r_weight
    if args.use_sim_w:
        LOSS_WEIGHTS["sim_w"] = args.sim_w_weight

    ###############################################################################
    # Build the model
    ###############################################################################

    model = BertSimModel(args)
    training_args = TrainingArguments(
        output_dir=args.save,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        weight_decay=0.01,
        gradient_accumulation_steps=args.grad_accumulation,
        prediction_loss_only=True
        
    )
    trainer = SimilarityTrainer(
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
