from flask import Flask, render_template, request
from WebScraper import webscraper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import TFAutoModelForSeq2SeqLM
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_metric
from typing import List, Iterable, Callable
from torch import nn
import time
import torch
import argparse

app = Flask(__name__, template_folder='template')

args_dict = dict(
    output_dir="", # path to save the checkpoints
    model_name_or_path='t5-small',
    tokenizer_name_or_path='t5-small',
    max_input_length=512,
    max_output_length=150,
    freeze_encoder=False,
    freeze_embeds=False,
    learning_rate=10e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=1,
    eval_batch_size=1,
    num_train_epochs=1,
    gradient_accumulation_steps=8,
    n_gpu=1,
    resume_from_checkpoint=None,
    val_check_interval = 0.5,
    n_val=1000,
    n_train=-1,
    n_test=-1,
    early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)

args_dict.update({'output_dir': 'New_Summarizer_En', 'num_train_epochs': 2,
                 'train_batch_size': 1, 'eval_batch_size': 4})
args = argparse.Namespace(**args_dict)

class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
        self.rouge_metric = load_metric('rouge')

        if self.hparams.freeze_embeds:
            self.freeze_embeds()
        if self.hparams.freeze_encoder:
            self.freeze_params(self.model.get_encoder())
            self.assert_all_frozen(self.model.get_encoder())

        n_observations_per_split = {
            "train": self.hparams.n_train,
            "validation": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

    def grad_status(self, model: nn.Module) -> Iterable:
        return (par.requires_grad for par in model.parameters())

    def lmap(self, f: Callable, x: Iterable) -> List:
        return list(map(f, x))

    def assert_all_frozen(self, model):
        model_grads: List[bool] = list(self.grad_status(model))
        n_require_grad = sum(self.lmap(int, model_grads))
        npars = len(model_grads)
        assert not any(model_grads), f"{n_require_grad / npars:.1%} of {npars} weights require grad"

    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            self.freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                self.freeze_params(d.embed_positions)
                self.freeze_params(d.embed_tokens)
        except AttributeError:
            self.freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                self.freeze_params(d.embed_tokens)

    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        return self.lmap(str.strip, gen_text)

    def _generative_step(self, batch):


        generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            decoder_attention_mask=batch['target_mask'],
            max_length=150,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=False
        )
        preds = self.ids_to_clean_text(generated_ids)
        return preds

model_en = T5FineTuner(args)
model_en.load_state_dict(torch.load("model_wikisum_en.model"))
model = AutoModelForSeq2SeqLM.from_pretrained("model_wikisum_tr.model")
model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained("model_wikisum_tr.model")
tokenizer_en = T5Tokenizer.from_pretrained('t5-small')


def predict(text):
    preds = ""
    print(len(text))
    if len(text) > 1000:
        for i in range(0, int(len(text)/500)):
            source_encoding = tokenizer(
                text[i*500:(i+1)*500],
                max_length=786,
                # padding="max_length",
                # truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt")

            source_encoding["input_ids"] = source_encoding["input_ids"].to("cuda")
            source_encoding["attention_mask"] = source_encoding["attention_mask"].to("cuda")

            generated_ids = model.generate(
                input_ids=source_encoding["input_ids"],
                attention_mask=source_encoding["attention_mask"],
                num_beams=2,
                max_length=786,
                repetition_penalty=2.5,
                length_penalty=1.5,
                early_stopping=False,
                use_cache=True
            )
            preds += "".join([tokenizer.decode(gen_id, skip_special_tokens=True)
                    for gen_id in generated_ids])
            preds += " "
        return preds
    else:
        source_encoding = tokenizer(
            text,
            max_length=786,
            # padding="max_length",
            # truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt")

        generated_ids = model.generate(
            input_ids=source_encoding["input_ids"],
            attention_mask=source_encoding["attention_mask"],
            num_beams=2,
            max_length=786,
            repetition_penalty=2.5,
            length_penalty=1.5,
            early_stopping=False,
            use_cache=False
        )
        preds += "".join([tokenizer.decode(gen_id, skip_special_tokens=True)
                          for gen_id in generated_ids])
        return preds


def predict_en(text):
    text = text.replace('\n', '')
    text = text.replace('``', '')
    text = text.replace('"', '')
    source_encoding = tokenizer_en(
        text,
        max_length=786,
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=False,
        return_tensors="pt")

    batch = {'source_ids': source_encoding["input_ids"], 'source_mask': source_encoding["attention_mask"],
             'target_mask': None}
    generated_ids = model_en._generative_step(
        batch
    )

    return generated_ids[0]

@app.route('/tr', methods=["GET", "POST"])
def home():
    if request.method == "POST":
        url = request.form.get("urlInput")
        if url.replace(" ","") == "":
            word = request.form.get("wordInput")
            if word.replace(" ","") == "":
                return render_template('MainScreenTr.html')
            else:
                output = predict(word)
                return render_template('ResultScreenTr.html', output=output, input=word)
        else:
            word = webscraper(url)
            output = predict(word)
            return render_template('ResultScreenTr.html', output=output, input=word)
    return render_template('MainScreenTr.html')


@app.route('/en', methods=["GET", "POST"])
def home_en():
    if request.method == "POST":
        url = request.form.get("urlInput")
        if url.replace(" ","") == "":
            word = request.form.get("wordInput")
            if word.replace(" ","") == "":
                return render_template('MainScreenEn.html')
            else:
                output = predict_en(word)
                return render_template('ResultScreenEn.html', output=output, input=word)
        else:
            word = webscraper(url)
            output = predict_en(word)
            return render_template('ResultScreenEn.html', output=output, input=word)
    return render_template('MainScreenEn.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8181)




