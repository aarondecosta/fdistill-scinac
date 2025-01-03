#!/usr/bin/env python

import argparse
import datetime
import json
import time
import warnings
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import calculate_bleu, calculate_rouge, chunks, parse_numeric_n_bool_cl_kwargs, use_task_specific_params


logger = getLogger(__name__)


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_logits_and_labels(
    examples: List[str],
    targets: List[str],
    save_path: str,
    model_name: str,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    fp16=False,
    task="summarization",
    prefix=None,
    **generate_kwargs,
):
    """Save model.generate results to <out_file>, and return how long it took."""
    # fout = Path(out_file).open("w", encoding="utf-8") # just the folder
    model_name = str(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if fp16:
        model = model.half()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Inferred tokenizer type: {tokenizer.__class__}")  # if this is wrong, check config.model_type.
    print(model.model.encoder.embed_tokens.weight.size())

    confidences_list = [] # model confidence for top-1 prediction
    predictions_list = [] # model top-1 prediction
    targets_list = [] # tokenized target summarizations

    start_time = time.time()
    # update config with task specific params
    use_task_specific_params(model, task)
    if prefix is None:
        prefix = prefix or getattr(model.config, "prefix", "") or ""
    for examples_chunk in tqdm(list(chunks(examples, batch_size))):
        examples_chunk = [prefix + text for text in examples_chunk]
        batch = tokenizer(examples_chunk, return_tensors="pt", truncation=True, padding="longest").to(device)
        # print(len(tokenizer))
        # print(batch.input_ids.size(), batch.input_ids.max(), batch.input_ids.min())
        # with torch.no_grad():
        #     logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
        output = model.generate(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            max_new_tokens=50,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_scores=True,
            **generate_kwargs,
        )
        sequences = output.sequences
        # num_beams = generate_kwargs.get("num_beams", 6)
        # sequence_scores = torch.stack(output.sequence_scores, dim=1)
        # best_beam_idx = torch.argmax(sequence_scores, dim=1)
        logits = torch.stack(output.scores, dim=1)
        # logits = logits.view(batch_size, num_beams, -1, logits.size(-1))
        # logits = logits[:,best_beam_idx,:,:]
        print(logits.shape)

        # pad logits to max_new_tokens
        max_len = 50
        pad_len = max_len - logits.size(1)
        if pad_len > 0:
            padding = torch.zeros(logits.shape[0], pad_len, logits.shape[2], device=logits.device)
            pad_token_id = tokenizer.pad_token_id
            pad_logits = torch.full_like(padding, float('-inf'))
            pad_logits[:,:,pad_token_id] = float('inf')
            logits = torch.cat([logits, pad_logits], dim=1)

        print(logits.shape)

        softmaxes = torch.nn.functional.softmax(logits, dim=-1)
        conf, pred = torch.max(softmaxes, dim=-1)
        print(conf.shape)
        print(pred.shape)
        print()
        confidences_list.append(conf)
        predictions_list.append(pred)
        
    for targets_chunk in tqdm(list(chunks(targets, batch_size))):
        batch = tokenizer(targets_chunk, return_tensors="pt", truncation=True, padding="max_length", max_length=50).to(device)
        tokenized_targets = batch.input_ids
        # print(tokenized_targets.shape)
        targets_list.append(tokenized_targets)

    confidences = torch.cat(confidences_list, 0)
    predictions = torch.cat(predictions_list, 0)
    targets = torch.cat(targets_list, 0)

    print(confidences.shape)
    print(predictions.shape)
    print(targets.shape)

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(confidences.cpu().detach(), save_path / "confidences.pt")
    torch.save(predictions.cpu().detach(), save_path / "predictions.pt")
    torch.save(targets.cpu().detach(), save_path / "labels.pt")


    runtime = int(time.time() - start_time)  # seconds
    n_obs = len(examples)
    return dict(n_obs=n_obs, runtime=runtime, seconds_per_sample=round(runtime / n_obs, 4))



def datetime_now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_generate(verbose=True):
    """

    Takes input text, calculates confidences and predictions for generation, and saves as pytorch tensors.

    Args:
        verbose (:obj:`bool`, `optional`, defaults to :obj:`True`): print results to stdout
        
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("--input_path", type=str, help="like cnn_dm/test.source")
    parser.add_argument("--save_path", type=str, help="where to save summaries")
    parser.add_argument("--reference_path", type=str, required=False, help="like cnn_dm/test.target")
    parser.add_argument("--score_path", type=str, required=False, default="metrics.json", help="where to save metrics")
    parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
    parser.add_argument(
        "--prefix", type=str, required=False, default=None, help="will be added to the begininng of src examples"
    )
    parser.add_argument("--task", type=str, default="summarization", help="used for task_specific_params + metrics")
    parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
    parser.add_argument(
        "--n_obs", type=int, default=-1, required=False, help="How many observations. Defaults to all."
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--dump-args", action="store_true", help="print the custom hparams with the results")
    parser.add_argument(
        "--info",
        nargs="?",
        type=str,
        const=datetime_now(),
        help="use in conjunction w/ --dump-args to print with the results whatever other info you'd like, e.g. lang=en-ru. If no value is passed, the current datetime string will be used.",
    )
    # Unspecified args like --num_beams=2 --decoder_start_token_id=4 are passed to model.generate
    args, rest = parser.parse_known_args()
    parsed_args = parse_numeric_n_bool_cl_kwargs(rest)
    if parsed_args and verbose:
        print(f"parsed the following generate kwargs: {parsed_args}")
    examples = [" " + x.rstrip() if "t5" in args.model_name else x.rstrip() for x in open(args.input_path).readlines()]
    targets = [" " + x.rstrip() if "t5" in args.model_name else x.rstrip() for x in open(args.reference_path).readlines()]
    if args.n_obs > 0:
        examples = examples[: args.n_obs]
        targets = targets[: args.n_obs]
    Path(args.save_path).parent.mkdir(exist_ok=True)
    if args.reference_path is None and Path(args.score_path).exists():
        warnings.warn(f"score_path {args.score_path} will be overwritten unless you type ctrl-c.")
    runtime_metrics = get_logits_and_labels(
        examples,
        targets,
        args.save_path,
        args.model_name,
        batch_size=args.bs,
        device=args.device,
        fp16=args.fp16,
        task=args.task,
        prefix=args.prefix,
        **parsed_args,
    )

    if args.reference_path is None:
        return {}

if __name__ == "__main__":
    # Usage for MT:
    # python run_eval.py MODEL_NAME $DATA_DIR/test.source $save_dir/test_translations.txt --reference_path $DATA_DIR/test.target --score_path $save_dir/test_bleu.json  --task translation $@
    run_generate(verbose=True)
