from tunekit.constants import PAST_CTX_LEN, FUTURE_CTX_LEN

import re
import typing

import numpy as np
import pandas as pd

import torch

from tqdm.autonotebook import tqdm
from transformers.tokenization_utils_base import BatchEncoding


def whitespace_prefix_mask(code: str) -> np.ndarray:
    """
    Defines a character-based boolean mask to select the whitespace
    parts following before the code at the beginning of each line.

    See the analysis.ipynb for a demonstration.
    """
    mask = list()

    for line in code.split('\n'):
        line += '\n'
        mask.extend([False] * len(line))

        for i, char in enumerate(line, start=len(mask) - len(line)):
            if char.isspace():
                mask[i] = True
            else:
                break

    return np.array(mask[:-1])


def python_comment_mask(code: str) -> np.ndarray:
    """
    Defines a character-based boolean mask for selecting single-line Python comments.

    See the analysis.ipynb for a demonstration.
    """
    code += '\n'
    mask = [False] * len(code)

    single_line_pattern = r'([\t ]*#.*)($|\n)'
    for match in re.finditer(single_line_pattern, code):
        for i in range(match.start(1), match.end(1) + bool(match.group(2))):
            mask[i] = True

    return np.array(mask[:-1])


def python_ignore_chars(code: str) -> np.ndarray:
    return whitespace_prefix_mask(code) | python_comment_mask(code)


def kotlin_comment_mask(code: str) -> np.ndarray:
    """
    Defines a character-based boolean mask for selecting Kotlin comments.

    See the analysis.ipynb for a demonstration.
    """
    code += '\n'
    mask = [False] * len(code)

    single_line_pattern = r'([\t ]*//.*)($|\n)'
    for match in re.finditer(single_line_pattern, code):
        for i in range(match.start(1), match.end(1) + bool(match.group(2))):
            mask[i] = True

    multi_line_pattern = r'([\t ]*/\*.*?\*/)($|\n)'
    for match in re.finditer(multi_line_pattern, code, re.DOTALL):
        for i in range(match.start(1), match.end(1) + bool(match.group(2))):
            mask[i] = True

    return np.array(mask[:-1])


def kotlin_ignore_chars(code: str) -> np.ndarray:
    return whitespace_prefix_mask(code) | kotlin_comment_mask(code)


def precalculate_masks_positions(df: pd.DataFrame,
                                 ignore_chars: typing.Callable,
                                 verbose: bool = True,
                                 ) -> pd.DataFrame:
    tqdm.pandas(desc='Precalculating masks positions', disable=not verbose)
    df['mask_positions'] = df.code.progress_apply(lambda x: np.arange(len(x))[~ignore_chars(x)])
    df = df[df.mask_positions.apply(np.ndarray.__len__) != 0]
    return df


def truncate_and_tokenize_context(tokenizer,
                                  targets: torch.Tensor,
                                  past_ctx: str,
                                  future_ctx: str,
                                  past_ctx_len: int,
                                  future_ctx_len: int,
                                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts the submitted context into an acceptable form for the model.

    The basic rules of tokenization can be described as follows:
        1. Minimal use of padding (the model should receive maximum useful information)
        2. In case of data redundancy - hold the mask on the past_ctx_len index
        3. Number of output tokens equals past_ctx_len + future_ctx_len + 1
    """
    ctx_len = past_ctx_len + future_ctx_len

    tokenizer.truncation_side = 'left'
    tokenizer.padding_side = 'left'
    tokenized_past_ctx = tokenizer(
        past_ctx, padding='max_length', truncation=True, return_tensors='pt', max_length=ctx_len)

    tokenizer.truncation_side = 'right'
    tokenizer.padding_side = 'right'
    tokenized_future_ctx = tokenizer(
        future_ctx, padding='max_length', truncation=True, return_tensors='pt', max_length=ctx_len)

    n_past_ctx_tokens = tokenized_past_ctx.attention_mask.sum(dim=-1)
    n_future_ctx_tokens = tokenized_future_ctx.attention_mask.sum(dim=-1)
    n_past_ctx_paddings = ctx_len - n_past_ctx_tokens

    ctx_start = torch.where(
        condition=n_past_ctx_tokens < past_ctx_len,
        input=n_past_ctx_paddings,
        # elif
        other=torch.where(
            condition=n_future_ctx_tokens < future_ctx_len,
            input=torch.relu(n_future_ctx_tokens - n_past_ctx_paddings) + n_past_ctx_paddings,
            # else
            other=future_ctx_len,
        ))

    offset = ctx_start.unsqueeze(-1) + torch.arange(ctx_len + 1)

    tokens = torch.hstack([
        tokenized_past_ctx.input_ids,
        torch.full_like(targets, tokenizer.mask_token_id),
        tokenized_future_ctx.input_ids,
    ]).gather(dim=1, index=offset)

    attn_mask = torch.hstack([
        tokenized_past_ctx.attention_mask,
        torch.ones_like(targets),
        tokenized_future_ctx.attention_mask,
    ]).gather(dim=1, index=offset)

    return tokens, attn_mask


class Preprocessing:
    def __init__(self, tokenizer,
                 past_ctx_len: int = PAST_CTX_LEN,
                 future_ctx_len: int = FUTURE_CTX_LEN,
                 randomize: bool = True,
                 random_seed: int | None = None) -> None:
        if random_seed is None:
            random_seed = np.random.randint(2 ** 32)

        self.tokenizer = tokenizer

        self.past_ctx_len = past_ctx_len
        self.future_ctx_len = future_ctx_len

        self.randomize = randomize
        self.random_seed = random_seed
        self.random_state = None
        self.reset_random_state()

    def reset_random_state(self) -> None:
        self.random_state = np.random.RandomState(self.random_seed)

    def slice_sample(self, code: str, mask: np.ndarray) -> tuple[str, str, str]:
        if not self.randomize:
            self.reset_random_state()

        mask_start = self.random_state.choice(mask)
        mask_end = code.find('\n', mask_start) % len(code) + 1

        past_ctx = code[:mask_start]
        completion_line = code[mask_start:mask_end]
        future_ctx = code[mask_end:]

        return past_ctx, completion_line, future_ctx

    def transform(self, batch: dict) -> BatchEncoding:
        code, mask_positions, *_ = batch.values()
        past_ctx, completion_line, future_ctx = zip(*(
            self.slice_sample(c, m) for c, m in zip(code, mask_positions)
        ))

        targets = self.tokenizer(
            completion_line, padding=False, truncation=True, return_tensors='pt',
            return_attention_mask=False, max_length=1).input_ids

        tokens, attn_mask = truncate_and_tokenize_context(
            self.tokenizer, targets, past_ctx, future_ctx,
            self.past_ctx_len, self.future_ctx_len)

        return BatchEncoding(data=dict(
            input_ids=tokens,
            attention_mask=attn_mask,
            labels=targets,
        ))
