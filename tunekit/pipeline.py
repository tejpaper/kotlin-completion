from tunekit.constants import PAST_CTX_LEN, FUTURE_CTX_LEN
from tunekit.dataset import truncate_and_tokenize_context

import torch

from transformers import Pipeline
from transformers.utils import ModelOutput

from colorama import Fore
from tqdm.autonotebook import trange


class MaskFillingBeamSearch(Pipeline):
    def __init__(self, model, tokenizer,
                 beam_width: int = 5,
                 max_search_depth: int = 20,
                 max_n_completions: int = 50,
                 past_ctx_len: int = PAST_CTX_LEN,
                 future_ctx_len: int = FUTURE_CTX_LEN,
                 verbose: bool = True,
                 colorize: bool = True) -> None:
        super().__init__(model, tokenizer)

        self.beam_width = beam_width
        self.max_search_depth = max_search_depth
        self.max_n_completions = max_n_completions
        self.past_ctx_len = past_ctx_len
        self.future_ctx_len = future_ctx_len
        self.verbose = verbose
        self.colorize = colorize

        # end of line tokens serve as EOS token in this task
        self.eol_token_ids = torch.tensor([
            token_id for token_id in range(tokenizer.vocab_size)
            if '\n' in tokenizer.decode(token_id) and token_id not in tokenizer.all_special_ids
        ])
        self.n_eol_tokens = self.eol_token_ids.shape[0]

    def _sanitize_parameters(self, **kwargs) -> tuple[dict, dict, dict]:
        preprocess_params = {k: v for k, v in kwargs.items()
                             if k in ('past_ctx_len',
                                      'future_ctx_len')}
        forward_params = {k: v for k, v in kwargs.items()
                          if k in ('beam_width',
                                   'max_search_depth',
                                   'max_n_completions',
                                   'verbose',
                                   'past_ctx_len',
                                   'future_ctx_len')}
        postprocess_params = {k: v for k, v in kwargs.items()
                              if k in ('colorize',)}
        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, masked_code: str, **preprocess_parameters) -> dict:
        past_ctx_len = preprocess_parameters.get('past_ctx_len', self.past_ctx_len)
        future_ctx_len = preprocess_parameters.get('future_ctx_len', self.future_ctx_len)

        n_masks = masked_code.count(self.tokenizer.mask_token)
        if n_masks != 1:
            if n_masks == 0:
                raise ValueError('The input string must contain one mask token.')
            raise ValueError(f'Only one line completion is supported at a time. '
                             f'Found {n_masks} mask tokens.')

        past_ctx, future_ctx = masked_code.split(self.tokenizer.mask_token)
        dummy_target = torch.tensor([[1]])

        tokens, attn_mask = truncate_and_tokenize_context(
            self.tokenizer, dummy_target, past_ctx, future_ctx,
            past_ctx_len, future_ctx_len)

        return dict(original_string=masked_code, input_ids=tokens, attention_mask=attn_mask)

    @torch.no_grad()
    def _forward(self, model_inputs: dict, **forward_parameters) -> dict:
        beam_width = forward_parameters.get('beam_width', self.beam_width)
        max_search_depth = forward_parameters.get('max_search_depth', self.max_search_depth)
        max_n_completions = forward_parameters.get('max_n_completions', self.max_n_completions)
        verbose = forward_parameters.get('verbose', self.verbose)
        past_ctx_len = forward_parameters.get('past_ctx_len', self.past_ctx_len)
        future_ctx_len = forward_parameters.get('future_ctx_len', self.future_ctx_len)
        ctx_len = past_ctx_len + future_ctx_len

        device = model_inputs['input_ids'].device
        original_string = model_inputs.pop('original_string')

        # creating empty beams
        beam_comps = torch.full((1, max_search_depth), self.tokenizer.pad_token_id, device=device)
        beam_comps_log_probs = torch.full((1, max_search_depth), torch.nan, device=device)

        # preparing output-leaderboard storage
        final_comps = torch.tensor([], dtype=torch.long, device=device)
        final_comps_log_probs = torch.tensor([], dtype=torch.float32, device=device)

        # variables for input length treatment
        padding_len = (model_inputs['attention_mask'][0] == 0).sum()
        _, mask_idx = torch.nonzero(model_inputs['input_ids'] == self.tokenizer.mask_token_id, as_tuple=True)

        for t in trange(max_search_depth, disable=not verbose):
            # getting logits from the model
            model_outputs = self.model(**model_inputs)
            logits = model_outputs.logits

            # two computational branches: regular tokens and EOL
            log_probs = torch.log_softmax(logits, dim=-1)
            eol_log_probs = log_probs[:, self.eol_token_ids]

            # filling in all possible line endings
            fork_comps = beam_comps.unsqueeze(0).repeat_interleave(self.n_eol_tokens, dim=0)
            fork_comps_log_probs = beam_comps_log_probs.unsqueeze(0).repeat_interleave(self.n_eol_tokens, dim=0)

            fork_comps[..., t] = self.eol_token_ids.unsqueeze(-1)
            fork_comps_log_probs[..., t] = eol_log_probs.mT

            fork_comps = fork_comps.flatten(end_dim=1)
            fork_comps_log_probs = fork_comps_log_probs.flatten(end_dim=1)

            # output table update
            final_comps = torch.cat([final_comps, fork_comps])
            final_comps_log_probs = torch.cat([final_comps_log_probs, fork_comps_log_probs])

            final_pruning_k = min(final_comps.shape[0], max_n_completions)
            _, final_idx = torch.topk(final_comps_log_probs.nanmean(dim=-1), k=final_pruning_k)

            final_comps = final_comps[final_idx]
            final_comps_log_probs = final_comps_log_probs[final_idx]

            if t == max_search_depth - 1:
                continue

            # continue construction of the beams
            log_probs[:, self.eol_token_ids] = -torch.inf
            conditioned_log_probs = (log_probs + beam_comps_log_probs.nansum(dim=-1, keepdim=True)) / (t + 1)

            # getting the following best tokens
            _, beam_tokens_flatten_idx = torch.topk(conditioned_log_probs.flatten(), k=beam_width)
            beam_idx, beam_tokens_ids = torch.unravel_index(beam_tokens_flatten_idx, log_probs.shape)

            # beams update
            beam_comps = beam_comps[beam_idx]
            beam_comps_log_probs = beam_comps_log_probs[beam_idx]

            beam_comps[:, t] = beam_tokens_ids
            beam_comps_log_probs[:, t] = log_probs.flatten()[beam_tokens_flatten_idx]

            # inputs update
            input_ids = model_inputs['input_ids'].expand(beam_width, ctx_len + 1)
            input_ids = torch.cat([
                input_ids[:, :mask_idx],
                beam_tokens_ids.unsqueeze(-1),
                input_ids[:, mask_idx:],
            ], dim=-1)

            if padding_len > 0 or mask_idx < past_ctx_len:
                padding_len -= 1
                mask_idx += 1
                input_ids = input_ids[:, :-1]
            else:
                input_ids = input_ids[:, 1:]

            model_inputs = dict(
                input_ids=input_ids,
                attention_mask=input_ids != self.tokenizer.pad_token_id,
            )

        n_tokens2pred = self.tokenizer.vocab_size - len(self.tokenizer.all_special_ids)
        scores = -torch.log(torch.tensor(n_tokens2pred, device=device)) / final_comps_log_probs.nanmean(-1)

        return dict(
            original_string=original_string,
            completion_tokens=final_comps,
            tokens_probs=final_comps_log_probs.exp(),
            scores=scores,
        )

    def postprocess(self, model_outputs: dict, **postprocess_parameters) -> dict:
        colorize = postprocess_parameters.get('colorize', self.colorize)

        completions = [
            comp.replace(self.tokenizer.pad_token, '')
            for comp in self.tokenizer.batch_decode(model_outputs['completion_tokens'])
        ]

        original_string = model_outputs.pop('original_string')
        filled_string = [
            original_string.replace(
                self.tokenizer.mask_token,
                ('', Fore.RED)[colorize] + comp + ('', Fore.RESET)[colorize]
            ) for comp in completions
        ]

        return model_outputs | ModelOutput(
            completions=completions,
            filled_string=filled_string,
        )
