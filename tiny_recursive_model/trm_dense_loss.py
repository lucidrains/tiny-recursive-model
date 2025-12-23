from __future__ import annotations

import torch
from torch import nn, stack, cat, arange, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Reduce, Rearrange

# ein

# b - batch
# n - sequence
# t - refinement steps
# d - feature dimension

# network related

from tiny_recursive_model.mlp_mixer_1d import MLPMixer1D

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def is_empty(t):
    return t.numel() == 0

def range_from_one(n):
    return range(1, n + 1)

# classes

class TinyRecursiveModel(Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        network: Module,
        num_refinement_blocks = 3,   # T in paper
        num_latent_refinements = 6,  # n in paper - 1 output refinement per N latent refinements
        halt_loss_weight = 1.,
        num_register_tokens = 0,
        recurrent_grad_depth = 1 # in TRM, they only have gradients through last step, but can increase this
    ):
        super().__init__()
        assert num_refinement_blocks > 1

        self.input_embed = nn.Embedding(num_tokens, dim)
        self.output_init_embed = nn.Parameter(torch.randn(dim) * 1e-2)
        self.latent_init_embed = nn.Parameter(torch.randn(dim) * 1e-2)

        self.network = network

        self.num_latent_refinements = num_latent_refinements
        self.num_refinement_blocks = num_refinement_blocks

        # register tokens for the self attend version

        self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim) * 1e-2)

        # prediction heads

        self.to_pred = nn.Linear(dim, num_tokens, bias = False)

        self.to_halt_pred = nn.Sequential(
            Reduce('... n d -> ... d', 'mean'),
            nn.Linear(dim, 1, bias = False),
            Rearrange('... 1 -> ...')
        )

        self.halt_loss_weight = halt_loss_weight

        # init

        nn.init.zeros_(self.to_halt_pred[1].weight)

        # like urm

        assert 1 <= recurrent_grad_depth <= self.num_refinement_blocks

        self.recurrent_grad_depth = recurrent_grad_depth

    @property
    def device(self):
        return next(self.parameters()).device

    def get_initial(self):
        outputs = self.output_init_embed
        latents = self.latent_init_embed

        return outputs, latents

    def embed_inputs_with_registers(
        self,
        seq
    ):
        batch = seq.shape[0]

        inputs = self.input_embed(seq)

        # maybe registers

        registers = repeat(self.register_tokens, 'n d -> b n d', b = batch)

        inputs, packed_shape = pack([registers, inputs], 'b * d')

        return inputs, packed_shape

    def refine_latent_then_output_once(
        self,
        inputs,     # (b n d)
        outputs,    # (b n d)
        latents,    # (b n d)
    ):

        # so it seems for this work, they use only one network
        # the network learns to refine the latents if input is passed in, otherwise it refines the output

        for _ in range(self.num_latent_refinements):

            latents = self.network(outputs + latents + inputs)

        outputs = self.network(outputs + latents)

        return outputs, latents

    def deep_refinement(
        self,
        inputs,    # (b n d)
        outputs,   # (b n d)
        latents,   # (b n d)
    ):             # (t b n d), (b n d)

        all_outputs = []

        for step in range_from_one(self.num_refinement_blocks):

            is_recurrent_grad_step = step > (self.num_refinement_blocks - self.recurrent_grad_depth)

            outputs, latents = self.refine_latent_then_output_once(inputs, outputs, latents)

            all_outputs.append(outputs)

            if not is_recurrent_grad_step:
                outputs, latents = tuple(t.detach() for t in (outputs, latents))

        return stack(all_outputs), latents

    @torch.no_grad()
    def predict(
        self,
        seq,
        halt_prob_thres = 0.5,
        max_deep_refinement_steps = 12
    ):
        batch = seq.shape[0]

        inputs, packed_shape = self.embed_inputs_with_registers(seq)

        # initial outputs and latents

        outputs, latents = self.get_initial()

        # active batch indices, the step it exited at, and the final output predictions

        active_batch_indices = arange(batch, device = self.device, dtype = torch.long)

        preds = []
        exited_step_indices = []
        exited_batch_indices = []

        for step in range_from_one(max_deep_refinement_steps):
            is_last = step == max_deep_refinement_steps

            all_outputs, latents = self.deep_refinement(inputs, outputs, latents)

            outputs = all_outputs[-1] # use last refinement step output for prediction

            halt_prob = self.to_halt_pred(outputs).sigmoid()

            should_halt = (halt_prob >= halt_prob_thres) | is_last

            if not should_halt.any():
                continue

            # maybe remove registers

            registers, outputs_for_pred = unpack(outputs, packed_shape, 'b * d')

            # append to exited predictions

            pred = self.to_pred(outputs_for_pred[should_halt])
            preds.append(pred)

            # append the step at which early halted

            exited_step_indices.extend([step] * should_halt.sum().item())

            # append indices for sorting back

            exited_batch_indices.append(active_batch_indices[should_halt])

            if is_last:
                continue

            # ready for next round

            inputs = inputs[~should_halt]
            outputs = outputs[~should_halt]
            latents = latents[~should_halt]
            active_batch_indices = active_batch_indices[~should_halt]

            if is_empty(outputs):
                break

        preds = cat(preds).argmax(dim = -1)
        exited_step_indices = tensor(exited_step_indices)

        exited_batch_indices = cat(exited_batch_indices)
        sort_indices = exited_batch_indices.argsort(dim = -1)

        return preds[sort_indices], exited_step_indices[sort_indices]

    def forward(
        self,
        seq,
        outputs,
        latents,
        labels = None
    ):

        inputs, packed_shape = self.embed_inputs_with_registers(seq)

        all_outputs, latents = self.deep_refinement(inputs, outputs, latents)

        registers, outputs_for_pred = unpack(all_outputs, packed_shape, 't b * d')

        pred = self.to_pred(outputs_for_pred) # prediction now across all refinement steps

        halt_logits = self.to_halt_pred(all_outputs)

        last_outputs, last_pred = all_outputs[-1], pred[-1]

        halt_prob = halt_logits[-1].sigmoid()

        outputs, latents = last_outputs.detach(), latents.detach()

        return_package = (outputs, latents, last_pred, halt_prob)

        if not exists(labels):
            return return_package

        # calculate loss if labels passed in

        loss = F.cross_entropy(
            rearrange(pred, 't b n l -> b l n t'),
            repeat(labels, 'b n -> b n t', t = self.num_refinement_blocks),
            reduction = 'none'
        )

        loss = reduce(loss, 'b ... -> b', 'mean')

        is_all_correct = (pred.argmax(dim = -1) == labels).all(dim = -1)

        halt_loss = F.binary_cross_entropy_with_logits(halt_logits, is_all_correct.float(), reduction = 'none')

        halt_loss = reduce(halt_loss, 't b -> b', 'mean')

        # total loss and loss breakdown

        total_loss = (
            loss +
            halt_loss * self.halt_loss_weight
        )

        losses = (loss, halt_loss)

        return (total_loss.sum(), losses, *return_package)
