import pytest
param = pytest.mark.parametrize

import torch

from tiny_recursive_model.trainer import Trainer

@param('use_self_attn', (False, True))
@param('dense_loss', (False, True))
@param('registers', (0, 4))
def test_trm(
    use_self_attn,
    dense_loss,
    registers
):
    from torch.optim import AdamW

    if dense_loss:
        from tiny_recursive_model.trm import TinyRecursiveModel
    else:
        from tiny_recursive_model.trm_dense_loss import TinyRecursiveModel

    if use_self_attn:
        from x_transformers import Encoder
        network = Encoder(dim = 512, depth = 2)
    else:
        from tiny_recursive_model.mlp_mixer_1d import MLPMixer1D
        network = MLPMixer1D(dim = 512, depth = 2, seq_len = 1024 + registers)

    trm = TinyRecursiveModel(
        dim = 512,
        num_tokens = 256,
        num_register_tokens = registers,
        network = network
    )

    optim = AdamW(trm.parameters(), lr = 1e-4)

    seq = torch.randint(0, 256, (2, 1024))
    answer = torch.randint(0, 256, (2, 1024))

    outputs, latents = trm.get_initial()

    for _ in range(3):
        loss, losses, outputs, latents, pred, halt = trm(seq, outputs, latents, labels = answer)

        loss.backward()
        optim.step()
        optim.zero_grad()

    pred_answer, exit_indices = trm.predict(seq)

@param('dense_loss', (False, True))
def test_trainer(
    dense_loss
):
    from torch.utils.data import Dataset
    from tiny_recursive_model.mlp_mixer_1d import MLPMixer1D

    if dense_loss:
        from tiny_recursive_model.trm import TinyRecursiveModel
    else:
        from tiny_recursive_model.trm_dense_loss import TinyRecursiveModel

    trm = TinyRecursiveModel(
        dim = 16,
        num_tokens = 256,
        network = MLPMixer1D(
            dim = 16,
            depth = 2,
            seq_len = 256
        ),
    )

    class MockDataset(Dataset):
        def __len__(self):
            return 16

        def __getitem__(self, idx):
            inp = torch.randint(0, 256, (256,))
            out = torch.randint(0, 256, (256,))
            return inp, out

    trainer = Trainer(
        trm,
        MockDataset(),
        epochs = 1,
        batch_size = 16,
        cpu = True
    )

    trainer()

    pred_answer, exit_indices = trm.predict(torch.randint(0, 256, (1, 256)))

def test_gpt():
    from torch.utils.data import Dataset
    from x_transformers import Decoder
    from tiny_recursive_model.trm import TinyRecursiveModel

    trm = TinyRecursiveModel(
        dim = 16,
        num_tokens = 256,
        network = Decoder(
            dim = 16,
            depth = 2
        ),
    )

    class MockDataset(Dataset):
        def __len__(self):
            return 16

        def __getitem__(self, idx):
            seq = torch.randint(0, 256, (257,))
            return seq[:-1], seq[1:]

    trainer = Trainer(
        trm,
        MockDataset(),
        epochs = 1,
        batch_size = 16,
        cpu = True
    )

    trainer()

    pred_answer, exit_indices = trm.predict(torch.randint(0, 256, (1, 256)))
