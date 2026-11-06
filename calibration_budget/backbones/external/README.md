# Backbones to download

Two of the twelve backbones are not redistributed here. EEG Conformer is released under
GPL-3.0, and the LMDA-Net repository has no license file. Download them from the authors'
repositories and put them in this folder (`calibration_budget/backbones/external/`). The
other ten backbones need no download.

## LMDA-Net (`lmda_model.py`, used unchanged)

1. Download `lmda_model.py` from <https://github.com/MiaoZhengQing/LMDA-Code> into this folder.
2. Its first line is `from torchsummary import summary`. Either run `pip install torchsummary`
   or delete that line. `summary` is used only in the file's `__main__` block.

The class is built as `LMDA(chans=C, samples=T, num_classes=K)` with its default
hyperparameters: depth 9, temporal kernel 75, channel depths 24 and 9, average pooling 5.
Its constructor prints one line with the pooled output shape. Features for head fine-tuning
are the inputs of `classifier` (666 or 675 dimensions for the inputs used here). The
architecture and parameter names match the copy used for the paper. The random
initialization is drawn in a slightly different order, so new training runs are not
bit-identical to the paper's runs.

Miao, Zhao, Zhang and Ming, "LMDA-Net: A lightweight multi-dimensional attention network for
general EEG-based brain-computer interfaces and interpretability", *NeuroImage* 276, 120209
(2023).

## EEG Conformer (`conformer.py`, adapted)

The upstream file `conformer.py` from <https://github.com/eeyhsong/EEG-Conformer> is a
complete training script. It fixes the input at 22 channels and 1000 samples, and its model
returns a tuple. Create `conformer.py` in this folder as follows.

1. Copy only the model classes of the upstream file: `PatchEmbedding`, `MultiHeadAttention`,
   `ResidualAdd`, `FeedForwardBlock`, `TransformerEncoderBlock`, `TransformerEncoder`,
   `ClassificationHead` and `Conformer`. Import what they use:

   ```python
   import torch
   import torch.nn.functional as F
   from torch import nn, Tensor
   from einops import rearrange
   from einops.layers.torch import Rearrange
   ```

   Do not copy the module-level lines that set `CUDA_VISIBLE_DEVICES` or the cuDNN flags.
   Leave out the torchvision imports, the unused `GELU` helper, the `ExP` class and `main`.

2. Make three changes. They let the network take any number of channels and samples, and
   make it return logits.
   - In `PatchEmbedding`, add an argument `n_channels` and use it in the spatial
     convolution: `nn.Conv2d(40, 40, (n_channels, 1), (1, 1))` in place of `(22, 1)`.
   - In `ClassificationHead`, delete the unused `clshead` branch. Replace
     `nn.Linear(2440, 256)` by `nn.LazyLinear(256)` and the last layer `nn.Linear(32, 4)` by
     `nn.Linear(32, n_classes)`. Make `forward` return only
     `self.fc(x.contiguous().view(x.size(0), -1))`.
   - Replace the `Conformer` class by this wrapper:

     ```python
     class Conformer(nn.Module):
         def __init__(self, n_classes, channels, samples, emb_size=40, depth=6):
             super().__init__()
             self.patch_embedding = PatchEmbedding(emb_size, n_channels=channels)
             self.transformer = TransformerEncoder(depth, emb_size)
             self.classification_head = ClassificationHead(emb_size, n_classes)

         def forward(self, x):
             return self.classification_head(self.transformer(self.patch_embedding(x)))
     ```

All other hyperparameters stay at the upstream values: embedding size 40, six encoder blocks,
10 heads, 25-sample temporal kernel, average pooling of 75 samples with stride 15, and
dropout 0.5 (0.3 before the last layer). Features for head fine-tuning are the inputs of the
last linear layer (32 dimensions).

With these changes the network is identical to the one used for the paper: same parameter
names and shapes, same random initialization for a given seed, same outputs. To check the
adaptation, run the lines below. For 4 classes, 22 channels and 448 samples they should
print 410692.

```python
import torch
from calibration_budget.backbones.external import build_conformer
m = build_conformer(4, 22, 448)
m(torch.zeros(2, 1, 22, 448))          # materializes the LazyLinear layer
print(sum(p.numel() for p in m.parameters()))
```

Song, Zheng, Liu and Gao, "EEG Conformer: Convolutional transformer for EEG decoding and
visualization", *IEEE Trans. Neural Syst. Rehabil. Eng.* 31, 710–719 (2023).
