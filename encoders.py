import torch
import torchhd
from torchhd import embeddings


class BaseEncoder(torch.nn.Module):
    needs_features = False

    def __init__(self, D, device="cpu"):
        super().__init__()
        self.D = D
        self.device = device

    def item_memory_bits(self):
        raise NotImplementedError


class RecordEncoder(BaseEncoder):
    needs_features = False

    def __init__(self, D, seq_len=1250, n_levels=64, decimate=1, device="cpu"):
        super().__init__(D, device)
        self.n_pos = seq_len // decimate
        self.n_levels = n_levels
        self.decimate = decimate
        self.position = torchhd.random(self.n_pos, D, device=device)
        self.value = torchhd.level(n_levels, D, device=device)

    def encode(self, x):
        x = x.to(self.device)
        if x.dim() == 3:
            x = x[..., 0]
        if self.decimate > 1:
            x = x[:, :: self.decimate][:, : self.n_pos]
        idx = _quantize_per_segment(x, self.n_levels)
        bound = torchhd.bind(self.position, self.value[idx])
        return torchhd.hard_quantize(torchhd.multiset(bound))

    def item_memory_bits(self):
        return (self.n_pos + self.n_levels) * self.D


class NgramEncoder(BaseEncoder):
    """Permutation n-gram over quantised samples: order-sensitive, tiny item memory."""
    needs_features = False

    def __init__(self, D, n=4, n_levels=64, decimate=2, device="cpu"):
        super().__init__(D, device)
        self.n = n
        self.n_levels = n_levels
        self.decimate = decimate
        self.value = torchhd.level(n_levels, D, device=device)

    def encode(self, x):
        x = x.to(self.device)
        if x.dim() == 3:
            x = x[..., 0]
        if self.decimate > 1:
            x = x[:, :: self.decimate]
        idx = _quantize_per_segment(x, self.n_levels)
        hv = self.value[idx]                                  # (N, T, D)
        T = hv.shape[1]
        acc = None
        for k in range(self.n):                               # bind the k-th shifted copy
            part = torch.roll(hv, shifts=-k, dims=1)[:, : T - self.n + 1]
            part = torch.roll(part, shifts=k, dims=2)         # rho^k
            acc = part if acc is None else torchhd.bind(acc, part)
        return torchhd.hard_quantize(torchhd.multiset(acc))

    def item_memory_bits(self):
        return self.n_levels * self.D


class FeatureEncoder(BaseEncoder):
    """Record encoding over handcrafted features: the small, MCU-plausible option."""
    needs_features = True

    def __init__(self, D, n_features=30, n_levels=64, device="cpu"):
        super().__init__(D, device)
        self.n_features = n_features
        self.n_levels = n_levels
        self.position = torchhd.random(n_features, D, device=device)
        self.value = torchhd.level(n_levels, D, device=device)

    def encode(self, f):
        f = f.to(self.device)
        idx = (f.clamp(0, 1) * (self.n_levels - 1)).long()
        bound = torchhd.bind(self.position, self.value[idx])
        return torchhd.hard_quantize(torchhd.multiset(bound))

    def item_memory_bits(self):
        return (self.n_features + self.n_levels) * self.D


class SinusoidEncoder(BaseEncoder):
    """Random Fourier features of the feature vector (RBF kernel in hyperspace)."""
    needs_features = True

    def __init__(self, D, n_features=30, device="cpu"):
        super().__init__(D, device)
        self.n_features = n_features
        self.proj = embeddings.Sinusoid(n_features, D, device=device)

    def encode(self, f):
        return torchhd.hard_quantize(self.proj(f.to(self.device)))

    def item_memory_bits(self):
        return self.n_features * self.D * 8 + self.D * 8        # weights + phase, int8


def _quantize_per_segment(x, n_levels):
    lo = torch.quantile(x, 0.01, dim=1, keepdim=True)
    hi = torch.quantile(x, 0.99, dim=1, keepdim=True)
    xn = ((x - lo) / (hi - lo + 1e-6)).clamp(0, 1)
    return (xn * (n_levels - 1)).long()


def build_encoder(name, D, n_features=30, n_levels=64, device="cpu", **kw):
    if name == "record":
        return RecordEncoder(D, n_levels=n_levels, decimate=kw.get("decimate", 1), device=device)
    if name == "ngram":
        return NgramEncoder(D, n=kw.get("ngram", 4), n_levels=n_levels,
                            decimate=kw.get("decimate", 2), device=device)
    if name == "feature":
        return FeatureEncoder(D, n_features=n_features, n_levels=n_levels, device=device)
    if name == "sinusoid":
        return SinusoidEncoder(D, n_features=n_features, device=device)
    raise ValueError(name)
