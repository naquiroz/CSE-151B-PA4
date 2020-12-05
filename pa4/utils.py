import torch

DEVICE = device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    from tqdm.contrib import tenumerate
except ImportError:
    try:
        from tqdm import tqdm

        def tenumerate(iterable, start=0, **kwargs):
            return enumerate(tqdm(iterable, **kwargs), start=start)
    except ImportError:
        tenumerate = enumerate