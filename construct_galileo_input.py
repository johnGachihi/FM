from src.data.dataset import GalileoDataset
from src.data.utils import construct_galileo_input

ds = GalileoDataset("train.h5")
sample = ds[0]  # includes s1, s2, era5, dw, label

masked = construct_galileo_input(
    s1=sample["s1"],
    s2=sample["s2"],
    era5=sample["era5"],
    dw=sample["dw"],
    normalize=True,
)
