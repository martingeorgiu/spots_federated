from typing import Union

import torch

spots_norm_mean = [0.7630392, 0.5456477, 0.57004845]
spots_norm_std = [0.1409286, 0.15261266, 0.16997074]

# weight calculated from training set, to tackle class imbalance
proportional_training_weights = [
    0.36283185840707965,
    0.2222222222222222,
    0.10594315245478036,
    1.0,
    0.019052044609665426,
    0.8367346938775511,
    0.09820359281437126,
]

# weight calculated from training_weights taken to the power of 1/4
reduced_training_weights = [
    0.7761154935321088,
    0.6865890479690392,
    0.5705165179818953,
    1.0,
    0.3715227369341196,
    0.9564162451145308,
    0.5597986466874496,
]

lesion_type_dict = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions ",
    "df": "Dermatofibroma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
    "mel": "Melanoma",
}

lesion_type_id = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "nv",
    "vasc",
    "mel",
]

metadata_file_path = "assets/HAM10000_metadata.csv"


def get_alpha(arg: Union[str, None]) -> Union[torch.FloatTensor, None]:
    if arg == "proportional":
        return torch.FloatTensor(proportional_training_weights)
    if arg == "reduced":
        return torch.FloatTensor(reduced_training_weights)
    if arg is None:
        return None
    raise ValueError("Invalid value for alpha")
