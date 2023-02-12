spots_norm_mean = [0.7630392, 0.5456477, 0.57004845]
spots_norm_std = [0.1409286, 0.15261266, 0.16997074]

# weight calculated from training set, to tackle class imbalance
training_weights = [
    0.36283185840707965,
    0.2222222222222222,
    0.10594315245478036,
    1.0,
    0.019052044609665426,
    0.8367346938775511,
    0.09820359281437126,
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
