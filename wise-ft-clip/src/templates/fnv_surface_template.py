
fnv_surface_template = [
    # lambda c: f"a photo of a {c} from the category of fruit and vegetables on the surface of metal or glass.", # 0.5789 zeroshot with classes_in_old_dataset_before_rename.txt
    # lambda c: f"a photo of {c} from the category of fruit and vegetables on the surface of metal.", # 0.5805 zeroshot with classes_in_old_dataset_before_rename.txt
    lambda c: f"a photo of {c} from the category of fruit and vegetables on metal surface.", # 0.6120 zeroshot with classes_in_old_dataset_before_rename.txt
]