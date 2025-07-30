
fnv_template = [
    lambda c: f"a photo of a {c} from the category of fruit and vegetables.", # 0.6722 with classes_in_old_dataset_before_rename_step_by_step_tryout.txt, 0.6511 zeroshot with classes_in_old_dataset_before_rename.txt
    # lambda c: f"a photo of a {c} from the category of fruit and vegetables at a self-checkout in a supermarket." # 0.6323 zeroshot with classes_in_old_dataset_before_rename.txt
    # lambda c: f"a photo of a {c} from the category of fruit and vegetables." # 0.6436 zeroshot with classes_in_old_dataset_renamed.txt
    # lambda c: f"a photo of {c} from the category of fruit and vegetables." # 0.6226 zeroshot with classes_in_old_dataset_plural.txt
    # lambda c: f"a picture of a {c} from the category of fruit and vegetables."
    # lambda c: f"a photo of a {c} in the category of fruit and vegetables."
]