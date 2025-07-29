import os
import json

import torch
import numpy as np

from src.models import utils
from src.datasets.common import get_dataloader, maybe_dictionarize

import src.datasets as datasets
from src.models.utils import LabelSmoothing
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
from PIL import Image
from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
import torch.nn.functional as F

THRESHOLD = 0.1

def _convert_to_rgb(image):
    return image.convert('RGB')

def get_val_preprocess_without_norm():
    return Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            _convert_to_rgb,
            ToTensor(),
        ])


def camptum_visual_single_dataset(image_classifier, dataset, args):
    class_in_all_product_to_idx = {}
    with open(args.classes_list_path, "r") as f:
        for index, line in enumerate(f):
            class_in_all_product_to_idx[line.rstrip()] = index
    idx_to_class_in_all_product = {idx: class_in_all_product for class_in_all_product, idx in class_in_all_product_to_idx.items()}
    if args.freeze_encoder:
        model = image_classifier.classification_head
        input_key = 'features'
        image_enc = image_classifier.image_encoder
    else:
        model = image_classifier
        input_key = 'images'
        image_enc = None

    model.eval()

    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc)
    val_dataset = dataloader.dataset.datasets[0]
    folder_name_to_idx_during_eval = val_dataset.class_to_idx
    idx_to_folder_name_during_eval = {idx: folder_name for folder_name, idx in folder_name_to_idx_during_eval.items()}
    # batched_data = enumerate(dataloader)
    device = args.device
    model.to(device)
    # print(model)
    # exit()


    for index, (img, gt_label_idx) in enumerate(tqdm(val_dataset)):
        input = img.unsqueeze(0).to(device)
        output = model(input)
        scores = F.softmax(output[0])
        prediction_score, pred_label_idx = torch.topk(scores, 4)
        pred_class_name = [idx_to_class_in_all_product[i.item()] for i in pred_label_idx]
        gt_class_name = idx_to_folder_name_during_eval[gt_label_idx]
        gt_score = 0
        if gt_class_name in pred_class_name:
            gt_index = pred_class_name.index(gt_class_name)
            gt_score = prediction_score[gt_index]
            if gt_score.item() > THRESHOLD:  #If < here, then save correct pred images; if > here, then save wrong pred images
                # print(gt_score)
                continue  #skip this image
            else:
                gt_score = gt_score.item()
        
        # if gt_class_name not in pred_class_name:
        #     continue
    # if gt_class_name == pred_class_name:
    # if True:
        img_path = val_dataset.imgs[index][0]
        folder_name = os.path.join(*img_path.split("/")[-5:-1])
        file_name = img_path.split("/")[-1]
        file_name = f"gt_score_{gt_score}_pred_as_{pred_class_name}_{file_name}"
        img_name = os.path.join(folder_name, file_name)
        img = Image.open(img_path)
        transform = image_classifier.val_preprocess
        transformed_img = transform(img)

        transform_without_norm = get_val_preprocess_without_norm()
        transformed_without_norm_img = transform_without_norm(img)
        # save_path = os.path.join("transformed_img_without_norm_loose", args.save.split("/")[-1], img_name)
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # save_image(transformed_without_norm_img, save_path)

        integrated_gradients = IntegratedGradients(model)
        noise_tunnel = NoiseTunnel(integrated_gradients)
        model.zero_grad()
        attributions_ig_nt = noise_tunnel.attribute(input, nt_samples=5, nt_type='smoothgrad_sq', target=pred_label_idx[0])
        # attributions_ig_nt = integrated_gradients.attribute(input, target=pred_label_idx, n_steps=200)
        transformed_img = np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0))
        transformed_without_norm_img = np.transpose(transformed_without_norm_img.squeeze().cpu().detach().numpy(), (1,2,0))
        original_image = img
        attr = np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0))
        norm_attr = viz._normalize_attr(attr, sign='all', outlier_perc=2, reduction_axis=2)
        fig, axs = plt.subplots(1, 2)
        axs[0].set_title(f"GT: {gt_class_name}\nPred: {pred_class_name} ")
        axs[0].imshow(transformed_without_norm_img)
        axs[0].axis('off')
        axs[1].imshow(transformed_img)
        heatmap_img = axs[1].imshow(norm_attr, alpha=0.9, cmap='viridis')
        axs[1].axis('off')

        save_path = os.path.join("camptum_visual_new_articles", args.save.split("/")[-1], img_name)
        print(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        # plt.imsave(os.path.join("transformed_img_without_norm", args.save.split("/")[-1], img_name), transformed_img)
        # transformed_img = Image.fromarray(transformed_img)
        # transformed_img.save(os.path.join("transformed_img_without_norm", args.save.split("/")[-1], img_name))


def camptum_visual(image_classifier, args):
    if args.eval_datasets is None:
        return
    info = vars(args)
    for i, dataset_name in enumerate(args.eval_datasets):
        print('Evaluating on', dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            train_set=args.train_set,
            eval_set=args.eval_set,
            preprocess=image_classifier.val_preprocess,
            classes_list_path=args.classes_list_path,
            classes_list_ignore_during_val_path=args.classes_list_ignore_during_val_path,
            location=args.data_location,
            batch_size=args.batch_size
        )

        camptum_visual_single_dataset(image_classifier, dataset, args)