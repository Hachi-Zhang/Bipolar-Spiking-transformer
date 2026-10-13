## Environment and Datasets
For environment setup, dataset downloads, and data formats, we follow SDTrack exactly. Please refer to the [SDTrack repository](https://github.com/YmShan/SDTrack) and follow its instructions to set up the environment and prepare the datasets.



## Download the pre-trained weights on ImageNet-1K.
1. Download [Pretrain Model](https://drive.google.com/file/d/1OcXHCnibEv9F40gw5VwGO90adtE6E0Ik/view?usp=sharing).
2. Create the directory SDTrack/**pretrained_models** and place the two downloaded weight files into this directory.

## Modify the settings required for training and testing.
1. The training path configuration file is located at `SDTrack/lib/train/admin/local.py`.
2. The testing path configuration file is located at `SDTrack/lib/test/evaluation/local.py`.

## Training
```
# FE108
bash train_tiny_fe108.sh
# VisEvent
bash train_tiny_visevent.sh
# FELT
bash train_tiny_felt.sh
```

## Test
```
# FE108
bash test_tiny_fe108.sh
# VisEvent
bash test_tiny_visevent.sh
# FELT
bash test_tiny_felt.sh
```

## Before Running SDTrack-Tiny/Base On FELT Dataset.
1. **Transform Configuration Adjustment** Modify the data augmentation settings in the '/SDTrack/lib/train/base_functions.py' path to:
```
transform_train = tfm.Transform(tfm.ToTensor(), 
                               tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
                                )
transform_val = tfm.Transform(tfm.ToTensor(),
                              tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
                              )
```
2. **ToTensor Class Modification** Revise the transform_image method in the ToTensor class located at '/SDTrack/lib/train/data/transforms.py' to:
```
def transform_image(self, image):
    # handle numpy array
    if image.ndim == 2:
        image = image[:, :, None]

    image = torch.from_numpy(image.transpose((2, 0, 1)))
    # backward compatibility
    if isinstance(image, torch.ByteTensor):
        return image.float().div(255)
    else:
        return image
```
3. **Testing Phase Modification** Alter the Preprocessor class in '/SDTrack/lib/test/tracker/data_utils.py' to:
```
class Preprocessor(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        # Deal with the attention mask
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).cuda().unsqueeze(dim=0)  # (1,H,W)
        return NestedTensor(img_tensor_norm, amask_tensor)
```


## Evaluation
1. Download the MATLAB script for evaluation([FE108](https://drive.google.com/file/d/1sf2pSOAYAcsWbnxC2brsG_QnzvMP0rrJ/view?usp=sharing), [FELT](https://drive.google.com/file/d/1CqYK8q2mysR2FGZx9GJWY6lzbXSiUXxF/view?usp=sharing) and [VisEvent](https://drive.google.com/file/d/1QgZEMbnJifpSFjnUJIVlL9D3_AeOZWYf/view?usp=sharing)). The evaluation scripts for FELT and VisEvent were provided by [Xiao Wang](https://github.com/wangxiao5791509), while the evaluation script for FE108 was modified by us.
2. For the three datasets, before evaluation, the test results (including multiple .txt files) need to be copied to the `tracking_results` folder in the corresponding directory. Additionally, the `utils/config_tracker.m` file in the respective folder should be modified. Finally, run the corresponding MATLAB script to generate the evaluation results. It is important to note that before testing AUC, you need to set `ranking_type = AUC`, and before testing PR, you need to set `ranking_type = threshold`. For the FELT dataset, before moving the test results to the `tracking_results` folder, you first need to move the test results to the `processing_data` directory and run `processing_1.py` and `processing_2.py` to correct their format.



## Get the training and inference results.



# Acknowledgments
We thank the [SDTrack](https://github.com/YmShan/SDTrack) project for providing a codebase that enabled us to quickly implement our ideas.