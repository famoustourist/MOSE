# Multiplicative Orthogonal Sequential Editing for Language Models

This repo contains the source code for the paper: *Multiplicative Orthogonal Sequential Editing for Language Models*

Our repo is built on [EasyEdit](https://github.com/zjunlp/EasyEdit/tree/main). We thank them for their excellent work. Please follow its instructions for installation.

The configuration files for MOSE are located in the `hparams/OFT` folder. Please select the corresponding YAML file to perform model editing.

## Example
*The following example is adapted from the EasyEdit repository*:
```
from easyeditor import OFTHyperParams
## Loading config from hparams/OFT/gpt2-xl.yaml
hparams = MENDHyperParams.from_hparams('./hparams/OFT/gpt2-xl')

## edit descriptor: prompt that you want to edit
prompts = [
    'What university did Watts Humphrey attend?',
    'Which family does Ramalinaceae belong to',
    'What role does Denny Herzig play in football?'
]
## You can set `ground_truth` to None !!!(or set to original output)
ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender']
## edit target: expected output
target_new = ['University of Michigan', 'Lamiinae', 'winger']

## Construct Language Model Editor
editor = BaseEditor.from_hparams(hparams)

locality_inputs = {
    'neighborhood':{
        'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
        'ground_truth': ['piano', 'basketball', 'Finnish']
    },
    'distracting': {
        'prompt': ['Ray Charles, the violin Hauschka plays the instrument', 'Grant Hill is a professional soccer Magic Johnson is a professional', 'The law in Ikaalinen declares the language Swedish In Loviisa, the language spoken is'],
        'ground_truth': ['piano', 'basketball', 'Finnish']
    }
}

metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    target_new=target_new,
    locality_inputs=locality_inputs,
    sequential_edit=False # True: start continuous editing ✈️
)
## metrics: edit success, rephrase success, locality e.g.
## edited_model: post-edit model
```

## Citation
Please cite our paper if you use MOSE in your work.

Please also cite EasyEdit, as our model editing is based on their framework.

```

@article{zhang2024comprehensive,
  title={A Comprehensive Study of Knowledge Editing for Large Language Models},
  author={Zhang, Ningyu and Yao, Yunzhi and Tian, Bozhong and Wang, Peng and Deng, Shumin and Wang, Mengru and Xi, Zekun and Mao, Shengyu and Zhang, Jintian and Ni, Yuansheng and others},
  journal={arXiv preprint arXiv:2401.01286},
  year={2024}
}

@article{wang2023easyedit,
  title={Easyedit: An easy-to-use knowledge editing framework for large language models},
  author={Wang, Peng and Zhang, Ningyu and Xie, Xin and Yao, Yunzhi and Tian, Bozhong and Wang, Mengru and Xi, Zekun and Cheng, Siyuan and Liu, Kangwei and Zheng, Guozhou and others},
  journal={arXiv preprint arXiv:2308.07269},
  year={2023}
}

@article{yao2023editing,
  title={Editing Large Language Models: Problems, Methods, and Opportunities},
  author={Yao, Yunzhi and Wang, Peng and Tian, Bozhong and Cheng, Siyuan and Li, Zhoubo and Deng, Shumin and Chen, Huajun and Zhang, Ningyu},
  journal={arXiv preprint arXiv:2305.13172},
  year={2023}
}

@article{cheng2023edit,
  title={Can We Edit Multimodal Large Language Models?}, 
  author={Cheng, Siyuan and Tian, Bozhong and Liu, Qingbin and Chen, Xi and Wang, Yongheng and Chen, Huajun and Zhang, Ningyu},
  journal={arXiv preprint arXiv:2310.08475},
  year={2023}
}

@article{mao2023editing,
  title={Editing personality for llms},
  author={Mao, Shengyu and Zhang, Ningyu and Wang, Xiaohan and Wang, Mengru and Yao, Yunzhi and Jiang, Yong and Xie, Pengjun and Huang, Fei and Chen, Huajun},
  journal={arXiv preprint arXiv:2310.02168},
  year={2023}
}

@article{wang2024wise,
  title={WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models},
  author={Wang, Peng and Li, Zexi and Zhang, Ningyu and Xu, Ziwen and Yao, Yunzhi and Jiang, Yong and Xie, Pengjun and Huang, Fei and Chen, Huajun},
  journal={arXiv preprint arXiv:2405.14768},
  year={2024}
}
```