# Unsupervised Contrastive Audio-Sheet Music Retrieval
This repository contains the code for the paper [Self-Supervised Contrastive Learning for Robust Audio–Sheet Music Retrieval Systems](https://arxiv.org/abs/2309.12134) presented at [ACM MMSys 2023](https://2023.acmmmsys.org/).

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/luisfvc/ucasr.git
   cd ucasr
   ```

2. Install dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate ucasr
   ```
## Citation
If you use this code in your research, please cite our paper:
```bibtex
@inproceedings{CarvalhoWW23_SelfSupLearning_ASR_ACM-MMSys,
  title = {Self-Supervised Contrastive Learning for Robust Audio–Sheet Music Retrieval Systems},
  author = {Lu{\'i}s Carvalho and Tobias Wash{\"u}ttl and Gerhard Widmer},
  year = 2023,
  pages = {239--248},
  booktitle = {Proceedings of the {ACM} International Conference on Multimedia Systems ({ACM-MMSys})},
  address = {Vancouver, Canada},
  doi = {10.1145/3587819.3590968}
}
```
## Related Work

This work builds upon our previous research in audio-sheet music retrieval:
- [Learning audio–sheet music correspondences (TISMIR 2018)](https://github.com/CPJKU/audio_sheet_retrieval)
- [Attention models for tempo-invariant retrieval (ISMIR 2019)](https://github.com/CPJKU/audio_sheet_retrieval/tree/ismir-2019)
- [Exploiting temporal dependencies (EUSIPCO 2021)](https://github.com/CPJKU/audio_sheet_retrieval/tree/eusipco-2021)

## Acknowledgements

This work is supported by the European Research Council (ERC) under the EU’s Horizon 2020 research and innovation programme,
grant agreement No. 101019375 (“Whither Music?”), and the Federal State of Upper Austria (LIT AI Lab).

## Contact
For questions or issues, you can contact me here or via [email](mailto:luisfeliperj90@gmail.com).
