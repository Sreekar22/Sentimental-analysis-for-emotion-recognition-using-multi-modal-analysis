# Sentimental-analysis-for-emotion-recognition-using-multi-modal-analysis
A end to end model for emotion recognition with multiple modalities of data.
In this proejct, 
We propose a novel fully end-to-end model that seamlessly integrates both phases of feature extraction and optimization. By jointly optimizing these steps, we aim to improve the overall performance of multimodal affective computing tasks. To facilitate end-to-end training, we reorganize the available dataset(IEMOCAP) to ensure better alignment between the modalities and the target task.

Furthermore, to address the computational burden that the end-to-end model might introduce, we introduce a sparse cross-modal attention technique for feature extraction. This technique allows us to focus on the most relevant and informative elements across modalities while reducing the computational complexity.

In summary, our work introduces a comprehensive solution for multimodal affective computing tasks by combining feature extraction and end-to-end learning in an integrated manner. We also enhance the training process by rearranging datasets and implement a sparse cross-modal attention technique to make the model more efficient. This approach has the potential to improve the performance and scalability of emotion recognition and other multimodal tasks.

## Paper link and dataset
Link for the research paper: https://arxiv.org/pdf/2103.09666.pdf

To know more about the IEMOCAP dataset, click here: [IEMOCAP](https://sail.usc.edu/iemocap/)

To download the dataset, you can fill the form below for the owner to provide with the database access:

[form link](https://github.com/GANeelima/Multimodal-Person-Identification-)
## Environment:
+ Python 3.7

+ PyTorch 1.6.0

+ torchaudio 0.6.0

+ torchvision 0.7.0

+ transformers 3.4.0 (huggingface)

For above environments use pip or conda installations procedure.

+ sparseconvnet

+ facenet-pytorch 2.3.0

+ sentencepiece

for sparseconvnet, clone the repository:

```
!git clone https://github.com/facebookresearch/SparseConvNet
```

followed by 

```
!bash develop.sh
```

Creating venv in colab:

```
!pip install -q condacolab
import condacolab
condacolab.install()
```

### Train the MME2E
```
python main.py -lr=5e-5 -ep=40 -mod=tav -bs=8 --img-interval=500 --early-stop=6 --loss=bce --cuda=3 --model=mme2e --num-persons=10 --trans-dim=64 --trans-nlayers=4 --trans-nheads=4 --text-lr-factor=10 --text-model-size=base --text-max-len=100

```

#### Train the sparse MME2E
```
python main.py -lr=5e-5 -ep=40 -mod=tav -bs=2 --img-interval=500 --early-stop=6 --loss=bce --cuda=3 --model=mme2e_sparse --num-persons=10 --trans-dim=64 --trans-nlayers=4 --trans-nheads=4 --text-lr-factor=10 -st=0.8 --text-model-size=base --text-max-len=100
```
