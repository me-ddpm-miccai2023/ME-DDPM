
# Introduction
Code base for MICCAI submission #1725: 
> "Motion Exploiting Diffusion Models for Accelerated Dynamic MRI Reconstruction"


# Dependencies
Creating the Conda virtual environment:
```bash
conda create --name pt1.9 python pytorch==1.9.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda activate pt1.9
conda install matplotlib imageio ipython requests scipy pillow jupyter pandas tabulate mpi4py
conda install -c conda-forge nibabel scikit-image moviepy scikit-learn tqdm pytorch-lightning
pip install blobfile, tqdm, nibabel, scikit-image
```