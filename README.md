# music-genre-classification
A music genre classification task for the purposes of the course Pattern Recognition and Machine learning for the undergraduate program in Informatics at NKUA.

The task at hand is the classification of 1 second audio segments in one of the following four genres of music:

1. Classical
2. Hiphop
3. Hard-rock
4. Jazz

The MFFCs and the Mel-spectrograms of the audio segments can be downloaded from <a href="https://drive.google.com/drive/folders/1C8NNqWJspGxMpuSYv9NbTT6T0bPk3z5k">here</a>. These features will be used to train two types of Neural Networks: 1) A Feed Forward Neural Network using the vector representation of the audio provided by the MFCCs and 2) a Convolutional Neural Network using the 2D representation of the audio segments provided by the mel-spectrograms. At the end, we test the model into new unseen samples by downloading songs from YouTube.

To fully reproduce the code in `main.ipynb` you'll need to:

1. Clone this repo by using 

```bash 
git clone https://github.com/ChrisNick92/music-genre-classification.git 

```
2. Go to directory of the repo
```bash
cd music-genre-classification/
```
2. Unzip the data (download from <a href="https://drive.google.com/drive/folders/1C8NNqWJspGxMpuSYv9NbTT6T0bPk3z5k">here</a>) inside the data folder.
3. Create a virtual environment with a python version of 3.9.16.
4. Run 

```bash 
pip install -r requirements.txt

```

5. To download the videos from YouTube (Inference part) we use `yt-dlp`. To install run
```bash
python3 -m pip install -U yt-dlp
```