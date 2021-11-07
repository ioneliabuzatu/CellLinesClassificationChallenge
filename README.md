### Cell Lines Classifier 
##### Challenge at JKU - AI in Life Science S2021

Classification of 9 unknown cell lines given microscopy images. The classes to distinguish are:
`PC-3, U-251 MG, HeLa, A549, U-2 OS, MCF7, HEK 293, CACO-2 and RT4`.
Each cell consists of 3 seperate images showing different parts (nucleus, microtubules, 
endoplasmic reticulum) of the same cell.

Example of staining different parts of a cell where the rgb image is the result of the 3 seprate channels combined 
together: \
![Stained](assets/example_cell_channels.png)

## Usage of this repo
1. `preprocessing.py` Preprocess data:
    1. uncompress the files and then run the below cmd for the train images to remove leading zeros
    ```
    for FILE in `ls`; do mv $FILE `echo $FILE | sed -e 's:^0*::'`; done
    ```
   2. Change the path in the `config.py` as you need and then run `python preprocessing.py` this will save the rgb images.
3. `train.py` Train model by running `python train.py`
4. **`Inference.py`** Use the saved checkpoint to generate csv with the predicted classes of the testset. 


#### Report [link](https://docs.google.com/document/d/1mPjPGRh9-oD6d7X2MkJ17OJJs3LVaEpY4e1fZGGO_gA/edit?usp=sharing)

# Results
|Model | Accuracies (validation)|
|:---:|:---:|
|AlexNet| 90%
|VGG19| 92%
|VGG19 (bn) | 95% 
|Resnet18| 92%
Resnet34| **96%**
|Resnet50 | **96%**
|ResnetWide50| **96%**
|Resnet101| **96%**
|Resnet152| 95%


Files needed: 
- `images_train.tar`: Training set (#28896 images) containing three 64x64 pixel grayscale images per sample in png format. Each sample has a unique ID.
The three images per sample represent nucleus ("_blue.png"), microtubules ("_red.png") and endoplasmic reticulum ("_yellow.png").
- `images_test.tar`: Test set #20607 images (for public + private leaderboard) in same format as training set.
- `y_train.csv`: ID of the sample and corresponding label (cell line) for the training set.
- `sample_submission.csv`: The format in which the predictions must be submitted.
