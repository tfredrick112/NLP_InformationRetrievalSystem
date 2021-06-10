## Information Retrieval System

This file describes the steps to run our code.

The easiest way to run our code is to use the Colab file: https://colab.research.google.com/drive/1t3biVpS3wsB5KA8M-pkMjGj9wJJIMU-7?usp=sharing

This helps to:
- avoid Python/TensorFlow version issues
- avoid the need to download the Large data/GloVe files manually

### Instructions to use the Colab notebook:

To run this, open the Colab file and run all the cells one after the other. The 1st 2 cells may take longer as the large data zip file are loaded into the environment and unzipped. Run the 1st 4 cells. Then follow from step 6 below to see the results (1st cell and 2nd cell mentioned in the below steps refer to the 2 cells below the heading "Start of interactive application" in the Colab file)

### To run on your local machine:

Install these libraries: 
`
pip install python-Levenshtein

pip install wikipedia
`

Download the Large_files folder from: https://drive.google.com/drive/folders/1PEYU8C5BOCWa0NCTyiIaJZZNY3EdaTAK?usp=sharing
Add this folder to the Code folder.

Our code can be run both from the terminal and the Interactive application.

**Command line - example**
`
python main.py -dataset "cranfield\\" -out_folder "Output\\" -segmenter punkt -tokenizer ptb -preprocess -model lsa -k 200
`

**Interactive Application**
1. Interactive Application.ipynb must be opened as a Jupyter notebook
2. Execute the 1st cell, keep the options in the default configuration as they appear
3. Click on Preprocess@runtime
4. Execute the 2nd cell
5. This re-creates all relevant data files depending on the nltk/python versions in your system.
6. Execute the 1st cell (there is no need to click preprocess again)
7. Choose a model option from the dropdown list
8. Execute the 2nd cell to evaluate on the entire Cranfield dataset
8. OR you may click the custom query option to enter a custom query and see the autocomplete and spell correct features. Enter a query with a spelling mistake (say, airodynamics) to see the spell correct feature.
9. Note that clicking a button twice does not reset it to its original state. To reset (i.e, to avoid preprocess or to avoid custom query), the 1st cell should be run again.
