# CSE 40982 Project
#### Authors: Jack Flaherty & Jeremy Stevens

This repository will hold the files for the Movie Recommendation Project. 
The binary files for the program are stored on a [Google Drive folder](https://drive.google.com/drive/folders/1UDsz_Nnbgf1OWulKV8fJPh9CxudsZFRJ?usp=sharing). 
Add the .pkl files to the base root of the repository in order to execute any code. 

## How to run
In order to run our dialogue system, you would first need to get all of the required libraries using ```pip install -r requirements.txt```. 
Then run ```python3 main.py``` to execute the dialgoue system. 

## Our Plan
The idea of our dialogue system was to create a LLM model that is fine-tuned to act as the dialogue system for recommending movies. 
We got all of our data from IMDB, found at this [link](https://developer.imdb.com/non-commercial-datasets/). 
We then used python scripts to convert the .tsv files into binary files using pickle and pandas. 
After creating pandas dataframes for the movie data, we generated 3 vectors for a subset of the 10,000 most popular movies in the dataset. 
These 3 vectors would represent: 
1. The movie name and characteristics of the movie itself, such as year of release and genres of the movie etc.
2. The actors and actresses present in the movie. 
3. Possibly the director(s) who directed the movie. 

Once we created these vectors, we made a simple dialogue system with python in order to generate a ton of samples for finetuning a model. 
We generated 10,000 sample conversations by selecting a random movie and answering the systems questions in order to best fit the movie's data. 
After generating the sample conversations, we then tokenized each conversation for each question asked and each answer given. 

## Finetuning a model
Once we had all of our data generated, we started our attempt to fine-tune models.
We first started by trying to fine-tune GPT-J, but we were having difficulties with getting GPT-J to run. 
We tried getting GPT-J to run on Google-Colab, but the base model could not fit into the systems memory before being transferred over to the GPU's memory with the free version of Google-Colab. 
We tried implementing the 8bit quantization to decrease the size of the model when loaded in memory, but it did not help. 
We then tried finetuning the model on Paperspace, but also ran into issues with the libraries needed for 8bit quantization. 

After the issues with GPT-J, we then switched over the LLaMA-LoRA using this [repository](https://betterprogramming.pub/fine-tuning-gpt-j-6b-on-google-colab-or-equivalent-desktop-or-server-gpu-b6dc849cb205).
In this repository, we ran the Google-Colab that was provided so that we could fine-tune our model. 
Jeremy was unable to run this at all, while Jack was able to run this on his computer and produced this [model]().

While Jack was training with LLaMA-LoRA, Jeremy followed this [tutorial](https://www.mlexpert.io/machine-learning/tutorials/alpaca-fine-tuning) with Alpaca-LoRA to see if he would be able to fine-tune the model. 
Using this tutorial, Jeremy created this [Google-Colab notebook](https://colab.research.google.com/drive/1a1azF8bSw3GVSqYl-diBmmde1-H1kPiQ?usp=sharing) to fine-tune the data. 
With this, Jeremy created 3 HuggingFace models that held the weights for an Alpaca-LoRA model. ([model 1](https://huggingface.co/jsteve22/movie-weights), [model 2](https://huggingface.co/jsteve22/movie-weights1000), [model 3](https://huggingface.co/jsteve22/movie-weights100))
After generating the models, Jeremy would reset the Google-Colab notebook and use the last 3 cells in order to create an interface to test the models. 
However, the models often did not perform well if at all, even when provided direct sentences from the trained data the model would not reproduce the desired output. 

## Data for fine-tuning
In order to fine-tune the model, the model was given a prompt that it needed to complete while being subjected to a lot of examples. 
The data that was fed into the model was formatted in a tuple of instruction, input, and output. 
The instruction would inform the LLM how to respond, while the input would give context, and the output is the expected output.
For each of our questions, we prompted the LLM to ask a question in the same vain.
For example, we would instruct the model to "ask the user if they want to watch an old movie" and then the output we expected was along the lines of "do you want to watch an old movie?"
We did not give any input as most of the questions asked to the user did not required any previous knowledge of the conversation. 
Only when the system would make recommendations, we would give the dialogue history so that it could train to see what suggestions where made knowing what the inputs to the conversation were. 
We did this so that the hopefully the LLM would be trained to know what kind of movies to recommend with a user's given preferences. 
However, our models did not live up to our expectations and did not provide the expected output. 

## What we learned and how we would move forward
Through this project we learned a lot about how to find a data source and convert the data into a dataframe. 
We then also learned how to use the dataframe to generate vectors in order to make a recommendation system using cosine similarity. 
After creating a simple dialogue system, we then were able to generate a lot of examples that could be used for fine-tuning a model. 
We then learned how to finetune a model with the generated data, even if we were not able to get the desired results. 

In the future, if we were to continue working on this program, we would edit our generated data in a way such that it would be easier for the LLM to understand. 
We would also spend more time finetuning and testing the different parameters for finetuning the model. 
We would also invest in a GPU or GPU time so that we could reliably fine-tune a model instead of relying on the free GPUs provided by Google-Colab, since there were multiple times the notebook session would close or terminate while it was fine-tuning. 
