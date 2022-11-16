import re
import onnx
import onnxruntime as rt
import numpy as np
import librosa
import math

import variables

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences


#download stopwords
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')


def text_prediction(mytext, vocabsize=1000, sentencelen=85):

    vocab_size = vocabsize
    len_sentence = sentencelen

    #text preprocessing 
    newtext = re.sub("[^a-zA-Z]", " ", mytext)  #removes non-alphabetic characters          
    newtext = newtext.lower() #makes all characters lowercase
    newtext = newtext.split() #splits string into list       
    newtext = [stemmer.stem(word) for word in newtext if word not in stopwords] #stems each word
    newtext = " ".join(newtext) #joins list into string again
    
    one_hot_word = [one_hot(input_text=newtext, n=vocab_size)] #converts the text into integers

    #this normalizes the length of the vector to be inputted according to len_sentence
    padded_vector = pad_sequences(sequences=one_hot_word, maxlen=len_sentence, padding="pre")
    #padding="pre" means that the vector is populated at the end first, and zeroes are filled at the beginning

    input = np.array(padded_vector) #convert to numpy array

    #load onnx model
    filename = variables.TEXT_MODEL_FILENAME
    
    onnx_model = onnx.load(filename)
    onnx.checker.check_model(filename)

    #create onnx runtime session
    sess = rt.InferenceSession(filename)

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    #compute prediction
    probabilities = sess.run([label_name], {input_name: input.astype(np.float32)})

    return probabilities

def test_text():
    mytext = "I am having a wonderful day with a lot of fun and exciting things planned"
    prediction = text_prediction(mytext)
    assert prediction[0][0][0] > prediction[0][0][1]

def voice_prediction(audiofile, timestamps, length=216000, samplerate=44000, res_type='kaiser_fast'):
    #getting the phrase samples to be analyzed from timestamps
    sampled = librosa.load(audiofile, res_type, sr=samplerate)
    startframe = librosa.time_to_samples(timestamps[0], sr=samplerate)
    endframe = librosa.time_to_samples(timestamps[1], sr=samplerate)
    window = sampled[startframe.item():endframe.item()]

    #normalizing length of input
    if window.shape[0] > length:
        newwindow = window[:length]
    elif window.shape[0] < length:
        newwindow = np.pad(window,math.ceil((length-window.shape[0])/2), mode='median')
    else:
        newwindow = window

    #calculating mfccs
    mfcc = librosa.feature.mfcc(y=newwindow, sr=44000, n_mfcc=20)
    mfcc = mfcc.T
    
    #load onnx model
    filename = variables.AUDIO_MODEL_FILENAME

    onnx_model = onnx.load(filename)
    onnx.checker.check_model(filename)

    #create onnx runtime session
    sess = rt.InferenceSession(filename)

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    #compute prediction
    probabilities = sess.run([label_name], {input_name: mfcc.astype(np.float32)})

    return probabilities
