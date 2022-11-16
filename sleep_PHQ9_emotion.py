import variables
import dbconnect
import pdb
import audio_analysis
import audio_amazon_analysis

import error_management

import os
import audio_converter
from audio_prediction import text_prediction
from audio_prediction import voice_prediction

import re

from datetime import datetime
from uuid import uuid4

import numpy as np
import matplotlib as plt

variables.initialize()

transcribe = audio_amazon_analysis.setup_transcribe()

def analysis_from_link(linkURL, transcribe,OA_ID):
    
    job_name = datetime.now().strftime("%Y%m-%d%H-%M%S-") + str(uuid4())
    job_name = job_name.replace("-", "")

    audio_file = audio_converter.convert_video_to_audio(linkURL,job_name)
    s3_audio_file_name = audio_amazon_analysis.upload_audio_to_s3(audio_file)
    print("s3_audio_file_name: " + s3_audio_file_name)

    job_status = audio_amazon_analysis.transcript_conversion(transcribe, s3_audio_file_name, job_name, 'en-us')

    transcript = ""

    transcript_manager = {
        "transcript": "no",
        "length": -1,
        "Sentiment": 'no',
        "Mixed": -1,
        "Negative": -1,
        "Neutral": -1,
        "Positive": -1,
        "Phrases": [],
        "Phrase_Analysis": [],
        "Phrasetimes": [],
        "Vocal_Analysis": [],
        "items": "",
    }

    if job_status:
        jsonreturn = audio_amazon_analysis.get_transcription_text(job_name, transcribe)
        transcript = jsonreturn["transcripts"][0]["transcript"]
        items = jsonreturn['items']
        transcript_manager["length"] = len(transcript.strip().split(" "))
        transcript_manager["transcript"] = transcript
        transcript_manager["items"] = items
        audio_analysis.save_audio_sentiment_analysis(transcript_manager,OA_ID)
        # deal with a null transcript or missing the minimum number of words
        if transcript_manager["length"] > 15 :
            # now we can do some analysis.
            # setting the language code to match Amazon's standards which is just the 2 letter description.
            amazon_language_code = "en"
            transcript_manager = audio_amazon_analysis.get_transcription_sentiment(transcript_manager, amazon_language_code)
        
        #splitting the transcription into phrases to be analyzed
        phrases = re.split('[?.!]', transcript)
        input_phrases = []
        for phrase in phrases:
            if len(phrase) > 8:
                input_phrases.append(phrase)
        
        transcript_manager['Phrases'] = input_phrases
        
        predictionoutput = []
        for i in input_phrases:
            pred = text_prediction(i)
            predictionoutput.append(pred)
        transcript_manager["Phrase_Analysis"] = predictionoutput
        
        #getting timestamps for where the phrases begin and end
        starttimes = []
        endtimes = []
        starttimes.append(items[0]['start_time'])
        for i in range(len(items)):
            if items[i]['type'] == "punctuation":
                if i != (len(items) - 1):
                    phraseend = items[i-1]['end_time']
                    phrasestart = items[i+1]['start_time']
                    endtimes.append(phraseend)
                    starttimes.append(phrasestart)
                else:
                    phraseend = items[i-1]['end_time']
                    endtimes.append(phraseend)
        
        outputtimes = []
        for i in range(len(starttimes)):
            outputtimes.append((float(starttimes[i]), float(endtimes[i])))
        transcript_manager["Phrasetimes"] = outputtimes

        #running prediction on mfccs
        mfccoutput = []
        for i in outputtimes:
            out = voice_prediction(audio_file, i)
            mfccoutput.append(out)
        transcript_manager["Vocal_Analysis"] = mfccoutput

        #We now save the transcript information in the DB.
    # we are done with the audio analysis, we remove the temp audio files from our environment.
    os.remove(audio_file)

    #jobs completed
    return transcript_manager

def emo_counter(lst):
    count = 0
    for pred in lst:
        if pred[0][0][0] > pred[0][0][1]:
            count += 1
        else:
            count -= 1
    if count >= 0:
        emo = 0
    else:
        emo = 1

    return emo

def grapher(arr, title):
    x = []
    y = []
    for coord in arr:
        x.append(coord[0])
        y.append(coord[1])
    x = np.array(x)
    y = np.array(y)

    plt.scatter(x, y)  #BUGGED OUT HERE (something about scatter and MPL)
    plt.title(title)
    plt.show()


cursor,cnxn = dbconnect.setup_connection()
print ("Sleep_PHQ9_Emotion")
results = cursor.execute("SELECT OA_ID, PRSN_ANID, OA_DT, OA_RLTD_VID_ID, OA_LNKNM, OA_REPORTED_SLP_MIN, OA_REPORTED_PHQ9,[OA_AWS_FULL_REPORT],[OA_TRANSCRIPT] FROM OKAYA_AUDIO ORDER BY OA_ID DESC ")
list_results = results.fetchall()
slp_txt_arr = []
slp_aud_arr = []
phq_txt_arr = []
phq_aud_arr = []

# OA_AWS_FULL_REPORT contains the data from the AWS analysis so you can bypass it.,[OA_TRANSCRIPT] is the transcript as standalone
for i in list_results:
    try:
        if i[5]:
            mylink = 'https://smarttecresearch.s3.us-west-2.amazonaws.com/' + i[4]
            manager = analysis_from_link(mylink, transcribe, i[0])
        

            txt_emo = emo_counter(manager['Phrase_Analysis'])
            aud_emo = emo_counter(manager['Vocal_Analysis'])
            slp_txt_arr.append((txt_emo, int(i[5])))
            slp_aud_arr.append((aud_emo, int(i[5])))
        if i[6]:
            mylink = 'https://smarttecresearch.s3.us-west-2.amazonaws.com/' + i[4]
            manager = analysis_from_link(mylink, transcribe,i[0])
        

            txt_emo = emo_counter(manager['Phrase_Analysis'])
            aud_emo = emo_counter(manager['Vocal_Analysis'])
            phq_txt_arr.append((txt_emo, int(i[6])))
            phq_aud_arr.append((aud_emo, int(i[6])))
    except :
        print("Error with: " + str(i[0]))

grapher(slp_txt_arr, "Sleep Values, Text Model")
grapher(slp_aud_arr, "Sleep Values, Audio Model")
grapher(phq_txt_arr, "PHQ9 Values, Text Model")
grapher(phq_aud_arr, "PHQ9 Values, Audio Model")

