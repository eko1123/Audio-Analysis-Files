import variables
import dbconnect

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import ast
from datetime import datetime
import audio_converter
from uuid import uuid4




audio_oddities =[] # an object that will contain all the potential issues with an audio file.

variables.initialize()
cursor, cnxn = dbconnect.setup_connection()

    
def speech_speed(OA_AWS_FULL_REPORT,OA_TRANSCRIPT):
    #results = cursor.execute('SELECT OA_ID, PRSN_ANID, OA_DT, OA_RLTD_VID_ID, OA_TRANSCRIPT, OA_AWS_FULL_REPORT FROM OKAYA_AUDIO WHERE OA_LNKNM = ?', OA_LNKNM)
    #list_results = results.fetchall()[0]
    
    #items = list_results[5]
    listitems = ast.literal_eval(OA_AWS_FULL_REPORT)
    
    transcript = OA_TRANSCRIPT.split()
    running = True
    i = -1
    while running:
        if listitems[i]['type'] == 'pronunciation':
            endtime = float(listitems[i]['end_time'])
            running = False
        i -= 1
    speed = len(transcript)/(endtime/60)
    return speed
    

def pauses(OA_AWS_FULL_REPORT):
    #results = cursor.execute('SELECT OA_ID, PRSN_ANID, OA_DT, OA_RLTD_VID_ID, OA_TRANSCRIPT, OA_AWS_FULL_REPORT FROM OKAYA_AUDIO WHERE OA_LNKNM = ?', OA_LNKNM)
    #list_results = results.fetchall()[0]
    items = ast.literal_eval(OA_AWS_FULL_REPORT)
    times = []
    pauses = []
    for i in items:
        if i['type'] != 'punctuation':
            times.append((float(i['start_time']), float(i['end_time'])))
    for t in range(len(times)-1):
        pauselen = times[t+1][0] - times[t][1]
        if pauselen > .5:
            pauses.append({'time': times[t][1], 'length': round(pauselen, 2)})
    
    return pauses

def visualize(data, sample_rate):
    #job_name = datetime.now().strftime("%Y%m-%d%H-%M%S-") + str(uuid4())
    #job_name = job_name.replace("-", "") 
    #path = 'https://smarttecresearch.s3.us-west-2.amazonaws.com/' + OA_LNKNM
    #filename = audio_converter.convert_video_to_audio(path,job_name)
    #filename = r"C:\Users\elvin\OneDrive\Documents\Okaya\Data\BergenStudy\1_0_10_5_3_1_7_40_503.MOV"
    #data, sample_rate = librosa.load(OA_LNKNM, sr=44100)
    librosa.display.waveshow(data, sr=sample_rate)
    plt.show()

def melspec(data, sample_rate):
    #job_name = datetime.now().strftime("%Y%m-%d%H-%M%S-") + str(uuid4())
    #job_name = job_name.replace("-", "") 
    #path = 'https://smarttecresearch.s3.us-west-2.amazonaws.com/' + OA_LNKNM
    #filename = audio_converter.convert_video_to_audio(path,job_name)
    #filename = r"C:\Users\elvin\OneDrive\Documents\Okaya\Data\BergenStudy\1_0_10_5_3_1_7_40_503.MOV"
    #data, sample_rate = librosa.load(filename, sr=44100)
    fourier = librosa.stft(data)
    fourierdb = librosa.amplitude_to_db(abs(fourier))

    #plt.figure(figsize=(15,5))
    #librosa.display.specshow(fourierdb, sr=sample_rate, x_axis='time', y_axis='hz')
    #plt.colorbar()
    #plt.show()

def F0(OA_LNKNM):
    job_name = datetime.now().strftime("%Y%m-%d%H-%M%S-") + str(uuid4())
    job_name = job_name.replace("-", "") 
    path = 'https://smarttecresearch.s3.us-west-2.amazonaws.com/' + OA_LNKNM
    #filename = audio_converter.convert_video_to_audio(path,job_name)
    filename = r"C:\Users\elvin\OneDrive\Documents\Okaya\Data\BergenStudy\1_0_10_5_3_1_7_40_503.MOV"
    
    data, sample_rate = librosa.load(filename, sr=44100)
    f0 = librosa.yin(data, sr = sample_rate, fmin = librosa.note_to_hz('C2'), fmax= librosa.note_to_hz('C7'))

    times = librosa.times_like(f0)
    fig, ax = plt.subplots()
    ax.set(title='YIN fundamental frequency estimation')
    ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    ax.legend(loc='upper right')
    plt.show()

F0('1_1_8_1_4_1_7_40.MOV')


#get the file to process and get the audio.
'''
results = cursor.execute('SELECT TOP 5 OA_ID, PRSN_ANID, OA_DT, OA_RLTD_VID_ID, OA_TRANSCRIPT, OA_AWS_FULL_REPORT,OA_LNKNM FROM OKAYA_AUDIO WHERE OA_LNKNM is not null')
for row in results:
    if row[6] is not None:
        OA_LNKNM = 'https://smarttecresearch.s3.us-west-2.amazonaws.com/' + row[6]
        OA_TRANSCRIPT = row[4]
        OA_AWS_FULL_REPORT= row[5]
        audio_file = audio_converter.convert_video_to_audio(OA_LNKNM,"")
        #cleaning up the method call so that we do not do the same librosa work each time.
        data, sample_rate = librosa.load(audio_file, sr=44100)
        visualize (data, sample_rate)
        melspec (data, sample_rate)
        this_pause = pauses(OA_AWS_FULL_REPORT)
        print (this_pause)
        this_speed = speech_speed(OA_AWS_FULL_REPORT,OA_TRANSCRIPT)
        print (this_speed)
'''
#visualize('1_1_8_1_4_1_7_40.MOV')
