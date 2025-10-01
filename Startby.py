import pyttsx3
import datetime

engine = pyttsx3.init("sapi5")
voices = engine.getProperty("voices")
engine.setProperty("voice",voices[1].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()
    
def greetme():
    hour=int(datetime.datetime.now().hour)
    if hour>=0 and hour<=12:
        speak("Good Morning, Aayush")
    elif hour>=12 and hour<=18:
        speak("Good Afternoon, Aayush")
    else :
        speak("Good Night, Aayush")
        
    speak("Please Tell me , How can I help You ?")