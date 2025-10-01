import speech_recognition as sr
import pyttsx3
import pywhatkit

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("listening.....")
        r.pause_threshold = 1
        r.energy_threshold = 100
        audio = r.listen(source, 0, 4)

    try:
        print("Understanding...")
        query = r.recognize_google(audio, language="en-in")
        print(f"You said: {query}\n")
    except Exception as e:
        print("Say that again")
        return "None"
    return query

query = takeCommand().lower()

engine = pyttsx3.init("sapi5")
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[1].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def searchGoogle(query):
    if "google" in query:
        import wikipedia as googleScrap
        query = query.replace("buddy", "")
        query = query.replace("search on google", "")
        query = query.replace("google search", "")
        query = query.replace("search", "")
        query = query.replace("on google", "")
        query = query.replace("google", "")
        speak("This is what I found on Google")
        
        try:
            pywhatkit.search(query)
            result = googleScrap.summary(query, 2)
            speak(result)
            
        except Exception as e:
            speak("No speakable output available")
            
    else:
        speak("Something went wrong")
        
def searchYoutube(query):
    if "youtube" in query:
        import webbrowser
        query = query.replace("buddy", "")
        query = query.replace("youtube search", "")
        query = query.replace("search on youtube", "")
        query = query.replace("search", "")
        query = query.replace("on youtube", "")
        query = query.replace("youtube", "")
        speak("This is what I found on your search")
        web = "https://www.youtube.com/results?search_query=" + query
        webbrowser.open(web)
        pywhatkit.playonyt(query)
        speak("Done, sir")
    else:
        speak("Something went wrong")
        
def searchWikipedia(query):
    if "wikipedia" in query:
        import wikipedia
        query = query.replace("buddy", "")
        query = query.replace("wikipedia search", "")
        query = query.replace("search on wikipedia", "")
        query = query.replace("search", "")
        query = query.replace("on wikipedia", "")
        query = query.replace("wikipedia", "")
        speak("Searching on Wikipedia...")
        try:
            result = wikipedia.summary(query, sentences=1)
            speak("According to Wikipedia")
            print(result)
            speak(result)
        except wikipedia.exceptions.DisambiguationError as e:
            speak("The query is ambiguous, please specify further.")
        except wikipedia.exceptions.PageError as e:
            speak("Sorry, I couldn't find any page for the given query.")
        except Exception as e:
            speak("Something went wrong while searching Wikipedia")
    else:
        speak("Something went wrong")