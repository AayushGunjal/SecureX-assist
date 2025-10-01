import pyttsx3
import datetime
import os
import time as tm

engine = pyttsx3.init("sapi5")
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[1].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

# Read time from file
with open("Alarmtext.txt", "rt") as extractedtime:
    alarm_time_str = extractedtime.read().strip()
    print(f"Alarm time read from file: {alarm_time_str}")

# Clear the content of the file
with open("Alarmtext.txt", "r+") as deletetime:
    deletetime.truncate(0)

def ring(alarm_time_str):
    try:
        # Parse the time string into a datetime.time object
        Alarmtime = datetime.datetime.strptime(alarm_time_str, "%H:%M").time()
        print(f"Alarm set for: {Alarmtime.strftime('%H:%M')}")
    except ValueError:
        print("Time format error. Please ensure the time is in HH:MM format.")
        return

    while True:
        # Get the current time in HH:MM format
        current_time_str = datetime.datetime.now().strftime("%H:%M")
        # print(f"Current time: {current_time_str} | Alarm time: {Alarmtime.strftime('%H:%M')}")

        if current_time_str == Alarmtime.strftime("%H:%M"):
            speak("Alarm ringing")
            try:
                os.startfile("music.mp3")
            except Exception as e:
                print(f"Error opening the music file: {e}")
            break

        tm.sleep(1)  # Sleep for 1 second to avoid busy-waiting

ring(alarm_time_str)
