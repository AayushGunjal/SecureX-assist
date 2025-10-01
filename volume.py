from pynput.keyboard import Key, Controller
from time import sleep

Keyboard = Controller()

def volumeup():
    for i in range (5):
        Keyboard.press(Key.media_volume_up)
        Keyboard.release(Key.media_volume_up)
        sleep(0.1)

def volumedown():
     for i in range (5):
        Keyboard.press(Key.media_volume_down)
        Keyboard.release(Key.media_volume_down)
        sleep(0.1)