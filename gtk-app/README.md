# Lingo GTK: Voice Typer - GTK4 Desktop App 🎤

A lightweight Linux desktop application that provides system-wide voice-to-text input. Speak naturally and your words appear wherever your cursor is focused - in any application!

*Warning: This application has only been tested on Ubuntu Linux.*

![](screenshot.png)

## How It Works

1. **Select Microphone** - select your device for your mic
2. **Click Microphone Checkbox** - when the checkbox is checked aything you speak will be typed into wherever your edit cursor is, in any application system wide.  
3. **Speak naturally** - after 1 second of silence, your speech is transcribed
4. **Text is typed** wherever your cursor is focused

## Whisper Setup (from project root)

You only need to run this once.

```bash
cd /path/to/lingo2
./setup-whisper.sh
```

## Running the GTK App

```bash
cd gtk-app
./run.sh
```
The first run will create a virtual environment and install Python dependencies.

## Troubleshooting

Troubleshooting notes are [here](TROUBLESHOOTING.md)

## Tips

This application works best in a quiet room, with the threshold setting (on the GUI) set so low that it will even pick up mouse clicks and keyboard entry noises from time to time. The background of the app will turn green whenever it has detected something on your microphone, and it will successfully ignore things that aren't speech. 

However if you set the threshold too high, the input can sometimes miss chunks of your speech, so if you run into any problems at all, the first thing to try is to lower the threshold value. On a high quality microphone in a quiet room an ideal threshold may be as low as "0.001" but if you're in a more noisy environment, you might need to set the threshold up to "0.02" or even higher.

Whenever the microphone is detecting sound the background will turn green as a visual cue, however, if you see the application continuing to stay green constantly, that means you've got the threshold set too low. A common way this can happen is when you adjust the microphone threshold to what you think is a perfect value in a perfectly quiet room with total silence, and eventually perhaps your air conditioning system turns on creating some background noise, a low enough threshold setting can cause that to trigger the microphone continuously, which makes the application unable to work. So in general, if you see the green background continuously, then you need to raise the threshold setting.  

However, it's perfectly fine and probably preferable to keep the threshold set so low that random  sounds like setting a cup down on a desk or whatever, can trigger the microphone (indicated by the green background), because it will successfully ignore things that are not detected as speech.

**Voice Typer** - Speak anywhere, type everywhere! 🎙️✨
