## Integration Plan for Electron App

Once whisper.cpp is working standalone, integrate into the Electron app:

### Renderer Process (ChatContainer.tsx)
1. Use `MediaRecorder` API to capture audio from the microphone
2. Convert audio blob to the correct format (16kHz mono WAV)
3. Send audio data to main process via IPC

### Main Process
1. Receive audio data from renderer
2. Save to a temporary WAV file
3. Spawn whisper.cpp as a subprocess:
   ```bash
   ./whisper.cpp/main -m models/ggml-base.en.bin -f /tmp/audio.wav --output-txt --no-timestamps
   ```
4. Parse the output text
5. Return transcribed text to renderer via IPC

### Renderer Process (continued)
1. Receive transcribed text from main process
2. Insert text at cursor position in textarea

## Useful whisper.cpp CLI Options

```bash
./main -m models/ggml-base.en.bin -f audio.wav \
  --output-txt           # Output plain text
  --no-timestamps        # Don't include timestamps
  --language en          # Force English
  --threads 4            # Number of CPU threads to use
  --print-special false  # Don't print special tokens
```

## Resources

- whisper.cpp GitHub: https://github.com/ggerganov/whisper.cpp
- Original Whisper paper: https://arxiv.org/abs/2212.04356
- Model download scripts: https://github.com/ggerganov/whisper.cpp/tree/master/models
