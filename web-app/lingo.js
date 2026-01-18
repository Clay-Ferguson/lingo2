// ============================================================================
// Lingo - Speech-to-Text with Whisper.cpp + Text-to-Speech
// ============================================================================

// DOM Elements
const readBtn = document.getElementById("readBtn");
const stopBtn = document.getElementById("stopBtn");
const pauseBtn = document.getElementById("pauseBtn");
const resumeBtn = document.getElementById("resumeBtn");
const micBtn = document.getElementById("micBtn");
const copyBtn = document.getElementById("copyBtn");
const clearBtn = document.getElementById("clearBtn");
const voiceSelect = document.getElementById("voiceSelect");
const rateSelect = document.getElementById("rateSelect");
const textArea = document.getElementById("text");
const status = document.getElementById("status");

// Storage keys
const STORAGE_VOICE_KEY = "tts_selected_voice_name_v1";
const STORAGE_RATE_KEY = "tts_selected_rate_v1";
const STORAGE_TEXT_KEY = "tts_text_buffer_v1";

// ============================================================================
// TTS (Text-to-Speech) State
// ============================================================================
let currentUtterance = null;
let isTTSSpeaking = false;
let isTTSPaused = false;
let paragraphQueue = [];

// ============================================================================
// STT (Speech-to-Text) State - Whisper-based
// ============================================================================
let isRecording = false;
let mediaRecorder = null;
let audioContext = null;
let analyser = null;
let audioChunks = [];
let silenceStartTime = null;
let animationFrameId = null;
let audioStream = null;

// Configuration for silence detection
const SILENCE_THRESHOLD = 0.01;  // Audio level below this is considered silence
const SILENCE_DURATION_MS = 1000;  // 1 second of silence triggers transcription
const MIN_AUDIO_DURATION_MS = 500;  // Minimum audio length to bother transcribing

// ============================================================================
// Utility Functions
// ============================================================================

function supportsSpeech() {
  return "speechSynthesis" in window && typeof SpeechSynthesisUtterance !== "undefined";
}

function setStatus(text) {
  status.textContent = text || "";
}

function updateReadButton() {
  if (isTTSSpeaking) {
    readBtn.style.display = "none";
    stopBtn.style.display = "";
    if (isTTSPaused) {
      pauseBtn.style.display = "none";
      resumeBtn.style.display = "";
    } else {
      pauseBtn.style.display = "";
      resumeBtn.style.display = "none";
    }
  } else {
    readBtn.style.display = "";
    stopBtn.style.display = "none";
    pauseBtn.style.display = "none";
    resumeBtn.style.display = "none";
    readBtn.disabled = isRecording;
  }
}

function updateMicButton() {
  if (isRecording) {
    micBtn.classList.add('listening');
    micBtn.textContent = '⏹️ Stop';
    micBtn.disabled = false;
  } else {
    micBtn.classList.remove('listening');
    micBtn.textContent = '🎤 Mic';
    micBtn.disabled = isTTSSpeaking;
  }
}

function insertTextAtCursor(text) {
  const cursorPosition = textArea.selectionStart;
  const selectionEnd = textArea.selectionEnd;
  const currentContent = textArea.value;
  
  const beforeCursor = currentContent.substring(0, cursorPosition);
  const afterCursor = currentContent.substring(selectionEnd);
  
  const newContent = beforeCursor + text + afterCursor;
  textArea.value = newContent;
  
  const newCursorPosition = cursorPosition + text.length;
  textArea.setSelectionRange(newCursorPosition, newCursorPosition);
  textArea.focus();
}

function saveTextToStorage() {
  try {
    let textToSave = textArea.value;
    if (textToSave.trim() && !textToSave.trim().startsWith('---')) {
      textToSave = '\n---\n' + textToSave;
    }
    localStorage.setItem(STORAGE_TEXT_KEY, textToSave);
  } catch (e) {
    console.error('Error saving text buffer:', e);
  }
}

function savePreferences() {
  try {
    localStorage.setItem(STORAGE_VOICE_KEY, voiceSelect.value);
    localStorage.setItem(STORAGE_RATE_KEY, rateSelect.value);
  } catch (e) {
    // ignore storage errors
  }
}

// ============================================================================
// TTS (Text-to-Speech) Functions
// ============================================================================

function populateVoices() {
  const voices = window.speechSynthesis.getVoices() || [];
  
  // If no voices available yet, try again later
  if (voices.length === 0) {
    setTimeout(populateVoices, 100);
    return;
  }

  voiceSelect.innerHTML = "";

  voices.forEach((v, i) => {
    const option = document.createElement("option");
    option.value = v.name;
    option.textContent = `${v.name} (${v.lang})${v.default ? " — default" : ""}`;
    option.dataset.lang = v.lang;
    voiceSelect.appendChild(option);
  });

  // restore saved voice if present
  const saved = localStorage.getItem(STORAGE_VOICE_KEY);
  if (saved) {
    const found = Array.from(voiceSelect.options).find(o => o.value === saved);
    if (found) {
      voiceSelect.value = saved;
    }
  }

  // If nothing selected, pick the default voice or first
  if (!voiceSelect.value && voiceSelect.options.length) {
    const defaultIndex = Array.from(voiceSelect.options).findIndex(o => o.textContent.includes("— default"));
    voiceSelect.selectedIndex = defaultIndex >= 0 ? defaultIndex : 0;
  }

  // restore rate
  const savedRate = localStorage.getItem(STORAGE_RATE_KEY);
  if (savedRate) {
    rateSelect.value = savedRate;
  }
}

function speakText(text) {
  if (!text || !text.trim()) {
    setStatus("Nothing to read");
    return;
  }

  if (!supportsSpeech()) {
    setStatus("Browser does not support the Web Speech API.");
    return;
  }

  // Stop recording if active
  if (isRecording) {
    stopRecording();
  }

  window.speechSynthesis.cancel();
  paragraphQueue = [];

  const MAX_CHUNK_LENGTH = 200;
  
  const rawParagraphs = text.split(/\n\s*\n/).filter(p => p.trim().length > 0);
  
  // If no paragraph breaks found, treat the whole text as one paragraph
  const paragraphs = rawParagraphs.length > 0 ? rawParagraphs : [text];
  
  // Further split long paragraphs into sentences
  const chunks = [];
  for (const para of paragraphs) {
    const trimmed = para.trim().replace(/\n/g, ' '); // Replace single newlines with spaces
    
    if (trimmed.length <= MAX_CHUNK_LENGTH) {
      chunks.push(trimmed);
    } else {
      // Split on sentence boundaries (., !, ?)
      // Keep the punctuation with the sentence
      const sentences = trimmed.match(/[^.!?]+[.!?]+\s*/g) || [trimmed];
      
      let currentChunk = '';
      for (const sentence of sentences) {
        const trimmedSentence = sentence.trim();
        if (currentChunk.length + trimmedSentence.length <= MAX_CHUNK_LENGTH) {
          currentChunk += (currentChunk ? ' ' : '') + trimmedSentence;
        } else {
          if (currentChunk) {
            chunks.push(currentChunk);
          }
          // If a single sentence is too long, just add it anyway
          currentChunk = trimmedSentence;
        }
      }
      if (currentChunk) {
        chunks.push(currentChunk);
      }
    }
  }
  
  if (chunks.length === 0) {
    setStatus("Nothing to read");
    return;
  }

  // Populate the queue
  paragraphQueue = chunks;

  // Wait a moment for cancel to complete, then start reading
  setTimeout(() => {
    speakNextParagraph();
  }, 100);
}

function speakNextParagraph() {
  // If queue is empty, we're done
  if (paragraphQueue.length === 0) {
    currentUtterance = null;
    isTTSSpeaking = false;
    isTTSPaused = false;
    setStatus("Done");
    updateReadButton();
    updateMicButton();
    return;
  }

  const voices = window.speechSynthesis.getVoices() || [];

  // If no voices available, try to trigger voice loading
  if (voices.length === 0) {
    // Force voice loading by speaking empty text first
    const tempUtter = new SpeechSynthesisUtterance("");
    window.speechSynthesis.speak(tempUtter);
    window.speechSynthesis.cancel();
    
    // Retry after a moment
    setTimeout(() => speakNextParagraph(), 200);
    return;
  }

  // Get the first paragraph from the queue
  const paragraphText = paragraphQueue[0].trim();

  const utter = new SpeechSynthesisUtterance(paragraphText);

  // prefer exact match by name; fallback to currently selected index
  const chosenName = voiceSelect.value;
  const chosenVoice = voices.find(v => v.name === chosenName) || voices[voiceSelect.selectedIndex] || null;
  if (chosenVoice) {
    utter.voice = chosenVoice;
  }

  utter.rate = parseFloat(rateSelect.value) || 1.0;
  utter.pitch = 1.0;
  utter.volume = 1.0;

  utter.onstart = () => {
    currentUtterance = utter;
    isTTSSpeaking = true;
    const remaining = paragraphQueue.length;
    setStatus(`Speaking... (${remaining} paragraph${remaining > 1 ? 's' : ''} remaining)`);
    updateReadButton();
    updateMicButton();
  };

  utter.onend = () => {
    // Remove the paragraph we just finished reading
    paragraphQueue.shift();
    
    // Continue with the next paragraph
    speakNextParagraph();
  };

  utter.onerror = (evt) => {
    console.error("Speech error:", evt);
    
    // "interrupted" errors can happen during normal cancel operations
    // or when Chrome's speech synthesis times out - try to continue
    if (evt.error === 'interrupted' && paragraphQueue.length > 0) {
      // Remove the current chunk and try the next one
      paragraphQueue.shift();
      if (paragraphQueue.length > 0) {
        setStatus("Resuming after interruption...");
        setTimeout(() => speakNextParagraph(), 100);
        return;
      }
    }
    
    // On other errors, clear everything
    paragraphQueue = [];
    currentUtterance = null;
    isTTSSpeaking = false;
    isTTSPaused = false;
    setStatus("Error during speech: " + (evt.error || "Unknown error"));
    updateReadButton();
    updateMicButton();
  };

  try {
    window.speechSynthesis.speak(utter);
  } catch (e) {
    setStatus("Error: " + e.message);
    console.error("Speech synthesis error:", e);
    paragraphQueue = [];
    isTTSSpeaking = false;
    updateReadButton();
    updateMicButton();
  }
}

// ============================================================================
// STT (Speech-to-Text) Functions - Whisper-based
// ============================================================================

async function startRecording() {
  try {
    // Stop TTS if active
    if (isTTSSpeaking) {
      window.speechSynthesis.cancel();
      currentUtterance = null;
      isTTSSpeaking = false;
      updateReadButton();
    }

    // Request microphone access
    audioStream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        sampleRate: 16000
      } 
    });

    // Set up AudioContext for silence detection
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(audioStream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);

    // Set up MediaRecorder
    // Try to use a format that's widely supported
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') 
      ? 'audio/webm;codecs=opus'
      : MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : 'audio/mp4';
    
    mediaRecorder = new MediaRecorder(audioStream, { mimeType });
    audioChunks = [];

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      // Process the recorded audio
      if (audioChunks.length > 0) {
        const audioBlob = new Blob(audioChunks, { type: mimeType });
        await sendToWhisper(audioBlob);
      }
      audioChunks = [];
    };

    // Start recording
    mediaRecorder.start(100); // Collect data every 100ms
    isRecording = true;
    silenceStartTime = null;

    textArea.classList.add('listening-textarea');
    setStatus("Listening...");
    updateReadButton();
    updateMicButton();

    // Start silence detection loop
    detectSilence();

  } catch (error) {
    console.error('Error starting recording:', error);
    setStatus(`Microphone error: ${error.message}`);
    stopRecording();
  }
}

function detectSilence() {
  if (!isRecording || !analyser) {
    return;
  }

  const dataArray = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(dataArray);

  // Calculate RMS (root mean square) for volume level
  let sum = 0;
  for (let i = 0; i < dataArray.length; i++) {
    const normalized = (dataArray[i] - 128) / 128;
    sum += normalized * normalized;
  }
  const rms = Math.sqrt(sum / dataArray.length);

  const now = Date.now();

  if (rms < SILENCE_THRESHOLD) {
    // Below threshold - might be silence
    if (silenceStartTime === null) {
      silenceStartTime = now;
    } else if (now - silenceStartTime >= SILENCE_DURATION_MS) {
      // We've had enough silence - check if we have audio to process
      if (audioChunks.length > 0 && mediaRecorder && mediaRecorder.state === 'recording') {
        // Stop current recording to trigger processing
        mediaRecorder.stop();
        
        // Start a new recording session for continuous listening
        setTimeout(() => {
          if (isRecording && audioStream) {
            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') 
              ? 'audio/webm;codecs=opus'
              : MediaRecorder.isTypeSupported('audio/webm')
                ? 'audio/webm'
                : 'audio/mp4';
            
            mediaRecorder = new MediaRecorder(audioStream, { mimeType });
            audioChunks = [];

            mediaRecorder.ondataavailable = (event) => {
              if (event.data.size > 0) {
                audioChunks.push(event.data);
              }
            };

            mediaRecorder.onstop = async () => {
              if (audioChunks.length > 0) {
                const audioBlob = new Blob(audioChunks, { type: mimeType });
                await sendToWhisper(audioBlob);
              }
              audioChunks = [];
            };

            mediaRecorder.start(100);
            silenceStartTime = null;
          }
        }, 50);
      } else {
        silenceStartTime = null;
      }
    }
  } else {
    // Above threshold - not silence
    silenceStartTime = null;
  }

  // Continue the detection loop
  animationFrameId = requestAnimationFrame(detectSilence);
}

async function sendToWhisper(audioBlob) {
  // Don't process very short audio clips
  if (audioBlob.size < 1000) {
    return;
  }

  setStatus("Processing speech...");

  try {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');

    const response = await fetch('/transcribe', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();
    
    if (data.success && data.text && data.text.trim()) {
      // Lowercase the first letter (sentence fragments don't need caps)
      let text = data.text.trim();
      if (text.length > 0 && text[0] === text[0].toUpperCase() && text[0] !== text[0].toLowerCase()) {
        text = text[0].toLowerCase() + text.slice(1);
      }
      // Insert the transcribed text at cursor position
      insertTextAtCursor(text + ' ');
      setStatus("Listening...");
    } else {
      // No text detected, just continue listening
      setStatus("Listening...");
    }

  } catch (error) {
    console.error('Transcription error:', error);
    setStatus(`Error: ${error.message}`);
    
    // Continue listening even after an error
    setTimeout(() => {
      if (isRecording) {
        setStatus("Listening...");
      }
    }, 2000);
  }
}

function stopRecording() {
  isRecording = false;

  // Stop the silence detection loop
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }

  // Stop MediaRecorder
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  mediaRecorder = null;

  // Close AudioContext
  if (audioContext) {
    audioContext.close().catch(() => {});
    audioContext = null;
    analyser = null;
  }

  // Stop all audio tracks
  if (audioStream) {
    audioStream.getTracks().forEach(track => track.stop());
    audioStream = null;
  }

  audioChunks = [];
  silenceStartTime = null;

  textArea.classList.remove('listening-textarea');
  setStatus("Ready");
  updateReadButton();
  updateMicButton();

  // Save text when stopping
  saveTextToStorage();
}

function toggleRecording() {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
}

// ============================================================================
// Event Handlers
// ============================================================================

// TTS buttons
readBtn.addEventListener("click", () => {
  // Start TTS
  const selStart = textArea.selectionStart ?? 0;
  const selEnd = textArea.selectionEnd ?? 0;
  
  let textToRead;
  if (selEnd > selStart) {
    // If there's a selection, read only the selected text
    textToRead = textArea.value.substring(selStart, selEnd);
  } else {
    // Otherwise, read from cursor position to the end
    textToRead = textArea.value.substring(selStart);
    // If at the very end (nothing to read), start from the beginning
    if (!textToRead.length) {
      textToRead = textArea.value;
    }
  }
  speakText(textToRead);
});

stopBtn.addEventListener("click", () => {
  // Stop TTS and clear the queue
  if (supportsSpeech()) {
    window.speechSynthesis.cancel();
    setStatus("Stopped");
  }
  paragraphQueue = [];
  currentUtterance = null;
  isTTSSpeaking = false;
  isTTSPaused = false;
  updateReadButton();
  updateMicButton();
});

pauseBtn.addEventListener("click", () => {
  if (supportsSpeech() && isTTSSpeaking && !isTTSPaused) {
    window.speechSynthesis.pause();
    isTTSPaused = true;
    setStatus("Paused");
    updateReadButton();
  }
});

resumeBtn.addEventListener("click", () => {
  if (supportsSpeech() && isTTSSpeaking && isTTSPaused) {
    window.speechSynthesis.resume();
    isTTSPaused = false;
    setStatus("Speaking...");
    updateReadButton();
  }
});

voiceSelect.addEventListener("change", () => {
  savePreferences();
});

rateSelect.addEventListener("change", savePreferences);

// STT button (Whisper)
micBtn.addEventListener("click", toggleRecording);

// Copy button
copyBtn.addEventListener("click", async () => {
  const text = textArea.value;
  if (!text || !text.trim()) {
    setStatus("Nothing to copy");
    return;
  }

  try {
    await navigator.clipboard.writeText(text);
    setStatus("Copied to clipboard!");
    setTimeout(() => setStatus("Ready"), 2000);
  } catch (error) {
    console.error('Copy failed:', error);
    setStatus("Failed to copy to clipboard");
    setTimeout(() => setStatus("Ready"), 2000);
  }
});

// Clear button
clearBtn.addEventListener("click", () => {
  textArea.value = "";
  textArea.focus();
  setStatus("Text cleared");
  setTimeout(() => setStatus("Ready"), 2000);
});

// Save on page close
window.addEventListener("beforeunload", saveTextToStorage);

// Keyboard shortcuts
textArea.addEventListener("keydown", (evt) => {
  if ((evt.ctrlKey || evt.metaKey) && evt.key === "Enter") {
    evt.preventDefault();
    readBtn.click();
    return;
  }
  if (evt.key === "Escape") {
    evt.preventDefault();
    if (isRecording) {
      micBtn.click();
    } else if (isTTSSpeaking) {
      stopBtn.click();
    }
    return;
  }
  if ((evt.ctrlKey || evt.metaKey) && evt.key.toLowerCase() === "m") {
    evt.preventDefault();
    micBtn.click();
    return;
  }
});

// ============================================================================
// Initialization
// ============================================================================

// Initialize TTS
if (!supportsSpeech()) {
  setStatus("Web Speech API not supported in this browser. Use Chrome/Chromium for best results.");
  readBtn.disabled = true;
} else {
  setStatus("Loading voices...");
  populateVoices();
  window.speechSynthesis.onvoiceschanged = () => {
    populateVoices();
    setStatus("Ready");
  };
  setTimeout(() => {
    populateVoices();
    setStatus("Ready");
  }, 250);
}

// Restore saved text buffer on page load
try {
  const savedText = localStorage.getItem(STORAGE_TEXT_KEY);
  if (savedText) {
    textArea.value = savedText;
  }
} catch (e) {
  console.error('Error restoring text buffer:', e);
}
// Focus textarea
setTimeout(() => textArea.focus(), 1000);
setTimeout(() => textArea.setSelectionRange(0, 0), 1200);

// Check URL parameter for auto-start mic
function checkAutoMicStart() {
  const urlParams = new URLSearchParams(window.location.search);
  const micParam = urlParams.get('mic');
  if (micParam && micParam.toLowerCase() === 'on') {
    if (!isRecording) {
      toggleRecording();
    }
  }
}
setTimeout(checkAutoMicStart, 1500);

// Debug helpers
window.__tts = {
  speakNow: (txt) => speakText(String(txt)),
  cancel: () => { if (supportsSpeech()) window.speechSynthesis.cancel(); }
};
