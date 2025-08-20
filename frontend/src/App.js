import React, { useState, useRef } from 'react';
import axios from 'axios';
import { BsMicFill } from 'react-icons/bs'; // Mic icon
import './styles.css'; // Import custom CSS

function App() {
  const [recording, setRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState('');
  const [responseText, setResponseText] = useState('');
  const mediaRecorder = useRef(null);
  const chunks = useRef([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunks.current = [];
      recorder.ondataavailable = (e) => chunks.current.push(e.data);
      recorder.onstop = async () => {
        const blob = new Blob(chunks.current, { type: 'audio/webm' });
        const formData = new FormData();
        formData.append('audio', blob, 'audio.webm');

        try {
          const res = await axios.post(
            process.env.REACT_APP_BACKEND_URL || 'http://localhost:5000/process_audio',
            formData,
            { headers: { 'Content-Type': 'multipart/form-data' } }
          );
          setResponseText(res.data.text);
          const audioBlob = new Blob([Uint8Array.from(atob(res.data.audio), c => c.charCodeAt(0))], { type: 'audio/mp3' });
          setAudioUrl(URL.createObjectURL(audioBlob));
        } catch (err) {
          console.error('Error processing audio:', err);
          setResponseText('Error processing your request.');
        }
      };
      recorder.start();
      setRecording(true);
      mediaRecorder.current = recorder;
    } catch (err) {
      console.error('Error accessing microphone:', err);
      setResponseText('Please allow microphone access.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current) {
      mediaRecorder.current.stop();
      setRecording(false);
    }
  };

  return (
    <div className="app-container">
      <div className="card">
        <h1 className="title">Sampatti Sathi</h1>
        <p className="subtitle">Speak to find government schemes!</p>
        <div className="mic-container">
          <button
            onClick={recording ? stopRecording : startRecording}
            className={`mic-button ${recording ? 'recording' : ''}`}
            aria-label={recording ? 'Stop Recording' : 'Start Recording'}
          >
            <BsMicFill />
          </button>
        </div>
        <p className="status">{recording ? 'Recording...' : 'Click mic to start'}</p>
        <div className="response">
          <p><strong>Response:</strong> {responseText || 'Speak to get assistance.'}</p>
          {audioUrl && (
            <audio controls className="audio-player" autoPlay>
              <source src={audioUrl} type="audio/mp3" />
              Your browser does not support the audio element.
            </audio>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;