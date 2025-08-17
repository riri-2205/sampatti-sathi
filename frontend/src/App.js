import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [recording, setRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState('');
  const [responseText, setResponseText] = useState('');
  const [mediaRecorder, setMediaRecorder] = useState(null);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks = [];
      recorder.ondataavailable = e => chunks.push(e.data);
      recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        const formData = new FormData();
        formData.append('audio', blob, 'audio.webm');

        try {
          const res = await axios.post('http://localhost:5000/process_audio', formData);
          setResponseText(res.data.text);
          const audioBlob = new Blob([Uint8Array.from(atob(res.data.audio), c => c.charCodeAt(0))], { type: 'audio/mp3' });
          setAudioUrl(URL.createObjectURL(audioBlob));
        } catch (err) {
          console.error('Error processing audio:', err);
          setResponseText('Error processing your request.');
        }
      };
      recorder.start();
      setMediaRecorder(recorder);
      setRecording(true);
    } catch (err) {
      console.error('Error accessing microphone:', err);
      setResponseText('Please allow microphone access.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.stop();
      setRecording(false);
    }
  };

  return (
    <div style={{ textAlign: 'center', marginTop: '50px', fontFamily: 'Arial' }}>
      <h1>Sampatti Sathi - Farmer AI Chatbot</h1>
      <button
        onClick={recording ? stopRecording : startRecording}
        style={{
          padding: '10px 20px',
          fontSize: '16px',
          backgroundColor: recording ? '#ff4444' : '#4CAF50',
          color: 'white',
          border: 'none',
          borderRadius: '5px',
          cursor: 'pointer'
        }}
      >
        {recording ? 'Stop Recording' : 'Start Mic'}
      </button>
      <div style={{ marginTop: '20px', maxWidth: '600px', margin: '20px auto' }}>
        <p><strong>Response:</strong> {responseText || 'Speak to get assistance.'}</p>
        {audioUrl && <audio controls src={audioUrl} autoPlay style={{ marginTop: '10px' }} />}
      </div>
    </div>
  );
}

export default App;