#!/usr/bin/env python3
"""
FastAPI WebSocket Streaming Server for F5-TTS Progressive Generation
Real-time audio streaming with custom reference audio
"""

import asyncio
import json
import time
import warnings
import base64
import io
from typing import Dict, List, Optional
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from custom_progressive_test import CustomProgressiveStreamingTest
from f5_tts.infer.utils_infer import infer_batch_process, preprocess_ref_audio_text, chunk_text
import uvicorn

warnings.filterwarnings("ignore")

app = FastAPI(title="F5-TTS Progressive Streaming Server")

class StreamingTTSServer:
    def __init__(self):
        self.ref_audio_path = "../ref_converted.wav"
        self.ref_text = open("../ref.txt", "r").read().strip()
        self.tts_engine = None
        self.connected_clients: Dict[str, WebSocket] = {}
        
    async def initialize_tts(self):
        """Initialize TTS engine if not already done"""
        if self.tts_engine is None:
            print("🚀 Initializing F5-TTS engine...")
            self.tts_engine = CustomProgressiveStreamingTest(
                ref_audio_path=self.ref_audio_path,
                ref_text=self.ref_text,
                model_name="F5TTS_v1_Base",
                device="cuda"
            )
            # Warmup the model
            self.tts_engine.warmup_model()
            print("✅ F5-TTS engine ready for streaming!")
    
    async def stream_audio_generation(self, websocket: WebSocket, text: str, nfe_strategy: List[int] = None):
        """Stream audio generation with progressive NFE to WebSocket client"""
        if nfe_strategy is None:
            nfe_strategy = [6, 16, 32]  # Ultra-fast start strategy
        
        try:
            # Send initial status
            await websocket.send_json({
                "type": "status",
                "message": "Starting TTS generation...",
                "text": text,
                "nfe_strategy": nfe_strategy,
                "text_length": len(text)
            })
            
            # Use F5-TTS's built-in chunking
            chunks = chunk_text(text, max_chars=80)
            
            await websocket.send_json({
                "type": "chunks_info",
                "total_chunks": len(chunks),
                "chunks": [{"text": chunk, "nfe": nfe_strategy[min(i, len(nfe_strategy)-1)]} 
                          for i, chunk in enumerate(chunks)]
            })
            
            # Process reference audio
            ref_audio, ref_text = preprocess_ref_audio_text(
                self.ref_audio_path, 
                self.ref_text, 
                show_info=lambda x: None
            )
            
            # Load and prepare reference audio
            audio, sr = torchaudio.load(ref_audio)
            ref_audio_prepared = (audio, sr)
            
            # Progressive streaming generation
            total_start_time = time.time()
            cumulative_audio_duration = 0
            first_chunk_time = None
            
            for i, chunk in enumerate(chunks):
                # Determine NFE for this chunk
                nfe = nfe_strategy[min(i, len(nfe_strategy)-1)]
                
                await websocket.send_json({
                    "type": "chunk_start",
                    "chunk_index": i,
                    "chunk_text": chunk,
                    "nfe": nfe,
                    "total_chunks": len(chunks)
                })
                
                chunk_start_time = time.time()
                
                # Generate audio for this chunk
                chunk_audio_data = []
                
                # Use infer_batch_process with streaming=True for this chunk
                import tqdm
                for result in infer_batch_process(
                    ref_audio_prepared,
                    ref_text,
                    [chunk],  # Single chunk as batch
                    self.tts_engine.model_obj,
                    self.tts_engine.vocoder,
                    mel_spec_type=self.tts_engine.f5tts.mel_spec_type,
                    nfe_step=nfe,
                    cfg_strength=2,
                    device=self.tts_engine.device,
                    streaming=True,
                    chunk_size=2048,
                    progress=tqdm  # Use tqdm module for progress
                ):
                    # When streaming=True, result is (audio_chunk, sample_rate)
                    if len(result) == 2:
                        audio_chunk, sample_rate = result
                        chunk_audio_data.append(audio_chunk)
                        
                        # Record TTFB on first chunk, first audio data
                        if first_chunk_time is None:
                            first_chunk_time = time.time()
                            ttfb = first_chunk_time - chunk_start_time
                            await websocket.send_json({
                                "type": "ttfb",
                                "ttfb_seconds": ttfb,
                                "ttfb_ms": int(ttfb * 1000)
                            })
                
                chunk_end_time = time.time()
                chunk_generation_time = chunk_end_time - chunk_start_time
                
                if chunk_audio_data:
                    # Combine all audio chunks for this text chunk
                    combined_chunk_audio = np.concatenate(chunk_audio_data)
                    audio_duration = len(combined_chunk_audio) / sample_rate
                    cumulative_audio_duration += audio_duration
                    chunk_rtf = chunk_generation_time / audio_duration
                    
                    # Handle audio clipping
                    max_amplitude = np.max(np.abs(combined_chunk_audio))
                    if max_amplitude > 1.0:
                        combined_chunk_audio = combined_chunk_audio / max_amplitude * 0.95
                    
                    # Convert to tensor for processing
                    if isinstance(combined_chunk_audio, np.ndarray):
                        wav_tensor = torch.from_numpy(combined_chunk_audio)
                    else:
                        wav_tensor = combined_chunk_audio
                    if wav_tensor.dim() == 1:
                        wav_tensor = wav_tensor.unsqueeze(0)
                    
                    # Convert audio to base64 for WebSocket transmission
                    buffer = io.BytesIO()
                    torchaudio.save(buffer, wav_tensor, sample_rate, format="wav")
                    audio_base64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    # Send audio chunk to client
                    await websocket.send_json({
                        "type": "audio_chunk",
                        "chunk_index": i,
                        "audio_base64": audio_base64,
                        "sample_rate": sample_rate,
                        "generation_time": chunk_generation_time,
                        "audio_duration": audio_duration,
                        "chunk_rtf": chunk_rtf,
                        "cumulative_audio_duration": cumulative_audio_duration,
                        "nfe_used": nfe
                    })
                else:
                    await websocket.send_json({
                        "type": "chunk_error",
                        "chunk_index": i,
                        "message": f"Failed to generate chunk {i+1}"
                    })
            
            # Send completion status
            total_time = time.time() - total_start_time
            overall_rtf = total_time / cumulative_audio_duration if cumulative_audio_duration > 0 else float('inf')
            
            await websocket.send_json({
                "type": "completion",
                "total_time": total_time,
                "total_audio_duration": cumulative_audio_duration,
                "overall_rtf": overall_rtf,
                "ttfb": first_chunk_time - total_start_time if first_chunk_time else 0,
                "chunks_generated": len(chunks)
            })
            
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })

# Global server instance
streaming_server = StreamingTTSServer()

@app.on_event("startup")
async def startup_event():
    """Initialize TTS engine on startup"""
    await streaming_server.initialize_tts()

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming TTS"""
    await websocket.accept()
    client_id = f"client_{len(streaming_server.connected_clients)}"
    streaming_server.connected_clients[client_id] = websocket
    
    try:
        await websocket.send_json({
            "type": "connected",
            "client_id": client_id,
            "message": "Connected to F5-TTS Progressive Streaming Server"
        })
        
        while True:
            # Wait for client request
            data = await websocket.receive_json()
            
            if data.get("type") == "generate":
                text = data.get("text", "")
                nfe_strategy = data.get("nfe_strategy", [4, 16, 32])
                
                if not text:
                    await websocket.send_json({
                        "type": "error",
                        "message": "No text provided for generation"
                    })
                    continue
                
                # Stream audio generation
                await streaming_server.stream_audio_generation(
                    websocket, text, nfe_strategy
                )
                
            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected")
    except Exception as e:
        print(f"Error with client {client_id}: {e}")
    finally:
        if client_id in streaming_server.connected_clients:
            del streaming_server.connected_clients[client_id]

@app.get("/")
async def get_demo_page():
    """Simple HTML demo page for testing WebSocket streaming"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>F5-TTS Progressive Streaming Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .input-group { margin: 10px 0; }
            .input-group label { display: block; margin-bottom: 5px; }
            .input-group textarea { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
            .input-group input { padding: 8px; margin-right: 10px; }
            .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            .btn:hover { background: #0056b3; }
            .btn:disabled { background: #ccc; cursor: not-allowed; }
            .status { margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 4px; }
            .error { background: #f8d7da; color: #721c24; }
            .success { background: #d4edda; color: #155724; }
            .audio-player { margin: 10px 0; }
            .chunk-info { margin: 5px 0; padding: 5px; background: #e9ecef; border-radius: 3px; font-size: 12px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 F5-TTS Progressive Streaming Demo</h1>
            <p>Real-time TTS streaming with custom reference audio</p>
            
            <div class="input-group">
                <label for="text">Text to Generate:</label>
                <textarea id="text" rows="4" placeholder="Enter text to convert to speech...">Great! I'll read a quick disclaimer. Brian Thompson, today, on the twenty-fourth of June, twenty twenty five. You are authorizing a payment in the amount of two hundred and seventy dollars, plus a five dollars processing fee.</textarea>
            </div>
            
            <div class="input-group">
                <label>NFE Strategy:</label>
                <input type="number" id="nfe1" value="4" min="1" max="64"> →
                <input type="number" id="nfe2" value="16" min="1" max="64"> →
                <input type="number" id="nfe3" value="32" min="1" max="64">
            </div>
            
            <button id="generateBtn" class="btn" onclick="generateSpeech()">🎵 Generate Speech</button>
            <button id="stopBtn" class="btn" onclick="stopGeneration()" disabled>⏹️ Stop</button>
            
            <div id="status" class="status"></div>
            <div id="chunks"></div>
            <div id="audioPlayers"></div>
        </div>
        
        <script>
            let ws = null;
            let isGenerating = false;
            let audioContext = null;
            
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws/stream`);
                
                ws.onopen = function() {
                    updateStatus('Connected to streaming server', 'success');
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    handleMessage(data);
                };
                
                ws.onclose = function() {
                    updateStatus('Disconnected from server', 'error');
                    setTimeout(connectWebSocket, 3000);
                };
                
                ws.onerror = function(error) {
                    updateStatus(`WebSocket error: ${error}`, 'error');
                };
            }
            
            function handleMessage(data) {
                switch(data.type) {
                    case 'connected':
                        updateStatus(`Connected as ${data.client_id}`, 'success');
                        break;
                    case 'status':
                        updateStatus(`${data.message} (${data.text_length} chars)`, 'success');
                        break;
                    case 'chunks_info':
                        displayChunksInfo(data);
                        break;
                    case 'chunk_start':
                        updateChunkStatus(data.chunk_index, 'Generating...', 'info');
                        break;
                    case 'ttfb':
                        updateStatus(`⚡ TTFB: ${data.ttfb_ms}ms`, 'success');
                        break;
                    case 'audio_chunk':
                        handleAudioChunk(data);
                        break;
                    case 'completion':
                        handleCompletion(data);
                        break;
                    case 'error':
                        updateStatus(`Error: ${data.message}`, 'error');
                        isGenerating = false;
                        updateButtons();
                        break;
                }
            }
            
            function displayChunksInfo(data) {
                const chunksDiv = document.getElementById('chunks');
                chunksDiv.innerHTML = '<h3>Text Chunks:</h3>';
                data.chunks.forEach((chunk, i) => {
                    const chunkDiv = document.createElement('div');
                    chunkDiv.className = 'chunk-info';
                    chunkDiv.id = `chunk-${i}`;
                    chunkDiv.innerHTML = `<strong>Chunk ${i+1}:</strong> "${chunk.text}" (NFE: ${chunk.nfe})`;
                    chunksDiv.appendChild(chunkDiv);
                });
            }
            
            function updateChunkStatus(index, status, type) {
                const chunkDiv = document.getElementById(`chunk-${index}`);
                if (chunkDiv) {
                    chunkDiv.style.backgroundColor = type === 'info' ? '#cce5ff' : '#d4edda';
                    chunkDiv.innerHTML += ` - ${status}`;
                }
            }
            
            function handleAudioChunk(data) {
                // Update chunk status
                updateChunkStatus(data.chunk_index, 
                    `✅ Generated in ${data.generation_time.toFixed(3)}s (RTF: ${data.chunk_rtf.toFixed(4)})`, 
                    'success');
                
                // Create audio player
                const audioDiv = document.getElementById('audioPlayers');
                const audioPlayer = document.createElement('audio');
                audioPlayer.controls = true;
                audioPlayer.src = `data:audio/wav;base64,${data.audio_base64}`;
                audioPlayer.autoplay = true;
                
                const playerDiv = document.createElement('div');
                playerDiv.className = 'audio-player';
                playerDiv.innerHTML = `<strong>Chunk ${data.chunk_index + 1}:</strong> `;
                playerDiv.appendChild(audioPlayer);
                audioDiv.appendChild(playerDiv);
            }
            
            function handleCompletion(data) {
                updateStatus(`🎉 Generation complete! Total: ${data.total_time.toFixed(3)}s, RTF: ${data.overall_rtf.toFixed(4)}, TTFB: ${(data.ttfb * 1000).toFixed(0)}ms`, 'success');
                isGenerating = false;
                updateButtons();
            }
            
            function generateSpeech() {
                const text = document.getElementById('text').value;
                if (!text.trim()) {
                    updateStatus('Please enter text to generate', 'error');
                    return;
                }
                
                const nfe1 = parseInt(document.getElementById('nfe1').value);
                const nfe2 = parseInt(document.getElementById('nfe2').value);
                const nfe3 = parseInt(document.getElementById('nfe3').value);
                
                // Clear previous results
                document.getElementById('chunks').innerHTML = '';
                document.getElementById('audioPlayers').innerHTML = '';
                
                isGenerating = true;
                updateButtons();
                
                ws.send(JSON.stringify({
                    type: 'generate',
                    text: text,
                    nfe_strategy: [nfe1, nfe2, nfe3]
                }));
            }
            
            function stopGeneration() {
                if (ws) {
                    ws.close();
                }
                isGenerating = false;
                updateButtons();
                updateStatus('Generation stopped', 'error');
            }
            
            function updateStatus(message, type = 'info') {
                const statusDiv = document.getElementById('status');
                statusDiv.textContent = message;
                statusDiv.className = `status ${type}`;
            }
            
            function updateButtons() {
                document.getElementById('generateBtn').disabled = isGenerating;
                document.getElementById('stopBtn').disabled = !isGenerating;
            }
            
            // Initialize
            connectWebSocket();
        </script>
    </body>
    </html>
    """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "tts_engine_ready": streaming_server.tts_engine is not None,
        "connected_clients": len(streaming_server.connected_clients)
    }

if __name__ == "__main__":
    print("🚀 Starting F5-TTS Progressive Streaming Server...")
    print("📱 Demo page: http://localhost:8000")
    print("🔌 WebSocket endpoint: ws://localhost:8000/ws/stream")
    print("❤️  Health check: http://localhost:8000/health")
    
    uvicorn.run(app, host="0.0.0.0", port=8000) 