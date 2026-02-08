"""FastAPI application providing WebRTC ingress and audio pipeline stubs."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Set
from uuid import uuid4

from aiortc import RTCPeerConnection, RTCSessionDescription
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .audio import AdaptiveDenoiser, AudioPipeline, PipelineConfig
from .core import AudioChunk
from .services.conversation_stream import ConversationEventBus
from .video.pipeline import VideoPipeline, get_video_pipeline
from .services.convex_client import get_convex_service

logger = logging.getLogger("webrtc")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

app = FastAPI(title="Multimodal Ingress Prototype")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pcs: Set[RTCPeerConnection] = set()
ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_INDEX = ROOT_DIR / "dummy_frontend" / "index.html"

# Load environment variables from project root .env (if present)
load_dotenv(ROOT_DIR / ".env")
os.environ.setdefault("TORCHAUDIO_PYTHON_ONLY", "1")

pipeline_config = PipelineConfig()
conversation_bus = ConversationEventBus()
audio_pipeline = AudioPipeline(
    denoiser=AdaptiveDenoiser(),
    config=pipeline_config,
    conversation_bus=conversation_bus,
)

# Video pipeline for face recognition
convex_service = get_convex_service()
video_pipeline = get_video_pipeline(convex_service=convex_service)

# Session state for audio-video fusion
# Maps session_id -> {"speaker_id": str, "has_face": bool, "pending_face": List[float]}
session_states: dict[str, dict] = {}

# Global state for latest identified speaker (from audio)
# Used to associate faces with speakers in single-user context
latest_speaker_info: dict = {"id": None, "name": None, "ts": 0}

class SDPModel(BaseModel):
    sdp: str
    type: str


@app.on_event("startup")
async def on_startup() -> None:
    # Don't block startup - whisper will load on first use
    async def _warmup():
        try:
            await audio_pipeline.warm_whisper()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Whisper warm-up failed: %s", exc)
    asyncio.create_task(_warmup())


@app.get("/")
async def index() -> FileResponse:
    if not FRONTEND_INDEX.exists():
        raise HTTPException(status_code=500, detail="Frontend bundle missing")
    return FileResponse(FRONTEND_INDEX)


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe uploaded audio file and publish to conversation bus for speaker card."""
    import subprocess
    import tempfile
    import numpy as np
    from uuid import uuid4
    from .core import ConversationEvent, ConversationUtterance
    
    logger.info("Received audio upload: %s (%s)", audio.filename, audio.content_type)
    
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            input_path = tmp.name
        
        # Convert to WAV using ffmpeg
        output_path = input_path.replace(".webm", ".wav")
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "quiet", "-i", input_path,
                    "-ar", "16000", "-ac", "1", "-f", "wav", output_path
                ],
                capture_output=True,
                timeout=30,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                logger.error("FFmpeg error: %s", result.stderr.decode())
                raise HTTPException(status_code=500, detail="Audio conversion failed")
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="FFmpeg not installed")
        
        # Load audio as numpy array
        import wave
        with wave.open(output_path, "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        
        # Clean up temp files
        os.unlink(input_path)
        os.unlink(output_path)
        
        # Transcribe using existing pipeline
        segments = await audio_pipeline._transcribe_audio(audio_data)
        
        if segments:
            text = " ".join(seg.text for seg in segments)
            logger.info("Transcription result: %s", text[:100])
            
            # Generate session/conversation IDs for this recording
            session_id = f"record-{uuid4().hex[:8]}"
            conversation_id = f"{session_id}-conv{uuid4().hex[:6]}"
            
            # Use LLM to extract name and relationship from transcription
            extracted_name = "New Person"
            relationship = "Someone you know"
            summary = text[:200] if len(text) > 200 else text
            
            try:
                from openai import AsyncOpenAI
                groq_client = AsyncOpenAI(
                    api_key=os.getenv("GROQ_API_KEY"),
                    base_url="https://api.groq.com/openai/v1"
                )
                
                llm_response = await groq_client.chat.completions.create(
                    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                    messages=[
                        {"role": "system", "content": """You are a person identification AI. Extract the person's name from the conversation.
Return ONLY JSON: {"name": "extracted name or 'Unknown'", "relationship": "relationship like 'Your friend' or 'Visitor'", "summary": "brief summary"}
If no name is mentioned, return "Unknown"."""},
                        {"role": "user", "content": f"Extract person details from: {text}"}
                    ],
                    temperature=0.2,
                    max_tokens=150
                )
                
                import json
                result = json.loads(llm_response.choices[0].message.content.strip())
                extracted_name = result.get("name", "Unknown")
                relationship = result.get("relationship", "Visitor")
                summary = result.get("summary", text[:100])
                logger.info("LLM extracted name: %s (%s)", extracted_name, relationship)
            except Exception as e:
                logger.warning("LLM name extraction failed: %s", e)
            
            # Save to Convex - look up existing speaker first
            speaker_id = None
            final_name = extracted_name
            
            try:
                # If name was extracted, try to find existing speaker with that name
                if extracted_name and extracted_name not in ["Unknown", "New Person"]:
                    existing_speaker = await convex_service.get_speaker_by_name(extracted_name)
                    if existing_speaker:
                        speaker_id = existing_speaker.get("_id")
                        final_name = existing_speaker.get("name", extracted_name)
                        logger.info("Found existing speaker: %s (%s)", final_name, speaker_id)
                        
                        # Add conversation to existing speaker
                        await convex_service.save_conversation(
                            speaker_id=speaker_id,
                            transcript=text,
                            duration_seconds=10.0,
                            summary=summary,
                        )
                    else:
                        # New person with name - create speaker using find_or_create
                        speaker_result = await convex_service.find_or_create_speaker(
                            embedding=[0.0] * 512,
                            name=extracted_name,
                        )
                        if speaker_result:
                            speaker_id = speaker_result.get("speakerId")
                            logger.info("Created new speaker: %s (%s)", extracted_name, speaker_id)
                            await convex_service.save_conversation(
                                speaker_id=speaker_id,
                                transcript=text,
                                duration_seconds=10.0,
                                summary=summary,
                            )
                else:
                    # No name extracted - try to find most recent speaker
                    logger.info("No name extracted (got '%s'), looking up recent speakers...", extracted_name)
                    recent_speakers = await convex_service.list_speakers()
                    logger.info("list_speakers returned: %s", recent_speakers[:2] if recent_speakers else "empty")
                    
                    if recent_speakers and len(recent_speakers) > 0:
                        recent = recent_speakers[0]
                        speaker_id = recent.get("_id")
                        final_name = recent.get("name", "Unknown")
                        logger.info("Using most recent speaker: %s (%s)", final_name, speaker_id)
                        
                        # Add conversation to existing speaker
                        await convex_service.save_conversation(
                            speaker_id=speaker_id,
                            transcript=text,
                            duration_seconds=10.0,
                            summary=summary,
                        )
                    else:
                        # No speakers exist - create anonymous one
                        logger.info("No speakers found, creating Unknown Person")
                        speaker_result = await convex_service.find_or_create_speaker(
                            embedding=[0.0] * 512,
                            name="Unknown Person",
                        )
                        logger.info("find_or_create_speaker result: %s", speaker_result)
                        if speaker_result:
                            speaker_id = speaker_result.get("speakerId")
                            final_name = "Unknown Person"
                            await convex_service.save_conversation(
                                speaker_id=speaker_id,
                                transcript=text,
                                duration_seconds=10.0,
                                summary=summary,
                            )
            except Exception as e:
                logger.warning("Convex speaker operations failed: %s", e)
            
            # Publish to conversation bus for inference service  
            event = ConversationEvent(
                event_type="CONVERSATION_END",
                person_id=speaker_id or f"speaker_record_{uuid4().hex[:6]}",
                conversation_id=conversation_id,
                session_id=session_id,
                conversation=[ConversationUtterance(speaker=final_name, text=text)],
            )
            
            try:
                await conversation_bus.publish(event)
                logger.info("Published CONVERSATION_END for %s (session=%s)", final_name, session_id)
                
                # Update global active speaker state for video correlation
                if speaker_id:
                    global latest_speaker_info
                    latest_speaker_info = {
                        "id": speaker_id,
                        "name": final_name,
                        "ts": time.time()
                    }
                    logger.info("Updated active speaker: %s (%s)", final_name, speaker_id)

            except Exception as e:
                logger.warning("Failed to publish conversation event: %s", e)
            
            return {
                "text": text, 
                "segments": len(segments), 
                "speaker_id": speaker_id,
                "name": final_name,
                "relationship": relationship,
            }
        else:
            return {"text": "", "segments": 0}
            
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Transcription error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))



@app.post("/offer", response_model=SDPModel)
async def offer(session: SDPModel) -> SDPModel:
    logger.info("Received SDP offer payload length=%d", len(session.sdp))
    pc = RTCPeerConnection()
    pcs.add(pc)
    session_id = f"sess-{uuid4().hex[:8]}"


    @pc.on("track")
    def on_track(track) -> None:
        if track.kind == "audio":
            logger.info("Session %s audio track ready", session_id)

        if track.kind == "audio":

            async def consume_audio() -> None:
                last_sample_rate: int | None = None
                while True:
                    try:
                        frame = await track.recv()
                        if not frame.planes:
                            continue
                        plane = frame.planes[0]
                        try:
                            data = plane.to_bytes()
                        except AttributeError:
                            data = bytes(plane)
                        sample_rate = getattr(frame, "sample_rate", None) or 16000
                        last_sample_rate = sample_rate
                        chunk = AudioChunk(
                            session_id=session_id,
                            data=data,
                            sample_rate=sample_rate,
                            timestamp=datetime.utcnow(),
                        )
                        try:
                            await audio_pipeline.process_chunk(chunk)
                        except Exception as exc:  # noqa: BLE001
                            logger.exception(
                                "Audio pipeline error for session %s: %s",
                                session_id,
                                exc,
                            )
                            continue
                    except Exception:  # noqa: BLE001
                        break

                if last_sample_rate:
                    try:
                        await audio_pipeline.flush_session(session_id, last_sample_rate)
                    except Exception as exc:  # noqa: BLE001
                        logger.exception(
                            "Error flushing audio buffer for session %s: %s",
                            session_id,
                            exc,
                        )

            asyncio.create_task(consume_audio())

        else:
            # Video track - process for face recognition
            async def consume_video() -> None:
                import time
                frame_count = 0
                last_face_id = None
                
                while True:
                    try:
                        frame = await track.recv()
                        frame_count += 1
                        
                        # Only process every 30 frames (~1 FPS at 30fps input)
                        if frame_count % 30 != 0:
                            continue
                        
                        if not video_pipeline.is_available:
                            continue
                        
                        # Convert aiortc frame to numpy array
                        try:
                            import numpy as np
                            # Get frame as numpy array (BGR format)
                            img = frame.to_ndarray(format="bgr24")
                            timestamp = time.time()
                            
                            # Process frame for faces
                            detections = await video_pipeline.process_frame(img, timestamp)
                            
                            if detections:
                                # Try to match face to known speaker
                                face = detections[0]  # Use first/largest face
                                result = await video_pipeline.match_face_to_speaker(face.embedding)
                                
                                if result and result.get("found"):
                                    speaker_id = result.get("speakerId")
                                    if speaker_id != last_face_id:
                                        logger.info(
                                            "Face recognized: %s (score=%.2f)",
                                            result.get("speaker", {}).get("name", "Unknown"),
                                            result.get("score", 0)
                                        )
                                        last_face_id = speaker_id
                                else:
                                    # Unknown face - check if we can associate with active audio speaker
                                    global latest_speaker_info
                                    now = time.time()
                                    last_active_ts = latest_speaker_info.get("ts", 0)
                                    
                                    # If someone spoke in the last 15 seconds
                                    if now - last_active_ts < 15.0 and latest_speaker_info.get("id"):
                                        active_id = latest_speaker_info["id"]
                                        active_name = latest_speaker_info["name"]
                                        
                                        logger.info("Associating unknown face with active speaker: %s", active_name)
                                        success = await video_pipeline.update_speaker_face(
                                            active_id, 
                                            face.embedding
                                        )
                                        if success:
                                            logger.info("✓ Learned face for %s!", active_name)
                                            last_face_id = active_id
                                    else:
                                        logger.debug("Unknown face detected")
                                        last_face_id = None
                        except Exception as exc:
                            logger.debug("Video frame processing error: %s", exc)
                            continue
                            
                    except Exception:  # noqa: BLE001
                        break

            asyncio.create_task(consume_video())

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        if pc.connectionState in {"failed", "closed"}:
            await pc.close()
            pcs.discard(pc)

    logger.info("Processing SDP offer for %s", session_id)
    offer_sdp = RTCSessionDescription(sdp=session.sdp, type=session.type)
    await pc.setRemoteDescription(offer_sdp)

    for transceiver in pc.getTransceivers():
        if transceiver.kind in {"audio", "video"}:
            transceiver.direction = "recvonly"

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.1)

    logger.info("Returning answer for PeerConnection %s", id(pc))
    return SDPModel(sdp=pc.localDescription.sdp, type=pc.localDescription.type)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros, return_exceptions=True)
    pcs.clear()
    logger.info("Shutdown complete; closed all peer connections")


@app.get("/stream/conversation")
async def stream_conversation() -> StreamingResponse:
    """Server-Sent Events stream of conversation metadata events."""

    async def event_generator():
        queue = await conversation_bus.subscribe()
        try:
            while True:
                try:
                    event = await queue.get()
                except asyncio.CancelledError:
                    break
                payload = event.model_dump_json()
                yield f"event: conversation\ndata: {payload}\n\n"
        finally:
            await conversation_bus.unsubscribe(queue)

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


@app.get("/stream/inference")
async def stream_inference() -> StreamingResponse:
    """SSE endpoint for streaming inference events to the frontend."""

    async def event_generator():
        queue = await conversation_bus.subscribe()
        try:
            while True:
                try:
                    event = await queue.get()
                except asyncio.CancelledError:
                    break
                # Transform to match frontend expected format
                # Use speaker name from conversation if available
                display_name = "Unknown"
                if event.conversation and len(event.conversation) > 0:
                    display_name = event.conversation[0].speaker or "Unknown"

                payload = {
                    "name": display_name,
                    "description": " ".join(u.text for u in event.conversation) if event.conversation else "",
                    "relationship": "Speaker",
                    "person_id": event.person_id,
                }
                import json
                yield f"event: inference\ndata: {json.dumps(payload)}\n\n"
        finally:
            await conversation_bus.unsubscribe(queue)

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)

