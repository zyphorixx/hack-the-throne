"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Mic, MicOff, Video, VideoOff, Circle, Square, Loader2 } from "lucide-react"
import { FaceNotification } from "@/components/face-notification"
import { useFaceDetection } from "@/hooks/use-face-detection"
import { cn } from "@/lib/utils"
import {
  calculateVideoTransform,
  mapBoundingBoxToOverlay,
  calculateNotificationPosition,
} from "@/lib/coordinate-mapper"
import PersonContextCard from "./PersonContextCard"

interface PersonData {
  name: string
  description: string
  relationship: string
  person_id?: string
}

type FacePersonMap = Map<string, PersonData>

export default function WebcamStream() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const overlayRef = useRef<HTMLDivElement>(null)
  const pcRef = useRef<RTCPeerConnection | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [isVideoReady, setIsVideoReady] = useState(false)
  const [facePersonData, setFacePersonData] = useState<FacePersonMap>(new Map())
  const [latestPersonData, setLatestPersonData] = useState<PersonData | null>(null)
  const [activeSpeaker, setActiveSpeaker] = useState<PersonData | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [eventSource, setEventSource] = useState<EventSource | null>(null)
  const [isMuted, setIsMuted] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [transcription, setTranscription] = useState<string | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])

  const INFERENCE_BACKEND_URL = "http://localhost:8000"
  const OFFER_BACKEND_URL = "http://localhost:8000"

  const { detectedFaces, isLoading: isFaceDetectionLoading, error: faceDetectionError } = useFaceDetection(
    videoRef.current,
    {
      enabled: isStreaming && isVideoReady,
      minDetectionConfidence: 0.5,
      targetFps: 20,
      useWorker: true,
    }
  )

  useEffect(() => {
    if (!latestPersonData || detectedFaces.length === 0) {
      return
    }

    const mostProminentFace = detectedFaces.reduce((prev, current) => {
      const prevScore = (prev.boundingBox.width * prev.boundingBox.height) * prev.confidence
      const currentScore = (current.boundingBox.width * current.boundingBox.height) * current.confidence
      return currentScore > prevScore ? current : prev
    })

    setFacePersonData((prev) => {
      const newMap = new Map(prev)
      newMap.set(mostProminentFace.id, latestPersonData)
      return newMap
    })

    console.log(`[FaceDetection] Associated "${latestPersonData.name}" with face ${mostProminentFace.id}`)
    setLatestPersonData(null)
  }, [latestPersonData, detectedFaces])

  useEffect(() => {
    const currentFaceIds = new Set(detectedFaces.map(f => f.id))
    setFacePersonData((prev) => {
      const newMap = new Map(prev)
      for (const faceId of newMap.keys()) {
        if (!currentFaceIds.has(faceId)) {
          newMap.delete(faceId)
        }
      }
      return newMap
    })
  }, [detectedFaces])

  useEffect(() => {
    startWebcam()
    connectSSE()

    return () => {
      stopWebcam()
      disconnectSSE()
    }
  }, [])

  const connectSSE = () => {
    try {
      const es = new EventSource(`${INFERENCE_BACKEND_URL}/stream/inference`)

      console.log('[SSE] Connecting to:', `${INFERENCE_BACKEND_URL}/stream/inference`)

      es.onopen = () => {
        console.log('[SSE] Connected')
      }

      es.addEventListener('inference', (event) => {
        try {
          const message = JSON.parse(event.data)
          console.log('[SSE] Received inference event:', message)

          if (message.name && message.description && message.relationship) {
            const personData = {
              name: message.name,
              description: message.description,
              relationship: message.relationship,
              person_id: message.person_id,
            }
            setLatestPersonData(personData) // For face mapping (consumed)
            setActiveSpeaker(personData)    // For UI Card (persistent)
          }
        } catch (err) {
          console.error('[SSE] Error parsing message:', err)
        }
      })

      es.onerror = (error) => {
        console.error('[SSE] Error:', error)
      }

      setEventSource(es)
    } catch (err) {
      console.error('[SSE] Connection error:', err)
    }
  }

  const disconnectSSE = () => {
    if (eventSource) {
      eventSource.close()
      setEventSource(null)
    }
  }

  const waitForIceGathering = (pc: RTCPeerConnection): Promise<void> => {
    return new Promise((resolve) => {
      if (pc.iceGatheringState === 'complete') {
        resolve()
        return
      }

      const checkState = () => {
        if (pc.iceGatheringState === 'complete') {
          pc.removeEventListener('icegatheringstatechange', checkState)
          resolve()
        }
      }

      pc.addEventListener('icegatheringstatechange', checkState)
    })
  }

  const setupWebRTC = async (stream: MediaStream) => {
    try {
      console.log('[WebRTC] Setting up peer connection')
      const pc = new RTCPeerConnection()
      pcRef.current = pc

      stream.getTracks().forEach(track => {
        console.log('[WebRTC] Adding track:', track.kind)
        pc.addTrack(track, stream)
      })

      pc.onconnectionstatechange = () => {
        console.log('[WebRTC] Connection state:', pc.connectionState)
        setIsConnected(pc.connectionState === 'connected')
      }

      const offer = await pc.createOffer()
      await pc.setLocalDescription(offer)
      console.log('[WebRTC] Created offer, waiting for ICE gathering...')

      await waitForIceGathering(pc)
      console.log('[WebRTC] ICE gathering complete, sending offer to backend')

      const response = await fetch(`${OFFER_BACKEND_URL}/offer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: pc.localDescription!.sdp,
          type: pc.localDescription!.type
        })
      })

      if (!response.ok) {
        throw new Error(`Backend responded with ${response.status}`)
      }

      const answer = await response.json()
      console.log('[WebRTC] Received answer from backend')
      await pc.setRemoteDescription(answer)
      console.log('[WebRTC] Connection established!')

    } catch (err) {
      console.error('[WebRTC] Setup error:', err)
      setIsConnected(false)
    }
  }

  const startWebcam = async () => {
    try {
      console.log('[Webcam] Requesting media access...')
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: true,
      })

      if (videoRef.current) {
        const video = videoRef.current

        const handleMetadataLoaded = () => {
          console.log('[Webcam] Video metadata loaded:', {
            width: video.videoWidth,
            height: video.videoHeight
          })
          setIsVideoReady(true)
        }

        video.addEventListener('loadedmetadata', handleMetadataLoaded)

        if (video.videoWidth > 0 && video.videoHeight > 0) {
          handleMetadataLoaded()
        }

        video.srcObject = stream
        streamRef.current = stream
        stream.getAudioTracks().forEach((track) => {
          track.enabled = !isMuted
        })
        setIsStreaming(true)
        console.log('[Webcam] Stream started')

        setTimeout(() => setupWebRTC(stream), 1000)
      }
    } catch (err) {
      console.error("[Webcam] Error accessing webcam:", err)
    }
  }

  const stopWebcam = () => {
    console.log('[Webcam] Stopping stream')

    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach((track) => track.stop())
      videoRef.current.srcObject = null
    }

    if (pcRef.current) {
      pcRef.current.close()
      pcRef.current = null
    }

    streamRef.current = null
    setIsStreaming(false)
    setIsVideoReady(false)
    setIsConnected(false)
    setIsMuted(false)
    setActiveSpeaker(null)
  }

  const toggleMute = useCallback(() => {
    const stream = streamRef.current
    if (!stream) {
      return
    }
    setIsMuted((prev) => {
      const next = !prev
      stream.getAudioTracks().forEach((track) => {
        track.enabled = !next
      })
      return next
    })
  }, [])

  const startRecording = useCallback(() => {
    const stream = streamRef.current
    if (!stream) return

    audioChunksRef.current = []
    const audioStream = new MediaStream(stream.getAudioTracks())
    const mediaRecorder = new MediaRecorder(audioStream, { mimeType: 'audio/webm' })

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunksRef.current.push(event.data)
      }
    }

    mediaRecorder.onstop = async () => {
      setIsProcessing(true)
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })

      try {
        const formData = new FormData()
        formData.append('audio', audioBlob, 'recording.webm')

        const response = await fetch(`${INFERENCE_BACKEND_URL}/transcribe`, {
          method: 'POST',
          body: formData,
        })

        if (response.ok) {
          const result = await response.json()
          const displayName = result.name || 'Unknown'
          const displayText = result.text || 'No speech detected'

          setTranscription(`${displayName}: ${displayText}`)
          console.log('[Recording] Extracted name:', result.name, '| Relationship:', result.relationship)

          // Update speaker card with extracted info
          if (result.name && result.name !== 'Unknown') {
            setActiveSpeaker({
              name: result.name,
              description: displayText,
              relationship: result.relationship || 'Visitor',
              person_id: result.speaker_id,
            })
          }
        } else {
          setTranscription('Transcription failed')
        }
      } catch (err) {
        console.error('[Recording] Upload error:', err)
        setTranscription('Error processing audio')
      } finally {
        setIsProcessing(false)
      }
    }

    mediaRecorder.start()
    mediaRecorderRef.current = mediaRecorder
    setIsRecording(true)
    setTranscription(null)
    console.log('[Recording] Started')
  }, [])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      mediaRecorderRef.current = null
      setIsRecording(false)
      console.log('[Recording] Stopped')
    }
  }, [isRecording])

  const faceNotifications = detectedFaces.map((face) => {
    const video = videoRef.current
    const overlay = overlayRef.current

    if (!video || !overlay) return null

    const videoWidth = video.videoWidth
    const videoHeight = video.videoHeight
    const overlayWidth = overlay.clientWidth
    const overlayHeight = overlay.clientHeight

    if (videoWidth === 0 || videoHeight === 0) return null

    const transform = calculateVideoTransform(
      videoWidth,
      videoHeight,
      overlayWidth,
      overlayHeight
    )

    const overlayBox = mapBoundingBoxToOverlay(
      face.boundingBox,
      transform,
      overlayWidth,
      true
    )

    const position = calculateNotificationPosition(
      overlayBox,
      overlayWidth,
      overlayHeight
    )

    return {
      face,
      position,
    }
  }).filter((n) => n !== null)

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-black">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className={cn(
          "absolute inset-0 h-full w-full object-cover transition-all duration-300",
          "scale-100"
        )}
        style={{ transform: 'scaleX(-1)' }}
      />

      {/* Real-time Person Context Card */}
      <PersonContextCard
        speakerId={activeSpeaker?.person_id || null}
        speakerName={activeSpeaker?.name || null}
      />

      <div ref={overlayRef} className="absolute inset-0 pointer-events-none">
        {faceNotifications.map((notification) => {
          const person = facePersonData.get(notification!.face.id)
          return (
            <FaceNotification
              key={notification!.face.id}
              faceId={notification!.face.id}
              left={notification!.position.left}
              top={notification!.position.top}
              confidence={notification!.face.confidence}
              name={person?.name}
              description={person?.description}
              relationship={person?.relationship}
            />
          )
        })}
      </div>

      <div className="absolute top-4 right-4 flex items-center gap-2 px-3 py-2 bg-black/60 backdrop-blur-sm rounded-full">
        <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} ${isConnected ? 'animate-pulse' : ''}`} />
        <span className="text-xs text-white/80">{isConnected ? 'Connected (WebRTC)' : 'Disconnected'}</span>
      </div>

      {isFaceDetectionLoading && (
        <div className="absolute top-4 left-4 px-3 py-2 bg-black/60 backdrop-blur-sm rounded-full">
          <span className="text-xs text-white/80">Loading face detection...</span>
        </div>
      )}
      {faceDetectionError && (
        <div className="absolute top-4 left-4 px-3 py-2 bg-red-500/60 backdrop-blur-sm rounded-full">
          <span className="text-xs text-white/80">Face detection error</span>
        </div>
      )}

      <div className="absolute bottom-0 left-0 right-0 flex flex-col items-center px-6 py-4 bg-gradient-to-t from-black/80 to-transparent">
        {/* Transcription display */}
        {(transcription || isProcessing) && (
          <div className="mb-4 px-4 py-3 bg-black/70 backdrop-blur-sm rounded-lg max-w-lg w-full">
            {isProcessing ? (
              <div className="flex items-center gap-2 text-white/80">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-sm">Processing audio...</span>
              </div>
            ) : (
              <p className="text-sm text-white">{transcription}</p>
            )}
          </div>
        )}

        <div className="flex flex-wrap items-center gap-3">
          {/* Record Button */}
          <Button
            size="icon"
            variant={isRecording ? "destructive" : "default"}
            onClick={isRecording ? stopRecording : startRecording}
            className={cn(
              "h-14 w-14 rounded-full transition-all",
              isRecording && "animate-pulse ring-2 ring-red-500 ring-offset-2 ring-offset-black"
            )}
            disabled={!isStreaming || isProcessing}
          >
            {isRecording ? (
              <Square className="h-5 w-5 fill-current" />
            ) : (
              <Circle className="h-6 w-6 fill-red-500 text-red-500" />
            )}
          </Button>
          <span className="text-sm text-white/80 min-w-[80px]">
            {isRecording ? "Stop" : isProcessing ? "Processing..." : "Record"}
          </span>

          <Button
            size="icon"
            variant={isMuted ? "default" : "secondary"}
            onClick={toggleMute}
            className="h-12 w-12 rounded-full"
            disabled={!isStreaming}
            aria-pressed={isMuted}
          >
            {isMuted ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
          </Button>
          <span className="text-sm text-white/80">{isMuted ? "Unmute" : "Mute"}</span>

          <Button
            size="icon"
            variant={isStreaming ? "default" : "secondary"}
            onClick={isStreaming ? stopWebcam : startWebcam}
            className="h-12 w-12 rounded-full"
          >
            {isStreaming ? <Video className="h-5 w-5" /> : <VideoOff className="h-5 w-5" />}
          </Button>
          <span className="text-sm text-white/80">{isStreaming ? "Stop Video" : "Start Video"}</span>
        </div>
      </div>
    </div>
  )
}
