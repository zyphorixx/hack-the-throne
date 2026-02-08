"""
Video Pipeline for face detection and recognition.
Processes video frames to extract face embeddings for identity matching.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np

logger = logging.getLogger("webrtc.video")

# Try to import face_recognition (requires dlib)
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    logger.info("face_recognition library loaded successfully")
except ImportError:
    face_recognition = None
    FACE_RECOGNITION_AVAILABLE = False
    logger.warning("face_recognition not available - face detection disabled")

# Try to import OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available")


@dataclass
class FaceDetection:
    """Represents a detected face in a frame."""
    face_id: str
    embedding: List[float]  # 128-dim dlib embedding
    location: tuple  # (top, right, bottom, left)
    confidence: float
    timestamp: float


@dataclass
class VideoPipelineConfig:
    """Configuration for video processing."""
    target_fps: float = 1.0  # Process 1 frame per second (save resources)
    min_face_size: int = 50  # Minimum face size in pixels
    embedding_model: str = "large"  # 'small' (5 landmarks) or 'large' (68 landmarks)
    match_threshold: float = 0.6  # Face distance threshold for matching


class VideoPipeline:
    """
    Processes video frames to detect and recognize faces.
    
    Workflow:
    1. Receive video frame from WebRTC
    2. Detect faces using face_recognition (HOG or CNN)
    3. Extract 128-dim face embeddings
    4. Match against known speakers in Convex
    5. Return identity information
    """

    def __init__(
        self,
        config: Optional[VideoPipelineConfig] = None,
        convex_service: Optional[Any] = None,
    ):
        self.config = config or VideoPipelineConfig()
        self._convex_service = convex_service
        self._last_process_time: float = 0.0
        self._known_faces: Dict[str, np.ndarray] = {}  # speaker_id -> embedding
        self._face_cache: Dict[str, FaceDetection] = {}  # face_id -> detection
        
        if not FACE_RECOGNITION_AVAILABLE:
            logger.warning("VideoPipeline initialized but face_recognition not available")

    @property
    def is_available(self) -> bool:
        """Check if face recognition is available."""
        return FACE_RECOGNITION_AVAILABLE and CV2_AVAILABLE

    async def process_frame(self, frame: np.ndarray, timestamp: float) -> List[FaceDetection]:
        """
        Process a single video frame for face detection.
        
        Args:
            frame: BGR numpy array from OpenCV/WebRTC
            timestamp: Frame timestamp in seconds
            
        Returns:
            List of detected faces with embeddings
        """
        if not self.is_available:
            return []

        # Rate limiting - only process at target FPS
        elapsed = timestamp - self._last_process_time
        if elapsed < (1.0 / self.config.target_fps):
            return list(self._face_cache.values())

        self._last_process_time = timestamp

        try:
            # Convert BGR to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect face locations
            face_locations = await asyncio.to_thread(
                face_recognition.face_locations,
                rgb_frame,
                model="hog"  # Use 'cnn' for GPU acceleration
            )

            if not face_locations:
                self._face_cache.clear()
                return []

            # Extract face encodings (128-dim embeddings)
            face_encodings = await asyncio.to_thread(
                face_recognition.face_encodings,
                rgb_frame,
                face_locations,
                model=self.config.embedding_model
            )

            detections: List[FaceDetection] = []
            for i, (location, encoding) in enumerate(zip(face_locations, face_encodings)):
                top, right, bottom, left = location
                face_size = max(right - left, bottom - top)

                # Skip small faces
                if face_size < self.config.min_face_size:
                    continue

                # Generate stable face ID based on position
                face_id = f"face_{i}"

                detection = FaceDetection(
                    face_id=face_id,
                    embedding=encoding.tolist(),
                    location=location,
                    confidence=1.0,  # face_recognition doesn't provide confidence
                    timestamp=timestamp,
                )
                detections.append(detection)

            # Update cache
            self._face_cache = {d.face_id: d for d in detections}

            logger.debug("Detected %d faces in frame", len(detections))
            return detections

        except Exception as exc:
            logger.error("Face detection failed: %s", exc)
            return []

    async def match_face_to_speaker(
        self,
        face_embedding: List[float],
    ) -> Optional[Dict[str, Any]]:
        """
        Match a face embedding against known speakers in database.
        
        Args:
            face_embedding: 128-dim face embedding
            
        Returns:
            Speaker info if match found, None otherwise
        """
        if not self._convex_service or not self._convex_service.is_available:
            return None

        try:
            # Use Convex vector search for face matching
            # This would call a new action: findSpeakerByFace
            result = await self._convex_service.find_speaker_by_face(
                face_embedding,
                threshold=self.config.match_threshold
            )
            
            if result and result.get("found"):
                logger.info(
                    "Face matched to speaker %s (score=%.2f)",
                    result.get("speakerId"),
                    result.get("score", 0)
                )
                return result

            return None
            
        except Exception as exc:
            logger.error("Face matching failed: %s", exc)
            return None

    async def update_speaker_face(
        self,
        speaker_id: str,
        face_embedding: List[float],
    ) -> bool:
        """
        Associate a face embedding with an existing speaker.
        
        Args:
            speaker_id: Convex speaker ID
            face_embedding: 128-dim face embedding
            
        Returns:
            True if successful
        """
        if not self._convex_service or not self._convex_service.is_available:
            return False

        try:
            await self._convex_service.update_speaker_face(
                speaker_id,
                face_embedding
            )
            
            # Update local cache
            self._known_faces[speaker_id] = np.array(face_embedding)
            
            logger.info("Updated face embedding for speaker %s", speaker_id)
            return True
            
        except Exception as exc:
            logger.error("Failed to update speaker face: %s", exc)
            return False

    def compare_faces(
        self,
        known_embedding: List[float],
        unknown_embedding: List[float],
    ) -> float:
        """
        Compare two face embeddings and return distance.
        
        Args:
            known_embedding: Known face embedding
            unknown_embedding: Unknown face embedding
            
        Returns:
            Euclidean distance (lower = more similar, <0.6 is same person)
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return float('inf')

        distance = face_recognition.face_distance(
            [np.array(known_embedding)],
            np.array(unknown_embedding)
        )[0]
        
        return float(distance)


# Global instance
_video_pipeline: Optional[VideoPipeline] = None


def get_video_pipeline(
    config: Optional[VideoPipelineConfig] = None,
    convex_service: Optional[Any] = None,
) -> VideoPipeline:
    """Get or create the global video pipeline instance."""
    global _video_pipeline
    if _video_pipeline is None:
        _video_pipeline = VideoPipeline(config, convex_service)
    return _video_pipeline
