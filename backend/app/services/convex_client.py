"""
Convex Memory Service for storing and retrieving speaker profiles and conversations.
Uses Convex's real-time database with vector search for speaker matching.
"""

import asyncio
import logging
import os
from typing import Any, Optional

logger = logging.getLogger("webrtc.convex")

try:
    from convex import ConvexClient
    CONVEX_AVAILABLE = True
except ImportError:
    ConvexClient = None
    CONVEX_AVAILABLE = False


class ConvexMemoryService:
    """
    Service for persisting speaker profiles and conversations to Convex.
    
    Provides:
    - Speaker matching via vector search (256-dim pyannote embeddings)
    - Person context retrieval with conversation history
    - Real-time sync with frontend via Convex subscriptions
    """
    
    def __init__(self, convex_url: Optional[str] = None):
        self._convex_url = convex_url or os.environ.get("CONVEX_URL")
        self._client: Optional[ConvexClient] = None
        self._initialized = False
        
        if not CONVEX_AVAILABLE:
            logger.warning("Convex SDK not installed; memory persistence disabled")
        elif not self._convex_url:
            logger.warning("CONVEX_URL not set; memory persistence disabled")
    
    def _get_client(self) -> Optional[ConvexClient]:
        """Lazy initialization of Convex client."""
        if self._client is None and CONVEX_AVAILABLE and self._convex_url:
            try:
                self._client = ConvexClient(self._convex_url)
                self._initialized = True
                logger.info("Connected to Convex at %s", self._convex_url)
            except Exception as exc:
                logger.error("Failed to connect to Convex: %s", exc)
                self._client = None
        return self._client
    
    @property
    def is_available(self) -> bool:
        """Check if Convex is available and configured."""
        return CONVEX_AVAILABLE and self._convex_url is not None
    
    async def find_or_create_speaker(
        self,
        embedding: list[float],
        name: Optional[str] = None,
        speaking_time: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Find existing speaker by embedding similarity or create new one.
        
        Args:
            embedding: 256-dim pyannote speaker embedding
            name: Optional speaker name (if detected from "I'm X")
            speaking_time: Duration in seconds of this speech segment
            
        Returns:
            Dict with keys: isNew, speakerId, speaker, matchScore
        """
        client = self._get_client()
        if client is None:
            return {"isNew": True, "speakerId": None, "speaker": None, "matchScore": 0}
        
        try:
            # Build args, excluding None values (Convex doesn't accept null for optional fields)
            args = {"embedding": embedding}
            if name is not None:
                args["name"] = name
            if speaking_time is not None:
                args["speakingTime"] = speaking_time
            
            result = await asyncio.to_thread(
                client.action,
                "speakers:findOrCreateSpeaker",
                args
            )
            logger.info(
                "Speaker %s: %s (score=%.2f)",
                "created" if result.get("isNew") else "found",
                result.get("speakerId"),
                result.get("matchScore", 0),
            )
            return result
        except Exception as exc:
            logger.error("Convex findOrCreateSpeaker failed: %s", exc)
            return {"isNew": True, "speakerId": None, "speaker": None, "matchScore": 0}
    
    async def save_conversation(
        self,
        speaker_id: str,
        transcript: str,
        duration_seconds: float,
        summary: Optional[str] = None,
        topics: Optional[list[str]] = None,
    ) -> Optional[str]:
        """
        Save a conversation transcript for a speaker.
        
        Args:
            speaker_id: Convex ID of the speaker
            transcript: Full transcript text
            duration_seconds: Duration of the conversation
            summary: Optional LLM-generated summary
            topics: Optional list of detected topics
            
        Returns:
            Convex ID of the created conversation, or None on failure
        """
        client = self._get_client()
        if client is None or not speaker_id:
            return None
        
        try:
            # Build args, excluding None values
            args = {
                "speakerId": speaker_id,
                "transcript": transcript,
                "durationSeconds": duration_seconds,
            }
            if summary is not None:
                args["summary"] = summary
            if topics is not None:
                args["topics"] = topics
            
            conversation_id = await asyncio.to_thread(
                client.mutation,
                "context:saveConversation",
                args
            )
            logger.info("Saved conversation %s for speaker %s", conversation_id, speaker_id)
            return conversation_id
        except Exception as exc:
            logger.error("Convex saveConversation failed: %s", exc)
            return None
    
    async def get_person_context(self, speaker_id: str) -> Optional[dict[str, Any]]:
        """
        Get full context for a person including profile and recent conversations.
        
        Args:
            speaker_id: Convex ID of the speaker
            
        Returns:
            Person context with profile, lastSeenText, recentConversations, etc.
        """
        client = self._get_client()
        if client is None or not speaker_id:
            return None
        
        try:
            context = await asyncio.to_thread(
                client.query,
                "context:getPersonContext",
                {"speakerId": speaker_id}
            )
            return context
        except Exception as exc:
            logger.error("Convex getPersonContext failed: %s", exc)
            return None
    
    async def update_speaker_name(self, speaker_id: str, name: str) -> bool:
        """
        Update the name of a speaker (e.g., after detecting "I'm Sarah").
        
        Args:
            speaker_id: Convex ID of the speaker
            name: Detected/assigned name
            
        Returns:
            True if successful, False otherwise
        """
        client = self._get_client()
        if client is None or not speaker_id:
            return False
        
        try:
            await asyncio.to_thread(
                client.mutation,
                "speakers:updateSpeakerName",
                {"id": speaker_id, "name": name}
            )
            logger.info("Updated speaker %s name to '%s'", speaker_id, name)
            return True
        except Exception as exc:
            logger.error("Convex updateSpeakerName failed: %s", exc)
            return False
    
    async def update_speaker_profile(
        self,
        speaker_id: str,
        name: Optional[str] = None,
        relationship: Optional[str] = None,
        description: Optional[str] = None,
        photo_url: Optional[str] = None,
    ) -> bool:
        """
        Update speaker profile with additional information.
        
        Args:
            speaker_id: Convex ID of the speaker
            name: Display name
            relationship: E.g., "Your daughter"
            description: E.g., "Lives in Delhi, has 2 kids"
            photo_url: URL to profile photo
            
        Returns:
            True if successful, False otherwise
        """
        client = self._get_client()
        if client is None or not speaker_id:
            return False
        
        try:
            # Build args, excluding None values
            args = {"id": speaker_id}
            if name is not None:
                args["name"] = name
            if relationship is not None:
                args["relationship"] = relationship
            if description is not None:
                args["description"] = description
            if photo_url is not None:
                args["photoUrl"] = photo_url
            
            await asyncio.to_thread(
                client.mutation,
                "speakers:updateSpeakerProfile",
                args
            )
            logger.info("Updated speaker %s profile", speaker_id)
            return True
        except Exception as exc:
            logger.error("Convex updateSpeakerProfile failed: %s", exc)
            return False

    async def find_speaker_by_face(
        self,
        face_embedding: list[float],
        threshold: float = 0.6,
    ) -> dict[str, Any]:
        """
        Find speaker by face embedding similarity.
        
        Args:
            face_embedding: 128-dim dlib face embedding
            threshold: Maximum distance for match (lower = stricter)
            
        Returns:
            Dict with keys: found, speakerId, speaker, score
        """
        client = self._get_client()
        if client is None:
            return {"found": False, "speakerId": None, "speaker": None, "score": 0}

        try:
            result = await asyncio.to_thread(
                client.action,
                "speakers:findSpeakerByFace",
                {"faceEmbedding": face_embedding, "threshold": threshold}
            )
            return result
        except Exception as exc:
            logger.error("Convex findSpeakerByFace failed: %s", exc)
            return {"found": False, "speakerId": None, "speaker": None, "score": 0}

    async def update_speaker_face(
        self,
        speaker_id: str,
        face_embedding: list[float],
    ) -> bool:
        """
        Update the face embedding for a speaker.
        
        Args:
            speaker_id: Convex ID of the speaker
            face_embedding: 128-dim dlib face embedding
            
        Returns:
            True if successful
        """
        client = self._get_client()
        if client is None or not speaker_id:
            return False

        try:
            await asyncio.to_thread(
                client.mutation,
                "speakers:updateSpeakerFace",
                {"id": speaker_id, "faceEmbedding": face_embedding}
            )
            logger.info("Updated face embedding for speaker %s", speaker_id)
            return True
        except Exception as exc:
            logger.error("Convex updateSpeakerFace failed: %s", exc)
            return False

    async def get_speaker_by_name(self, name: str) -> Optional[dict[str, Any]]:
        """
        Get speaker by name.

        Args:
            name: Speaker name

        Returns:
            Speaker profile or None
        """
        client = self._get_client()
        if client is None or not name:
            return None

        try:
            speaker = await asyncio.to_thread(
                client.query,
                "speakers:getSpeakerByName",
                {"name": name}
            )
            return speaker
        except Exception as exc:
            logger.error("Convex getSpeakerByName failed: %s", exc)
            return None

    async def list_speakers(self) -> list[dict[str, Any]]:
        """
        Get all known speakers.
        
        Returns:
            List of speaker profiles
        """
        client = self._get_client()
        if client is None:
            return []
        
        try:
            speakers = await asyncio.to_thread(
                client.query,
                "speakers:listSpeakers",
                {}
            )
            return speakers or []
        except Exception as exc:
            logger.error("Convex listSpeakers failed: %s", exc)
            return []


# Global instance for use across the application
_convex_service: Optional[ConvexMemoryService] = None


def get_convex_service() -> ConvexMemoryService:
    """Get or create the global Convex memory service instance."""
    global _convex_service
    if _convex_service is None:
        _convex_service = ConvexMemoryService()
    return _convex_service
