import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
    // Speaker profiles with voice embeddings for vector search
    speakers: defineTable({
        name: v.optional(v.string()),           // "Sarah" (learned from "I'm Sarah" or manually set)
        relationship: v.optional(v.string()),   // "Your daughter"
        description: v.optional(v.string()),    // "Lives in Delhi, has 2 kids, works at Google"
        photoUrl: v.optional(v.string()),       // Profile photo URL
        embedding: v.array(v.float64()),        // 512-dim pyannote speaker embedding
        faceEmbedding: v.optional(v.array(v.float64())), // 128-dim dlib face embedding
        lastSeen: v.number(),                   // Unix timestamp (ms)
        seenCount: v.number(),                  // How many times detected
        totalSpeakingTime: v.float64(),         // Total seconds speaking
        createdAt: v.number(),                  // First encounter timestamp
    }).vectorIndex("by_embedding", {
        vectorField: "embedding",
        dimensions: 512,                        // pyannote embedding dimension
    }).vectorIndex("by_face_embedding", {
        vectorField: "faceEmbedding",
        dimensions: 128,                        // dlib embedding dimension
    }).index("by_name", ["name"]),

    // Conversation history - transcripts with speaker
    conversations: defineTable({
        speakerId: v.id("speakers"),
        transcript: v.string(),
        summary: v.optional(v.string()),        // LLM-generated summary
        topics: v.optional(v.array(v.string())),
        sentiment: v.optional(v.string()),      // positive/neutral/negative
        timestamp: v.number(),                  // When conversation happened
        durationSeconds: v.float64(),
    }).index("by_speaker", ["speakerId"])
        .index("by_timestamp", ["timestamp"]),

    // Active session state
    sessions: defineTable({
        sessionId: v.string(),                  // WebRTC session ID
        activeSpeakerId: v.optional(v.id("speakers")),
        startedAt: v.number(),
        lastActivity: v.number(),
    }).index("by_sessionId", ["sessionId"]),
});
