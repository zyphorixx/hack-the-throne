import { v } from "convex/values";
import { action, internalMutation, internalQuery, mutation, query } from "./_generated/server";
import { internal } from "./_generated/api";

// ==================== QUERIES ====================

// Get speaker by ID
export const getSpeaker = query({
    args: { id: v.id("speakers") },
    handler: async (ctx, { id }) => {
        return await ctx.db.get(id);
    },
});

// Get all speakers (for listing)
export const listSpeakers = query({
    args: {},
    handler: async (ctx) => {
        return await ctx.db.query("speakers").order("desc").take(50);
    },
});

// Get speaker by name
export const getSpeakerByName = query({
    args: { name: v.string() },
    handler: async (ctx, { name }) => {
        return await ctx.db
            .query("speakers")
            .withIndex("by_name", (q) => q.eq("name", name))
            .first();
    },
});

// ==================== MUTATIONS ====================

// Create a new speaker
export const createSpeaker = mutation({
    args: {
        embedding: v.array(v.float64()),
        name: v.optional(v.string()),
    },
    handler: async (ctx, { embedding, name }) => {
        const now = Date.now();
        return await ctx.db.insert("speakers", {
            name,
            embedding,
            lastSeen: now,
            seenCount: 1,
            totalSpeakingTime: 0,
            createdAt: now,
        });
    },
});

// Update speaker last seen
export const updateSpeakerLastSeen = mutation({
    args: {
        id: v.id("speakers"),
        speakingTime: v.optional(v.float64()),
    },
    handler: async (ctx, { id, speakingTime }) => {
        const speaker = await ctx.db.get(id);
        if (!speaker) return;

        await ctx.db.patch(id, {
            lastSeen: Date.now(),
            seenCount: speaker.seenCount + 1,
            totalSpeakingTime: speaker.totalSpeakingTime + (speakingTime || 0),
        });
    },
});

// Update speaker name (from "I'm Sarah" detection)
export const updateSpeakerName = mutation({
    args: {
        id: v.id("speakers"),
        name: v.string(),
    },
    handler: async (ctx, { id, name }) => {
        await ctx.db.patch(id, { name });
    },
});

// Update speaker profile (name, relationship, description)
export const updateSpeakerProfile = mutation({
    args: {
        id: v.id("speakers"),
        name: v.optional(v.string()),
        relationship: v.optional(v.string()),
        description: v.optional(v.string()),
        photoUrl: v.optional(v.string()),
    },
    handler: async (ctx, { id, ...updates }) => {
        // Filter out undefined values
        const patch: Record<string, unknown> = {};
        for (const [key, value] of Object.entries(updates)) {
            if (value !== undefined) patch[key] = value;
        }
        if (Object.keys(patch).length > 0) {
            await ctx.db.patch(id, patch);
        }
    },
});

// ==================== VECTOR SEARCH ACTION ====================

// Find existing speaker by embedding similarity
export const findSimilarSpeaker = action({
    args: {
        embedding: v.array(v.float64()),
        threshold: v.optional(v.float64()),
    },
    handler: async (ctx, { embedding, threshold = 0.75 }) => {
        const results = await ctx.vectorSearch("speakers", "by_embedding", {
            vector: embedding,
            limit: 1,
        });

        if (results.length > 0 && results[0]._score >= threshold) {
            const speaker = await ctx.runQuery(internal.speakers.getSpeakerInternal, {
                id: results[0]._id
            });
            return {
                found: true,
                speaker,
                score: results[0]._score,
            };
        }

        return { found: false, speaker: null, score: 0 };
    },
});

// Find or create speaker (main entry point)
export const findOrCreateSpeaker = action({
    args: {
        embedding: v.array(v.float64()),
        name: v.optional(v.string()),
        speakingTime: v.optional(v.float64()),
    },
    handler: async (ctx, { embedding, name, speakingTime }) => {
        // Search for existing speaker
        const searchResult = await ctx.vectorSearch("speakers", "by_embedding", {
            vector: embedding,
            limit: 1,
        });

        if (searchResult.length > 0 && searchResult[0]._score >= 0.70) {
            // Found existing speaker - update last seen
            const speakerId = searchResult[0]._id;
            await ctx.runMutation(internal.speakers.updateSpeakerLastSeenInternal, {
                id: speakerId,
                speakingTime,
            });

            // Get full speaker data
            const speaker = await ctx.runQuery(internal.speakers.getSpeakerInternal, {
                id: speakerId,
            });

            return {
                isNew: false,
                speakerId,
                speaker,
                matchScore: searchResult[0]._score,
            };
        }

        // Create new speaker
        const speakerId = await ctx.runMutation(internal.speakers.createSpeakerInternal, {
            embedding,
            name,
        });

        const speaker = await ctx.runQuery(internal.speakers.getSpeakerInternal, {
            id: speakerId,
        });

        return {
            isNew: true,
            speakerId,
            speaker,
            matchScore: 0,
        };
    },
});

// ==================== INTERNAL FUNCTIONS ====================

export const getSpeakerInternal = internalQuery({
    args: { id: v.id("speakers") },
    handler: async (ctx, { id }) => {
        return await ctx.db.get(id);
    },
});

export const createSpeakerInternal = internalMutation({
    args: {
        embedding: v.array(v.float64()),
        name: v.optional(v.string()),
    },
    handler: async (ctx, { embedding, name }) => {
        const now = Date.now();
        return await ctx.db.insert("speakers", {
            name,
            embedding,
            lastSeen: now,
            seenCount: 1,
            totalSpeakingTime: 0,
            createdAt: now,
        });
    },
});

export const updateSpeakerLastSeenInternal = internalMutation({
    args: {
        id: v.id("speakers"),
        speakingTime: v.optional(v.float64()),
    },
    handler: async (ctx, { id, speakingTime }) => {
        const speaker = await ctx.db.get(id);
        if (!speaker) return;

        await ctx.db.patch(id, {
            lastSeen: Date.now(),
            seenCount: speaker.seenCount + 1,
            totalSpeakingTime: speaker.totalSpeakingTime + (speakingTime || 0),
        });
    },
});

// ==================== FACE RECOGNITION ====================

// Find speaker by face embedding
export const findSpeakerByFace = action({
    args: {
        faceEmbedding: v.array(v.float64()),
        threshold: v.optional(v.float64()),
    },
    handler: async (ctx, { faceEmbedding, threshold = 0.6 }) => {
        // Vector search on face embeddings
        const results = await ctx.vectorSearch("speakers", "by_face_embedding", {
            vector: faceEmbedding,
            limit: 1,
        });

        // For face embeddings, lower distance = better match
        // Convex returns similarity score (higher = better)
        if (results.length > 0 && results[0]._score >= (1 - threshold)) {
            const speaker = await ctx.runQuery(internal.speakers.getSpeakerInternal, {
                id: results[0]._id
            });
            return {
                found: true,
                speakerId: results[0]._id,
                speaker,
                score: results[0]._score,
            };
        }

        return { found: false, speakerId: null, speaker: null, score: 0 };
    },
});

// Update speaker with face embedding
export const updateSpeakerFace = mutation({
    args: {
        id: v.id("speakers"),
        faceEmbedding: v.array(v.float64()),
    },
    handler: async (ctx, { id, faceEmbedding }) => {
        await ctx.db.patch(id, { faceEmbedding });
    },
});
