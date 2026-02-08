import { v } from "convex/values";
import { mutation, query } from "./_generated/server";

// ==================== QUERIES ====================

// Get full person context including recent conversations
export const getPersonContext = query({
    args: { speakerId: v.id("speakers") },
    handler: async (ctx, { speakerId }) => {
        const speaker = await ctx.db.get(speakerId);
        if (!speaker) return null;

        // Get recent conversations
        const recentConversations = await ctx.db
            .query("conversations")
            .withIndex("by_speaker", (q) => q.eq("speakerId", speakerId))
            .order("desc")
            .take(10);

        // Calculate time since last seen
        const lastSeenDate = new Date(speaker.lastSeen);
        const now = new Date();
        const diffMs = now.getTime() - lastSeenDate.getTime();
        const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
        const diffHours = Math.floor(diffMs / (1000 * 60 * 60));

        let lastSeenText = "Just now";
        if (diffDays > 0) {
            lastSeenText = `${diffDays} day${diffDays > 1 ? "s" : ""} ago`;
        } else if (diffHours > 0) {
            lastSeenText = `${diffHours} hour${diffHours > 1 ? "s" : ""} ago`;
        }

        return {
            ...speaker,
            lastSeenText,
            recentConversations,
            conversationCount: recentConversations.length,
        };
    },
});

// Get recent conversations for a speaker
export const getConversations = query({
    args: {
        speakerId: v.id("speakers"),
        limit: v.optional(v.number()),
    },
    handler: async (ctx, { speakerId, limit = 20 }) => {
        return await ctx.db
            .query("conversations")
            .withIndex("by_speaker", (q) => q.eq("speakerId", speakerId))
            .order("desc")
            .take(limit);
    },
});

// Get all recent conversations
export const getRecentConversations = query({
    args: { limit: v.optional(v.number()) },
    handler: async (ctx, { limit = 50 }) => {
        const conversations = await ctx.db
            .query("conversations")
            .withIndex("by_timestamp")
            .order("desc")
            .take(limit);

        // Attach speaker info
        const withSpeakers = await Promise.all(
            conversations.map(async (conv) => {
                const speaker = await ctx.db.get(conv.speakerId);
                return {
                    ...conv,
                    speakerName: speaker?.name || "Unknown",
                };
            })
        );

        return withSpeakers;
    },
});

// ==================== MUTATIONS ====================

// Save a new conversation
export const saveConversation = mutation({
    args: {
        speakerId: v.id("speakers"),
        transcript: v.string(),
        durationSeconds: v.float64(),
        summary: v.optional(v.string()),
        topics: v.optional(v.array(v.string())),
    },
    handler: async (ctx, args) => {
        return await ctx.db.insert("conversations", {
            ...args,
            timestamp: Date.now(),
        });
    },
});

// Update conversation with LLM-generated summary
export const updateConversationSummary = mutation({
    args: {
        id: v.id("conversations"),
        summary: v.string(),
        topics: v.optional(v.array(v.string())),
        sentiment: v.optional(v.string()),
    },
    handler: async (ctx, { id, ...updates }) => {
        await ctx.db.patch(id, updates);
    },
});
