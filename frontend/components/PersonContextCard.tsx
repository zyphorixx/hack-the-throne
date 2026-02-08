import { usePersonContext } from "@/hooks/usePersonContext";
import { User, Clock, MessageSquare, Heart } from "lucide-react";

interface PersonContextCardProps {
    speakerId: string | null;
    speakerName: string | null; // From real-time inference event
}

export default function PersonContextCard({ speakerId, speakerName }: PersonContextCardProps) {
    const { person, isLoading } = usePersonContext(speakerId);

    // If no speaker identified yet, show valid inference name or hidden
    if (!speakerId) {
        if (speakerName && speakerName !== "Unknown") {
            return (
                <div className="absolute top-6 right-6 w-80 bg-black/40 backdrop-blur-md border border-white/10 rounded-2xl p-5 shadow-2xl animate-in fade-in slide-in-from-right-4">
                    <div className="flex items-center gap-3 mb-2">
                        <div className="w-10 h-10 rounded-full bg-indigo-500/20 flex items-center justify-center border border-indigo-500/30">
                            <User className="w-5 h-5 text-indigo-300" />
                        </div>
                        <div>
                            <h2 className="text-xl font-semibold text-white tracking-tight">{speakerName}</h2>
                            <p className="text-sm text-indigo-200/60 font-medium">Identifying...</p>
                        </div>
                    </div>
                </div>
            )
        }
        return null;
    }

    if (isLoading) return null;
    if (!person) return null;

    return (
        <div className="absolute top-6 right-6 w-80 bg-black/60 backdrop-blur-xl border border-white/10 rounded-2xl p-5 shadow-2xl animate-in fade-in slide-in-from-right-4 z-50 transition-all duration-300">
            {/* Header Profile */}
            <div className="flex items-start gap-4 mb-4">
                <div className="relative">
                    {person.photoUrl ? (
                        <img src={person.photoUrl} alt={person.name} className="w-14 h-14 rounded-full object-cover border-2 border-indigo-500/50 shadow-lg" />
                    ) : (
                        <div className="w-14 h-14 rounded-full bg-gradient-to-br from-indigo-600 to-purple-700 flex items-center justify-center border-2 border-white/10 shadow-lg">
                            <span className="text-xl font-bold text-white">{person.name?.charAt(0) || "?"}</span>
                        </div>
                    )}
                    <div className="absolute -bottom-1 -right-1 bg-green-500 w-4 h-4 rounded-full border-2 border-black" />
                </div>

                <div className="flex-1 min-w-0">
                    <h2 className="text-2xl font-bold text-white tracking-tight truncate">
                        {person.name || "Unknown Person"}
                    </h2>
                    {person.relationship && (
                        <div className="flex items-center gap-1.5 text-indigo-200 mt-0.5">
                            <Heart className="w-3.5 h-3.5 fill-indigo-400/20" />
                            <span className="text-sm font-medium">{person.relationship}</span>
                        </div>
                    )}
                </div>
            </div>

            {/* Stats Row */}
            <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="bg-white/5 rounded-xl p-2.5 border border-white/5">
                    <div className="flex items-center gap-1.5 text-gray-400 text-xs uppercase font-bold tracking-wider mb-0.5">
                        <Clock className="w-3 h-3" />
                        Last Seen
                    </div>
                    <div className="text-sm font-medium text-white">
                        {person.lastSeenText}
                    </div>
                </div>
                <div className="bg-white/5 rounded-xl p-2.5 border border-white/5">
                    <div className="flex items-center gap-1.5 text-gray-400 text-xs uppercase font-bold tracking-wider mb-0.5">
                        <MessageSquare className="w-3 h-3" />
                        Chats
                    </div>
                    <div className="text-sm font-medium text-white">
                        {person.conversationCount} recorded
                    </div>
                </div>
            </div>

            {/* Description / Notes */}
            {person.description && (
                <div className="mb-4 bg-indigo-900/20 rounded-xl p-3 border border-indigo-500/20">
                    <p className="text-sm text-indigo-100 leading-relaxed">
                        {person.description}
                    </p>
                </div>
            )}

            {/* Recent Topic Hint */}
            {person.recentConversations && person.recentConversations.length > 0 && (
                <div className="space-y-2">
                    <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest pl-1">Latest Topic</h3>
                    <div className="bg-white/5 hover:bg-white/10 transition-colors p-3 rounded-xl border border-white/5 group cursor-default">
                        <p className="text-sm text-gray-300 leading-snug line-clamp-3">
                            "{person.recentConversations[0].transcript.substring(0, 100)}..."
                        </p>
                    </div>
                </div>
            )}

            {/* AI Assistant Hint (Placeholder for now) */}
            <div className="mt-4 pt-3 border-t border-white/10 flex items-center gap-2">
                <div className="w-6 h-6 rounded-full bg-emerald-500/20 flex items-center justify-center">
                    <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
                </div>
                <p className="text-xs font-medium text-emerald-300">AI Memory Active</p>
            </div>
        </div>
    );
}
