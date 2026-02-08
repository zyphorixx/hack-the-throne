
"use client";

import { useQuery } from "convex/react";
import { api } from "@/convex/_generated/api";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageSquare, Clock } from "lucide-react";

function formatTimeAgo(timestamp: number) {
    const seconds = Math.floor((Date.now() - timestamp) / 1000);
    if (seconds < 60) return "Just now";
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.floor(hours / 24)}d ago`;
}

export function ActivityFeed() {
    const conversations = useQuery(api.context.getRecentConversations, { limit: 20 });

    if (!conversations) {
        return <div className="p-4 text-center text-muted-foreground">Loading activity...</div>;
    }

    if (conversations.length === 0) {
        return (
            <div className="p-8 text-center text-muted-foreground">
                <MessageSquare className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>No recent conversations recorded.</p>
            </div>
        );
    }

    return (
        <Card className="h-full">
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Clock className="w-5 h-5" />
                    Recent Activity
                </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
                <ScrollArea className="h-[400px] pr-4">
                    <div className="space-y-4 p-4 pt-0">
                        {conversations.map((conv) => (
                            <div key={conv._id} className="flex gap-4 items-start pb-4 border-b last:border-0 last:pb-0">
                                <Avatar className="w-10 h-10 border">
                                    <AvatarFallback>{conv.speakerName[0]}</AvatarFallback>
                                </Avatar>
                                <div className="flex-1 space-y-1">
                                    <div className="flex items-center justify-between">
                                        <p className="text-sm font-medium leading-none">
                                            {conv.speakerName}
                                        </p>
                                        <span className="text-xs text-muted-foreground">
                                            {formatTimeAgo(conv.timestamp)}
                                        </span>
                                    </div>
                                    <p className="text-sm text-muted-foreground line-clamp-2">
                                        {conv.summary || conv.transcript}
                                    </p>
                                </div>
                            </div>
                        ))}
                    </div>
                </ScrollArea>
            </CardContent>
        </Card>
    );
}
