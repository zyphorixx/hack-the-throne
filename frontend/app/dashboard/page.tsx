
"use client";

import { useQuery } from "convex/react";
import { api } from "@/convex/_generated/api";
import { StatsCard } from "@/components/dashboard/StatsCard";
import { ActivityFeed } from "@/components/dashboard/ActivityFeed";
import { PeopleGrid } from "@/components/dashboard/PeopleGrid";
import { Users, MessageSquare, Clock, Heart } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";

export default function DashboardPage() {
    const speakers = useQuery(api.speakers.listSpeakers, {});
    const conversations = useQuery(api.context.getRecentConversations, { limit: 100 });

    const totalInteractions = conversations?.length || 0;
    const knownPeopleCount = speakers?.filter(s => s.name && s.name !== "Unknown" && s.name !== "Unknown Person").length || 0;

    // Calculate total speaking time in hours
    const totalMinutes = speakers?.reduce((acc, curr) => acc + (curr.totalSpeakingTime || 0), 0) || 0;
    const totalHours = Math.round(totalMinutes / 60);

    return (
        <div className="min-h-screen bg-background p-8">
            <div className="max-w-7xl mx-auto space-y-8">
                {/* Header */}
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-3xl font-bold tracking-tight">Caregiver Dashboard</h1>
                        <p className="text-muted-foreground">Monitor patient interactions and activity</p>
                    </div>
                    <Link href="/">
                        <Button variant="outline" className="gap-2">
                            <ArrowLeft className="w-4 h-4" />
                            Back to Live View
                        </Button>
                    </Link>
                </div>

                {/* Stats Row */}
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                    <StatsCard
                        title="Total Interactions"
                        value={totalInteractions.toString()}
                        icon={MessageSquare}
                        description="Recorded conversations"
                    />
                    <StatsCard
                        title="Known People"
                        value={knownPeopleCount.toString()}
                        icon={Users}
                        description="Identified contacts"
                    />
                    <StatsCard
                        title="Social Engagement"
                        value={`${totalHours} hrs`}
                        icon={Heart}
                        description="Total time speaking"
                    />
                    <StatsCard
                        title="Last Activity"
                        value={conversations?.[0] ? new Date(conversations[0].timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : "None"}
                        icon={Clock}
                        description={conversations?.[0] ? "Check activity feed" : "No recent activity"}
                    />
                </div>

                {/* Main Content Grid */}
                <div className="grid gap-8 lg:grid-cols-3">
                    {/* Left Column: People Directory (wider) */}
                    <div className="lg:col-span-2 space-y-8">
                        <PeopleGrid />
                    </div>

                    {/* Right Column: Activity Feed */}
                    <div className="space-y-8">
                        <ActivityFeed />
                    </div>
                </div>
            </div>
        </div>
    );
}
