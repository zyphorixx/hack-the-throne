
"use client";

import { useQuery } from "convex/react";
import { api } from "@/convex/_generated/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Users, User } from "lucide-react";

export function PeopleGrid() {
    const speakers = useQuery(api.speakers.listSpeakers, {});

    if (!speakers) {
        return <div className="p-4">Loading people...</div>;
    }

    const knownPeople = speakers.filter(s => s.name && s.name !== "Unknown" && s.name !== "Unknown Person");

    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Users className="w-5 h-5" />
                    People Directory
                </CardTitle>
            </CardHeader>
            <CardContent>
                {knownPeople.length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">
                        <User className="w-10 h-10 mx-auto mb-2 opacity-50" />
                        <p>No known people yet.</p>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {knownPeople.map((person) => (
                            <div key={person._id} className="flex items-center space-x-4 rounded-lg border p-4 hover:bg-muted/50 transition-colors">
                                <Avatar className="h-12 w-12 border-2 border-primary/20">
                                    <AvatarImage src={person.photoUrl} />
                                    <AvatarFallback className="text-lg bg-primary/10 text-primary">
                                        {person.name?.[0]}
                                    </AvatarFallback>
                                </Avatar>
                                <div className="flex-1 space-y-1">
                                    <div className="flex items-center justify-between">
                                        <p className="text-sm font-medium leading-none">{person.name}</p>
                                        {person.relationship && (
                                            <Badge variant="secondary" className="text-xs">
                                                {person.relationship}
                                            </Badge>
                                        )}
                                    </div>
                                    <p className="text-xs text-muted-foreground line-clamp-1">
                                        {person.description || "No description available"}
                                    </p>
                                    <p className="text-xs text-muted-foreground">
                                        Seen {person.seenCount} times
                                    </p>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </CardContent>
        </Card>
    );
}
