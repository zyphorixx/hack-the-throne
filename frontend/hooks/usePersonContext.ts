import { useQuery } from "convex/react";
import { api } from "../convex/_generated/api";
import { Id } from "../convex/_generated/dataModel";

export function usePersonContext(speakerId: string | null) {
    // convex ID validation: skip if null or if it's a local Pyannote ID (starts with "speaker_")
    const isInvalidId = !speakerId || speakerId.startsWith("speaker_");

    const data = useQuery(api.context.getPersonContext,
        isInvalidId ? "skip" : { speakerId: speakerId as Id<"speakers"> }
    );

    return {
        person: data,
        isLoading: data === undefined,
        isError: data === null && speakerId !== null,
    };
}
