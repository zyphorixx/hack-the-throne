import { mutation } from "./_generated/server";

export const fix = mutation({
    handler: async (ctx) => {
        const speakers = await ctx.db.query("speakers").collect();
        let count = 0;
        for (const s of speakers) {
            if (s.name === "Matt") {
                await ctx.db.patch(s._id, { name: "Mayank" });
                console.log(`Updated speaker ${s._id} from Matt to Mayank`);
                count++;
            }
        }
        return `Fixed ${count} speakers.`;
    },
});
