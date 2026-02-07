import { mutation } from "./_generated/server";

export const reset = mutation({ handler: async (ctx) => { const speakers = await ctx.db.query('speakers').collect(); for (const s of speakers) await ctx.db.delete(s._id); const convs = await ctx.db.query('conversations').collect(); for (const c of convs) await ctx.db.delete(c._id); } });
