# Users

> Agent 3 reads the **full** profile of each user and tailors every lesson accordingly.
> The parser keys off the `##` headings and `- key: value` lines. Recognized keys
> (all optional except a `##` name): `role, level, interests, tone, age, region,
> education, experience, languages, focus`. Anything else you write is still seen by
> the model via the raw profile text — so feel free to add free-form notes.

## Mike
- role: Senior ML Engineer
- level: expert
- interests: production systems, scaling, evaluation, tradeoffs
- tone: short, with math analogies and examples
- region: Russia
- experience: 8 years building ML platforms
- languages: English
- focus: when to use a technique and its tradeoffs, not derivations

- reading_style: result-first (find main theorem, then unpack backwards)
- explanation_style: give scaffold/structure, I handle details myself
- error_handling: direct correction — say 'no, it's X', don't waste time
- pace: fast — skip anything I'm expected to know
- tone_note: no over-explaining — respect that I know my basics
- background_gaps: skip prerequisites — I'll look them up myself if needed
- session_state: ready for depth — has time and energy

## Anna
- role: PhD student, mathematics, 2nd year
- level: intermediate
- interests: probability theory, statistics, mathematical analysis
- languages: English
- focus: understand the idea and intuition behind a proof, not memorize it

- reading_style: overview → details (start with structure, then dive in)
- explanation_style: concrete example first, then generalise
- error_handling: soft — acknowledge, correct quietly, move on without dwelling
- pace: medium — I'll flag if I'm lost
- tone_note: never say 'this is simple/obvious' — just explain it
- new_terms: introduce one term at a time, not all at once
- background_gaps: brief prerequisite recap, then continue
- session_state: ready for depth — has time and energy

Additional note: Anna tends to get anxious when she doesn't understand something — it helps to hear that getting stuck here is normal. She responds well to a "first let's understand X, then Y" structure. She dislikes being rushed.

## Dima
- role: undergraduate student, 3rd year, physics department
- level: beginner
- interests: machine learning, wants to learn from the ground up
- languages: English
- focus: understand the basic concepts, not get lost in notation

- reading_style: linear reading, stops at blockers
- explanation_style: explain from scratch even if I seem to know it
- error_handling: leading question so I find the error myself
- pace: slow, with repetition — depth over breadth
- tone_note: check understanding often — don't just lecture and move on
- new_terms: give a glossary / notation list before diving in
- background_gaps: full prerequisite explanation before continuing
- session_state: ready for depth — has time and energy

Additional note: Dima gets overwhelmed easily when a paragraph introduces too many new terms at once. It helps to explicitly separate "what we already know" from "what we're introducing now". Physics analogies work well for him. If he nods but can't repeat it back, he hasn't understood — slow down.

## Lena
- role: ML engineer, 3 years in industry
- level: intermediate
- interests: practical applications, architectures, getting up to speed quickly on a new topic
- languages: English
- focus: how it works in practice and where it's applied — not theory for its own sake

- reading_style: result-first (find main theorem, then unpack backwards)
- explanation_style: concrete example first, then generalise
- error_handling: direct correction — say 'no, it's X', don't waste time
- pace: fast — skip anything I'm expected to know
- tone_note: concise — skip preamble and filler
- new_terms: context before definition — show WHY the term exists first
- background_gaps: brief prerequisite recap, then continue
- session_state: short on time — give only the essentials

Additional note: Lena gets impatient with long theoretical preambles. The best format for her is to show what a thing solves in practice first, then give the formal definition. Code examples or real systems work better for her than abstract mathematical constructs.
