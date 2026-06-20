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

## Rita
- role: student
- level: intermediate
- interests: practice
- languages: English

<!-- learning profile (generated) -->
- reading_style: overview → details (start with structure, then dive in)
- explanation_style: explain from scratch even if I seem to know it
- error_handling: direct correction — say 'no, it's X', don't waste time
- pace: medium — I'll flag if I'm lost
- tone_note: concise — skip preamble and filler
- focus: practical — show me how this is used in code or real problems
- new_terms: give a glossary / notation list before diving in
- background_gaps: full prerequisite explanation before continuing

- session_state: short on time — give only the essentials
