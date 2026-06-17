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
- languages: English, Russian
- focus: when to use a technique and its tradeoffs, not derivations

- reading_style: result-first (find main theorem, then unpack backwards)
- explanation_style: give scaffold/structure, I handle details myself
- error_handling: direct correction — say 'no, it's X', don't waste time
- pace: fast — skip anything I'm expected to know
- tone_note: no over-explaining — respect that I know my basics
- background_gaps: skip prerequisites — I'll look them up myself if needed
- session_state: ready for depth — has time and energy

## Anna
- role: аспирант, математика, 2й год
- level: intermediate
- interests: теория вероятностей, статистика, математический анализ
- languages: Russian
- focus: понять идею и интуицию за доказательством, не заучить его

- reading_style: overview → details (start with structure, then dive in)
- explanation_style: concrete example first, then generalise
- error_handling: soft — acknowledge, correct quietly, move on without dwelling
- pace: medium — I'll flag if I'm lost
- tone_note: never say 'this is simple/obvious' — just explain it
- new_terms: introduce one term at a time, not all at once
- background_gaps: brief prerequisite recap, then continue
- session_state: ready for depth — has time and energy

Дополнительно: Анна склонна тревожиться когда не понимает, ей важно слышать что застрять в этом месте — нормально. Хорошо реагирует на структуру вида «сначала поймём X, потом Y». Не любит когда её торопят.

## Dima
- role: студент бакалавриата, 3й курс, физфак
- level: beginner
- interests: машинное обучение, хочет разобраться с нуля
- languages: Russian
- focus: понять базовые концепции, не теряться в обозначениях

- reading_style: linear reading, stops at blockers
- explanation_style: explain from scratch even if I seem to know it
- error_handling: leading question so I find the error myself
- pace: slow, with repetition — depth over breadth
- tone_note: check understanding often — don't just lecture and move on
- new_terms: give a glossary / notation list before diving in
- background_gaps: full prerequisite explanation before continuing
- session_state: ready for depth — has time and energy

Дополнительно: Дима легко перегружается когда в одном абзаце много новых терминов сразу. Нужно явно разделять «что мы уже знаем» и «что вводим сейчас». Полезны аналогии из физики. Если кивает но не может пересказать — значит не понял, надо замедлиться.

## Lena
- role: ML-инженер, 3 года в индустрии
- level: intermediate
- interests: практическое применение, архитектуры, быстрый старт в новой теме
- languages: Russian, English
- focus: как это работает на практике и где применяется — не теория ради теории

- reading_style: result-first (find main theorem, then unpack backwards)
- explanation_style: concrete example first, then generalise
- error_handling: direct correction — say 'no, it's X', don't waste time
- pace: fast — skip anything I'm expected to know
- tone_note: concise — skip preamble and filler
- new_terms: context before definition — show WHY the term exists first
- background_gaps: brief prerequisite recap, then continue
- session_state: short on time — give only the essentials

Дополнительно: Лена раздражается от длинных теоретических вступлений. Лучший формат для неё — сразу показать что это решает на практике, потом формальное определение. Примеры кода или реальных систем работают лучше абстрактных математических конструкций.
