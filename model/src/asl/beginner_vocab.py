"""Candidate ASL-1 beginner vocabulary.

A broad candidate pool of common ASL-1 signs grouped by pedagogical category.
Phase 0 intersects this with the ASL Citizen data audit and keeps only signs
that clear the per-class data bar; the frozen final list is <= this pool.

Glosses are uppercased to match ASL Citizen's gloss convention; the audit does
case-insensitive matching and reports candidates present/absent in the dataset.
"""

CANDIDATE_VOCAB = {
    "greetings_social": [
        "HELLO", "GOODBYE", "PLEASE", "THANK-YOU", "SORRY", "YES", "NO",
        "NAME", "MEET", "FRIEND", "HELP", "WELCOME",
    ],
    "people_family": [
        "MOTHER", "FATHER", "SISTER", "BROTHER", "FAMILY", "BABY", "BOY",
        "GIRL", "MAN", "WOMAN", "CHILD", "TEACHER", "STUDENT",
    ],
    "pronouns_question": [
        "WHO", "WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHICH",
    ],
    "colors": [
        "RED", "BLUE", "GREEN", "YELLOW", "BLACK", "WHITE", "ORANGE",
        "PURPLE", "PINK", "BROWN",
    ],
    "food_drink": [
        "EAT", "DRINK", "WATER", "MILK", "COFFEE", "BREAD", "APPLE",
        "BANANA", "HUNGRY", "FOOD", "COOK",
    ],
    "common_verbs": [
        "GO", "COME", "WANT", "NEED", "LIKE", "KNOW", "UNDERSTAND", "LEARN",
        "READ", "WRITE", "WORK", "PLAY", "SLEEP", "SEE", "GIVE", "MAKE",
        "FINISH", "STOP", "START",
    ],
    "school_objects": [
        "BOOK", "PAPER", "PEN", "PENCIL", "COMPUTER", "PHONE", "TABLE",
        "CHAIR", "DOOR", "HOUSE", "SCHOOL", "CAR",
    ],
    "time_days": [
        "DAY", "NIGHT", "MORNING", "TODAY", "TOMORROW", "YESTERDAY", "WEEK",
        "MONTH", "YEAR", "TIME", "NOW",
    ],
    "descriptors_feelings": [
        "GOOD", "BAD", "HAPPY", "SAD", "ANGRY", "TIRED", "SICK", "BIG",
        "SMALL", "HOT", "COLD", "NEW", "OLD",
    ],
    "animals": [
        "DOG", "CAT", "BIRD", "FISH", "HORSE", "COW",
    ],
}


def all_candidates():
    seen, out = set(), []
    for words in CANDIDATE_VOCAB.values():
        for w in words:
            if w not in seen:
                seen.add(w)
                out.append(w)
    return out


if __name__ == "__main__":
    cands = all_candidates()
    print(f"{len(cands)} candidate beginner signs across "
          f"{len(CANDIDATE_VOCAB)} categories")
