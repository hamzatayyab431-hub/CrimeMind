import pandas as pd
import random

random.seed(42)

minor_clues = [
    "footprints near the window", "broken coffee mug on the floor",
    "unlocked back door", "open window in the hallway", "missing pen from the desk",
    "overturned chair in the kitchen", "muddy shoe prints on the carpet",
    "torn piece of fabric on the fence", "half eaten food on the table",
    "lights left on in the basement", "scratches on the door lock",
    "cigarette butt near the entrance", "displaced flower pot outside",
    "wet umbrella near the coat rack", "handwriting on a sticky note",
    "empty glass on the nightstand", "displaced rug near the fireplace",
    "open drawer in the bedroom", "cracked window pane in the study",
    "dog barking reported by neighbor", "missing keys from the hook",
    "chair moved from its usual position", "curtains partially drawn",
    "newspaper left on the doorstep", "smudge marks on the windowsill",
    "faint smell of perfume in the hallway", "coat left on the floor",
    "drawer left slightly open", "phone charger unplugged oddly",
    "trash bin knocked over outside", "garden gate left unlatched",
    "water glass knocked on its side", "book left open on the table",
    "loose floorboard near the entrance", "paint scratches on the wall",
    "misplaced keys on the kitchen counter", "wet footprints on the tile",
    "picture frame slightly tilted", "dust disturbed on the shelf",
    "milk left out on the counter", "calendar page torn off",
]

major_clues = [
    "blood stains found on the kitchen knife", "cyanide vial discovered in the bin",
    "strangled victim found in the library", "gunshot residue on the suspect hands",
    "threatening letter with fingerprints", "poison detected in the wine glass",
    "body found with stab wounds in the cellar", "victim was pushed from the balcony",
    "arsenic found in the victims tea", "masked intruder caught on cctv",
    "ransom note written in victims blood", "murder weapon hidden under floorboards",
    "victim strangled with a rope", "forced entry through basement window",
    "suspects dna found under victims fingernails", "victim poisoned with rare toxin",
    "financial fraud linked to suspects account", "eyewitness saw suspect flee scene",
    "secret compartment found with stolen documents", "accelerant found at fire origin",
    "victim shows signs of blunt force trauma", "illegal firearm found in suspects car",
    "forged signature on the victims will", "surveillance footage deleted remotely",
    "suspect found with victims stolen jewelry", "victim was drugged before death",
    "hair sample matches suspect dna profile", "victim received death threat last week",
    "suspects fingerprints on the murder weapon", "toxicology confirms deliberate poisoning",
    "encrypted messages found on suspects phone", "victim tied up before being killed",
    "witness heard screaming from the property", "victim life insurance recently increased",
    "suspect lied about alibi on camera", "victim had restraining order against suspect",
    "explosive residue on suspects jacket", "victim was blackmailing the suspect",
    "hidden camera recorded the crime", "suspect passport shows false identity",
]

# Generate with augmentation — slight variations per row
templates_minor = [
    "{}", "officers noticed {}", "investigators found {}", 
    "evidence suggests {}", "witness reported {}"
]
templates_major = [
    "{}", "forensics confirmed {}", "detective uncovered {}",
    "lab results revealed {}", "crime scene showed {}"
]

rows = []
for _ in range(600):
    clue = random.choice(minor_clues)
    tmpl = random.choice(templates_minor)
    rows.append({"description": tmpl.format(clue), "severity": 0})

    clue = random.choice(major_clues)
    tmpl = random.choice(templates_major)
    rows.append({"description": tmpl.format(clue), "severity": 1})

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("crime_data.csv", index=False)
print(f"Generated {len(df)} rows -> crime_data.csv")