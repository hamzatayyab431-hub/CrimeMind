import pandas as pd
import random

print("🧬 Initializing Advanced Multi-Class Evidence Generator...")

# Class 0: Trace/Circumstantial Evidence
trace_clues = [
    "muddy footprints near the window", "broken coffee mug on the floor",
    "unlocked back door", "open window in the hallway", "missing pen from the desk",
    "overturned chair in the kitchen", "torn piece of fabric on the fence",
    "cigarette butt near the entrance", "wet umbrella near the coat rack",
    "faint smell of perfume in the hallway", "dust disturbed on the shelf",
    "misplaced keys on the kitchen counter", "calendar page torn off",
    "scratches on the door lock", "displaced rug near the fireplace"
]

# Class 1: Biological/Chemical Evidence
bio_clues = [
    "blood stains found on the kitchen floor", "cyanide vial discovered in the bin",
    "poison detected in the wine glass", "arsenic found in the victims tea",
    "suspects dna found under victims fingernails", "victim poisoned with rare toxin",
    "hair sample matches suspect dna profile", "toxicology confirms deliberate poisoning",
    "saliva found on the discarded cigarette", "blood spatter pattern on the wall",
    "traces of chloroform on the rag", "unknown biological fluid on the bedsheets"
]

# Class 2: Weaponry/Violent Evidence
weapon_clues = [
    "gunshot residue on the suspect hands", "strangled victim found in the library",
    "body found with stab wounds in the cellar", "murder weapon hidden under floorboards",
    "victim strangled with a rope", "victim shows signs of blunt force trauma",
    "illegal firearm found in suspects car", "suspects fingerprints on the murder weapon",
    "explosive residue on suspects jacket", "bloody knife found in the dumpster",
    "spent shell casings found at the scene", "heavy blunt object with blood found"
]

# Class 3: Digital/Financial Evidence
digital_clues = [
    "financial fraud linked to suspects account", "secret compartment found with stolen documents",
    "forged signature on the victims will", "surveillance footage deleted remotely",
    "encrypted messages found on suspects phone", "victim life insurance recently increased",
    "suspect lied about alibi on camera", "victim was blackmailing the suspect via email",
    "hidden camera recorded the crime", "suspect passport shows false identity",
    "deleted text messages recovered from victims phone", "unauthorized bank transfer at 3 AM"
]

templates = [
    "{}", "officers noticed {}", "investigators found {}", 
    "evidence suggests {}", "forensics confirmed {}", 
    "detective uncovered {}", "lab results revealed {}", "crime scene showed {}"
]

rows = []

# Generate balanced dataset
for _ in range(1500):
    rows.append({"description": random.choice(templates).format(random.choice(trace_clues)), "category": 0, "label": "Trace Evidence"})
    rows.append({"description": random.choice(templates).format(random.choice(bio_clues)), "category": 1, "label": "Biological Evidence"})
    rows.append({"description": random.choice(templates).format(random.choice(weapon_clues)), "category": 2, "label": "Weaponry"})
    rows.append({"description": random.choice(templates).format(random.choice(digital_clues)), "category": 3, "label": "Digital Evidence"})

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("crime_data.csv", index=False)
print(f"✅ Generated {len(df)} multi-class rows -> crime_data.csv")