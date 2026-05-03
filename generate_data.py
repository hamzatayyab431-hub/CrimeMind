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
    "scratches on the door lock", "displaced rug near the fireplace",
    "footprints at the scene", "door was left open", "window was broken",
    "something is missing", "clues found near the entrance", "signs of a struggle",
    "chair knocked over", "object out of place", "trail of footsteps",
    "evidence of a forced entry", "marks on the floor", "debris in the hallway"
]

# Class 1: Biological/Chemical Evidence
bio_clues = [
    "blood stains found on the kitchen floor", "cyanide vial discovered in the bin",
    "poison detected in the wine glass", "arsenic found in the victims tea",
    "suspects dna found under victims fingernails", "victim poisoned with rare toxin",
    "hair sample matches suspect dna profile", "toxicology confirms deliberate poisoning",
    "saliva found on the discarded cigarette", "blood spatter pattern on the wall",
    "traces of chloroform on the rag", "unknown biological fluid on the bedsheets",
    "there is blood on the floor", "blood was found at the scene", "blood everywhere",
    "victim was poisoned", "poison was used", "dna evidence found", "dna sample collected",
    "biological material recovered", "toxin detected", "chemical substance found",
    "victim had been drugged", "traces of poison", "blood on the wall", "dna match confirmed"
]

# Class 2: Weaponry/Violent Evidence
weapon_clues = [
    "gunshot residue on the suspect hands", "strangled victim found in the library",
    "body found with stab wounds in the cellar", "murder weapon hidden under floorboards",
    "victim strangled with a rope", "victim shows signs of blunt force trauma",
    "illegal firearm found in suspects car", "suspects fingerprints on the murder weapon",
    "explosive residue on suspects jacket", "bloody knife found in the dumpster",
    "spent shell casings found at the scene", "heavy blunt object with blood found",
    "he has a gun", "suspect had a gun", "they found a gun", "gun was discovered",
    "he is armed with a gun", "she pulled out a gun", "a loaded gun was found",
    "suspect carrying a weapon", "he has a knife", "she had a knife",
    "knife was found at the scene", "he was holding a weapon", "armed suspect",
    "weapon found nearby", "firearm recovered", "pistol found at scene",
    "rifle discovered", "shotgun at the scene", "bomb was planted", "explosive device found",
    "he shot the victim", "gunshot wound", "stabbed with a knife", "victim was shot",
    "he attacked with a weapon", "assault with a deadly weapon", "armed robbery occurred"
]

# Class 3: Digital/Financial Evidence
digital_clues = [
    "financial fraud linked to suspects account", "secret compartment found with stolen documents",
    "forged signature on the victims will", "surveillance footage deleted remotely",
    "encrypted messages found on suspects phone", "victim life insurance recently increased",
    "suspect lied about alibi on camera", "victim was blackmailing the suspect via email",
    "hidden camera recorded the crime", "suspect passport shows false identity",
    "deleted text messages recovered from victims phone", "unauthorized bank transfer at 3 AM",
    "evidence of money fraud on the computer", "large amounts of stolen money",
    "massive credit card fraud detected", "embezzled funds and money fraud",
    "money was stolen", "fraud was committed", "he stole money", "they committed fraud",
    "bank account was hacked", "digital evidence recovered", "phone records found",
    "camera footage exists", "emails were deleted", "data was stolen",
    "financial records tampered", "online transaction suspicious", "hacked into the system"
]

templates = [
    "{}", "officers noticed {}", "investigators found {}", 
    "evidence suggests {}", "forensics confirmed {}", 
    "detective uncovered {}", "lab results revealed {}", "crime scene showed {}"
]

# Add short-form raw clues directly (no template wrapping) to ensure they appear as-is
short_weapon = ["he has a gun", "she had a knife", "suspect armed", "gun found", "weapon at scene", "he was armed", "firearm recovered", "knife at scene"]
short_bio    = ["blood on floor", "blood found", "poison used", "dna found", "victim poisoned", "blood at scene", "chemical evidence"]
short_digital= ["money fraud", "he stole money", "fraud detected", "money was stolen", "bank fraud", "credit card fraud", "financial crime"]
short_trace  = ["footprints found", "door was open", "window broken", "chair knocked over", "forced entry", "signs of struggle"]

rows = []

# Generate balanced dataset
for _ in range(2000):
    rows.append({"description": random.choice(templates).format(random.choice(trace_clues)), "category": 0, "label": "Trace Evidence"})
    rows.append({"description": random.choice(templates).format(random.choice(bio_clues)), "category": 1, "label": "Biological Evidence"})
    rows.append({"description": random.choice(templates).format(random.choice(weapon_clues)), "category": 2, "label": "Weaponry"})
    rows.append({"description": random.choice(templates).format(random.choice(digital_clues)), "category": 3, "label": "Digital Evidence"})

# Inject short-form phrases directly (no template)
for _ in range(300):
    rows.append({"description": random.choice(short_weapon), "category": 2, "label": "Weaponry"})
    rows.append({"description": random.choice(short_bio),    "category": 1, "label": "Biological Evidence"})
    rows.append({"description": random.choice(short_digital),"category": 3, "label": "Digital Evidence"})
    rows.append({"description": random.choice(short_trace),  "category": 0, "label": "Trace Evidence"})

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("crime_data.csv", index=False)
print(f"✅ Generated {len(df)} multi-class rows -> crime_data.csv")