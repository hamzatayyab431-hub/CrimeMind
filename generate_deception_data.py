import pandas as pd
import random

print("🧬 Initializing Advanced Multi-Class Deception Dataset Generator...")

# --- LINGUISTIC BUILDING BLOCKS ---
times = ["8:00 AM", "9:15 AM", "noon", "1:30 PM", "3:45 PM", "5:00 PM", "7:20 PM", "midnight", "2:00 AM", "early morning", "late at night"]
locations = ["the office", "home", "the grocery store", "the gas station", "my friend's apartment", "the parking garage", "the diner downtown", "the subway station", "the warehouse", "the gym"]
actions = ["watching a movie", "eating dinner", "working on a report", "sleeping", "driving", "talking on the phone", "fixing my car", "walking my dog", "waiting for a cab"]
people = ["my wife", "my boss", "my roommate", "a colleague", "my brother", "the cashier", "the security guard"]
objects = ["the black bag", "the car keys", "the laptop", "the money", "the documents", "the weapon", "the safe"]

# CLASS 0: TRUTHFUL (Direct, First-Person, High Detail)
truthful_templates = [
    "I was at {location} around {time}.",
    "I left {location} at {time} and drove straight to {location}.",
    "I have never seen {object} in my entire life.",
    "I was {action} at {time}.",
    "I swiped my keycard at {location} at exactly {time}.",
    "I saw him leave holding {object}.",
    "I was with {people} at {location} while I was {action}.",
    "My car was parked at {location}. I didn't go anywhere.",
    "I didn't take {object}, you can check the cameras.",
    "I arrived at {location} at {time} and immediately started {action}."
]

# CLASS 1: EVASIVE (Memory loss, stalling, fillers, vague)
evasive_templates = [
    "I can't really remember where I was at {time}.",
    "I might have been at {location}, but my memory is foggy.",
    "Why does it matter if I was {action}?",
    "I sort of recall {action}, but I'm not sure.",
    "Maybe I was near {location}? It's hard to say.",
    "I guess I could have seen {object}, but I don't remember.",
    "Let me think... I was probably at {location}.",
    "I don't know anything about {object}.",
    "I was doing a lot of things around {time}.",
    "Could we talk about this later? My head hurts."
]
evasive_fillers = ["Um, ", "Uh, ", "Well, ", "You know, ", "Let's see... "]

# CLASS 2: DEFENSIVE (Hostile, attacking, victimizing)
defensive_templates = [
    "You have no right to ask me about {location}!",
    "Are you accusing me of taking {object}?",
    "I demand to speak to my lawyer. I wasn't at {location}!",
    "Why are you targeting me? I was just {action}!",
    "This is harassment! I didn't even know {people}!",
    "You cops are all the same. I don't know anything about {object}!",
    "Prove it! You can't prove I was at {location} at {time}!",
    "I'm not answering any more questions about {object}.",
    "Am I under arrest? If not, I'm leaving.",
    "I told you already, I was {action}! Leave me alone!"
]

# CLASS 3: FABRICATED/OVER-JUSTIFIED (Too much detail, third-person distancing, unnatural)
fabricated_templates = [
    "The {object} might have been moved, but it certainly wasn't by me.",
    "Someone could say they saw me at {location}, but they would be entirely mistaken.",
    "Why would I even want {object}? I am a very successful person.",
    "Because I was extremely tired, I had to go to {location} to start {action}.",
    "The unfortunate events at {location} have absolutely nothing to do with me.",
    "I was actively {action} and minding my own personal business.",
    "That man took {object}, not me. He looked very suspicious.",
    "To tell you the honest truth, I was far away from {location}.",
    "I would never do such a thing at {time}. It goes against my morals.",
    "I was nowhere near the vicinity of {location}."
]
fabricated_qualifiers = ["Honestly, ", "To tell you the truth, ", "Frankly, ", "Believe me, ", "I swear, "]

def generate_statement(category):
    if category == 0:
        template = random.choice(truthful_templates)
        statement = template.format(time=random.choice(times), location=random.choice(locations), action=random.choice(actions), people=random.choice(people), object=random.choice(objects))
        if random.random() < 0.2 and "at" in statement:
            statement = statement.replace("at", "at exactly")
            
    elif category == 1:
        template = random.choice(evasive_templates)
        statement = template.format(time=random.choice(times), location=random.choice(locations), action=random.choice(actions), people=random.choice(people), object=random.choice(objects))
        if random.random() < 0.6:
            # Check if statement starts with a letter before applying upper/lower case logic
            if len(statement) > 0 and statement[0].isalpha() and statement[0].isupper():
                statement = random.choice(evasive_fillers) + statement[0].lower() + statement[1:]
            else:
                statement = random.choice(evasive_fillers) + statement
            
    elif category == 2:
        template = random.choice(defensive_templates)
        statement = template.format(time=random.choice(times), location=random.choice(locations), action=random.choice(actions), people=random.choice(people), object=random.choice(objects))
        
    elif category == 3:
        template = random.choice(fabricated_templates)
        statement = template.format(time=random.choice(times), location=random.choice(locations), action=random.choice(actions), people=random.choice(people), object=random.choice(objects))
        if random.random() < 0.4:
            if len(statement) > 0 and statement[0].isalpha() and statement[0].isupper():
                statement = random.choice(fabricated_qualifiers) + statement[0].lower() + statement[1:]
            else:
                statement = random.choice(fabricated_qualifiers) + statement

    # Clean up grammar
    if len(statement) > 0:
        statement = statement[0].upper() + statement[1:]
    return statement

data = []
# Generate 8,000 records
for i in range(8000):
    cat = i % 4
    data.append({"statement": generate_statement(cat), "label": cat})

df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv("deception_data.csv", index=False)
print(f"✅ Generated deception_data.csv with {len(df)} multi-class records!")
