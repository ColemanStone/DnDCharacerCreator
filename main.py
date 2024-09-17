import random
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image
from diffusers import StableDiffusionPipeline
import torch


class Character:
    def __init__(self, name, race, char_class, strength, dexterity, constitution, intelligence, wisdom, charisma,
                 starting_items, skills_with_damage, background, alignment, additional_equipment, age, gender, height,
                 weight, eye_color, hair_color, skin_color, personality_traits, ideals, bonds, flaws):
        self.name = name
        self.race = race
        self.char_class = char_class
        self.strength = strength
        self.dexterity = dexterity
        self.constitution = constitution
        self.intelligence = intelligence
        self.wisdom = wisdom
        self.charisma = charisma
        self.starting_items = starting_items
        self.skills_with_damage = skills_with_damage
        self.background = background
        self.alignment = alignment
        self.additional_equipment = additional_equipment
        self.age = age
        self.gender = gender
        self.height = height
        self.weight = weight
        self.eye_color = eye_color
        self.hair_color = hair_color
        self.skin_color = skin_color
        self.personality_traits = personality_traits
        self.ideals = ideals
        self.bonds = bonds
        self.flaws = flaws


def roll_stat():
    rolls = [random.randint(1, 6) for _ in range(4)]
    return sum(sorted(rolls)[1:])


def generate_image(character):
    description = generate_character_description(character)
    image = generate_character_image(description)
    if image:
        image.save(f"{character.name}_character_image.png")


def generate_character_description(character):
    return (f"Name: {character.name}, Race: {character.race}, Class: {character.char_class}, Background: {character.background}, "
            f"Alignment: {character.alignment}, Additional Equipment: {character.additional_equipment}, Age: {character.age}, "
            f"Gender: {character.gender}, Height: {character.height}, Weight: {character.weight}, Eye Color: {character.eye_color}, "
            f"Hair Color: {character.hair_color}, Skin Color: {character.skin_color}, Personality Traits: {character.personality_traits}, "
            f"Ideals: {character.ideals}, Bonds: {character.bonds}, Flaws: {character.flaws}")


def generate_character_image(description):
    print(f"Generating image for description: {description}")  # Debug print

    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        image = model(description).images[0]  # Get the first generated image

    return image


def is_suitable(stats, char_class):
    preferred_stats = {
        "Barbarian": ["strength", "constitution"],
        "Bard": ["charisma", "dexterity"],
        "Cleric": ["wisdom", "strength"],
        "Druid": ["wisdom", "intelligence"],
        "Fighter": ["strength", "constitution"],
        "Monk": ["dexterity", "wisdom"],
        "Paladin": ["strength", "charisma"],
        "Ranger": ["dexterity", "wisdom"],
        "Rogue": ["dexterity", "intelligence"],
        "Sorcerer": ["charisma", "constitution"],
        "Warlock": ["charisma", "intelligence"],
        "Wizard": ["intelligence", "wisdom"]
    }
    primary, secondary = preferred_stats[char_class]
    return stats[primary] >= 15 and stats[secondary] >= 13


def roll_stats_for_class(char_class):
    while True:
        stats = {
            "strength": roll_stat(),
            "dexterity": roll_stat(),
            "constitution": roll_stat(),
            "intelligence": roll_stat(),
            "wisdom": roll_stat(),
            "charisma": roll_stat()
        }
        if is_suitable(stats, char_class):
            return stats


def get_starting_items_and_skills(char_class):
    items_and_skills = {
        "Barbarian": (["Great Axe", "Explorer's Pack"], {"Rage": "1d12", "Unarmored Defense": "N/A"}),
        "Bard": (["Lute", "Diplomat's Pack"], {"Spellcasting": "Varies", "Bardic Inspiration": "N/A"}),
        "Cleric": (["Mace", "Priest's Pack"], {"Spellcasting": "Varies", "Divine Domain": "N/A"}),
        "Druid": (["Quarterstaff", "Explorer's Pack"], {"Spellcasting": "Varies", "Wild Shape": "N/A"}),
        "Fighter": (["Longsword", "Dungeoneer's Pack"], {"Fighting Style": "Varies", "Second Wind": "N/A"}),
        "Monk": (["Shortsword", "Explorer's Pack"], {"Martial Arts": "1d6", "Unarmored Defense": "N/A"}),
        "Paladin": (["Warhammer", "Priest's Pack"], {"Divine Sense": "N/A", "Lay on Hands": "N/A"}),
        "Ranger": (["Longbow", "Explorer's Pack"], {"Favored Enemy": "N/A", "Natural Explorer": "N/A"}),
        "Rogue": (["Dagger", "Burglar's Pack"], {"Sneak Attack": "1d6", "Thieves' Cant": "N/A"}),
        "Sorcerer": (["Dagger", "Arcane Focus"], {"Spellcasting": "Varies", "Sorcerous Origin": "N/A"}),
        "Warlock": (["Light Crossbow", "Scholar's Pack"], {"Otherworldly Patron": "N/A", "Pact Magic": "Varies"}),
        "Wizard": (["Spellbook", "Component Pouch"], {"Spellcasting": "Varies", "Arcane Recovery": "N/A"})
    }
    return items_and_skills[char_class]


def create_character():
    name = name_entry.get()
    race = race_var.get()
    char_class = class_var.get()
    background = background_entry.get()
    alignment = alignment_var.get()
    additional_equipment = equipment_entry.get()
    age = age_entry.get()
    gender = gender_var.get()
    height = height_entry.get()
    weight = weight_entry.get()
    eye_color = eye_color_entry.get()
    hair_color = hair_color_entry.get()
    skin_color = skin_color_entry.get()
    personality_traits = personality_traits_entry.get()
    ideals = ideals_entry.get()
    bonds = bonds_entry.get()
    flaws = flaws_entry.get()

    if not name or not race or not char_class or not background or not alignment or not age or not gender or not height or not weight or not eye_color or not hair_color or not personality_traits or not ideals or not bonds or not flaws or not skin_color:
        messagebox.showerror("Input Error", "Please fill in all fields")
        return

    stats = roll_stats_for_class(char_class)
    starting_items, skills_with_damage = get_starting_items_and_skills(char_class)
    character = Character(name, race, char_class, stats["strength"], stats["dexterity"], stats["constitution"],
                          stats["intelligence"], stats["wisdom"], stats["charisma"], starting_items, skills_with_damage,
                          background, alignment, additional_equipment, age, gender, height, weight, eye_color, hair_color,
                          skin_color, personality_traits, ideals, bonds, flaws)
    generate_image(character)
    messagebox.showinfo("Success", f"Character {name} created and saved to image")


def generate_random_name():
    random_names = [
        "Aragorn", "Legolas", "Gimli", "Frodo", "Gandalf", "Boromir", "Samwise", "Pippin", "Merry", "Eowyn",
        "Arwen", "Elrond", "Galadriel", "Thranduil", "Celeborn", "Glorfindel", "Eomer", "Theoden", "Faramir", "Denethor"
    ]
    name_entry.delete(0, tk.END)
    name_entry.insert(0, random.choice(random_names))


def generate_random_character():
    random_names = [
        "Aragorn", "Legolas", "Gimli", "Frodo", "Gandalf", "Boromir", "Samwise", "Pippin", "Merry", "Eowyn",
        "Arwen", "Elrond", "Galadriel", "Thranduil", "Celeborn", "Glorfindel", "Eomer", "Theoden", "Faramir", "Denethor"
    ]
    random_races = ["Human", "Elf", "Dwarf", "Halfling", "Dragonborn", "Gnome", "Half-Elf", "Half-Orc", "Tiefling"]
    random_classes = ["Barbarian", "Bard", "Cleric", "Druid", "Fighter", "Monk", "Paladin", "Ranger", "Rogue", "Sorcerer",
                      "Warlock", "Wizard"]
    random_alignments = ["Lawful Good", "Neutral Good", "Chaotic Good", "Lawful Neutral", "True Neutral", "Chaotic Neutral",
                         "Lawful Evil", "Neutral Evil", "Chaotic Evil"]
    random_genders = ["Male", "Female", "Non-binary", "Other"]

    name_entry.delete(0, tk.END)
    name_entry.insert(0, random.choice(random_names))
    race_var.set(random.choice(random_races))
    class_var.set(random.choice(random_classes))
    background_entry.delete(0, tk.END)
    background_entry.insert(0, "Random Background")
    alignment_var.set(random.choice(random_alignments))
    equipment_entry.delete(0, tk.END)
    equipment_entry.insert(0, "Random Equipment")
    age_entry.delete(0, tk.END)
    age_entry.insert(0, random.randint(18, 100))
    gender_var.set(random.choice(random_genders))
    height_entry.delete(0, tk.END)
    height_entry.insert(0, f"{random.randint(4, 7)}' {random.randint(0, 11)}\"")
    weight_entry.delete(0, tk.END)
    weight_entry.insert(0, random.randint(100, 300))
    eye_color_entry.delete(0, tk.END)
    eye_color_entry.insert(0, "Random Eye Color")
    hair_color_entry.delete(0, tk.END)
    hair_color_entry.insert(0, "Random Hair Color")
    skin_color_entry.delete(0, tk.END)
    skin_color_entry.insert(0, "Random Skin Color")

    # Insert the new block here (around line 45 in your script):
    random_personality_traits = ["Brave", "Cunning", "Honest", "Deceitful", "Impulsive", "Cautious"]
    random_ideals = ["Justice", "Freedom", "Power", "Knowledge", "Wealth", "Glory"]
    random_bonds = ["Family", "Honor", "Friends", "A mentor", "A former lover", "A sacred vow"]
    random_flaws = ["Greedy", "Proud", "Hot-headed", "Cowardly", "Reckless", "Vindictive"]

    personality_traits_entry.delete(0, tk.END)
    personality_traits_entry.insert(0, random.choice(random_personality_traits))
    ideals_entry.delete(0, tk.END)
    ideals_entry.insert(0, random.choice(random_ideals))
    bonds_entry.delete(0, tk.END)
    bonds_entry.insert(0, random.choice(random_bonds))
    flaws_entry.delete(0, tk.END)
    flaws_entry.insert(0, random.choice(random_flaws))


app = tk.Tk()
app.title("DnD Character Maker")

tk.Label(app, text="Name:").grid(row=0, column=0)
name_entry = tk.Entry(app)
name_entry.grid(row=0, column=1)
tk.Button(app, text="Random Name", command=generate_random_name).grid(row=0, column=2)

tk.Label(app, text="Race:").grid(row=1, column=0)
race_var = tk.StringVar()
race_options = ["Human", "Elf", "Dwarf", "Halfling", "Dragonborn", "Gnome", "Half-Elf", "Half-Orc", "Tiefling"]
race_menu = ttk.Combobox(app, textvariable=race_var, values=race_options)
race_menu.grid(row=1, column=1)

tk.Label(app, text="Class:").grid(row=2, column=0)
class_var = tk.StringVar()
class_options = ["Barbarian", "Bard", "Cleric", "Druid", "Fighter", "Monk", "Paladin", "Ranger", "Rogue", "Sorcerer",
                 "Warlock", "Wizard"]
class_menu = ttk.Combobox(app, textvariable=class_var, values=class_options)
class_menu.grid(row=2, column=1)

tk.Label(app, text="Background:").grid(row=3, column=0)
background_entry = tk.Entry(app)
background_entry.grid(row=3, column=1)

tk.Label(app, text="Alignment:").grid(row=4, column=0)
alignment_var = tk.StringVar()
alignment_options = ["Lawful Good", "Neutral Good", "Chaotic Good", "Lawful Neutral", "True Neutral", "Chaotic Neutral",
                     "Lawful Evil", "Neutral Evil", "Chaotic Evil"]
alignment_menu = ttk.Combobox(app, textvariable=alignment_var, values=alignment_options)
alignment_menu.grid(row=4, column=1)

tk.Label(app, text="Additional Equipment:").grid(row=5, column=0)
equipment_entry = tk.Entry(app)
equipment_entry.grid(row=5, column=1)

tk.Label(app, text="Age:").grid(row=6, column=0)
age_entry = tk.Entry(app)
age_entry.grid(row=6, column=1)

tk.Label(app, text="Gender:").grid(row=7, column=0)
gender_var = tk.StringVar()
gender_options = ["Male", "Female", "Non-binary", "Other"]
gender_menu = ttk.Combobox(app, textvariable=gender_var, values=gender_options)
gender_menu.grid(row=7, column=1)

tk.Label(app, text="Height:").grid(row=8, column=0)
height_entry = tk.Entry(app)
height_entry.grid(row=8, column=1)

tk.Label(app, text="Weight:").grid(row=9, column=0)
weight_entry = tk.Entry(app)
weight_entry.grid(row=9, column=1)

tk.Label(app, text="Eye Color:").grid(row=10, column=0)
eye_color_entry = tk.Entry(app)
eye_color_entry.grid(row=10, column=1)

tk.Label(app, text="Hair Color:").grid(row=11, column=0)
hair_color_entry = tk.Entry(app)
hair_color_entry.grid(row=11, column=1)

tk.Label(app, text="Skin Color:").grid(row=12, column=0)
skin_color_entry = tk.Entry(app)
skin_color_entry.grid(row=12, column=1)

tk.Label(app, text="Personality Traits:").grid(row=13, column=0)
personality_traits_entry = tk.Entry(app)
personality_traits_entry.grid(row=13, column=1)

tk.Label(app, text="Ideals:").grid(row=14, column=0)
ideals_entry = tk.Entry(app)
ideals_entry.grid(row=14, column=1)

tk.Label(app, text="Bonds:").grid(row=15, column=0)
bonds_entry = tk.Entry(app)
bonds_entry.grid(row=15, column=1)

tk.Label(app, text="Flaws:").grid(row=16, column=0)
flaws_entry = tk.Entry(app)
flaws_entry.grid(row=16, column=1)

create_button = tk.Button(app, text="Create Character", command=create_character)
create_button.grid(row=17, columnspan=2)

random_character_button = tk.Button(app, text="Random Character", command=generate_random_character)
random_character_button.grid(row=18, columnspan=2)

app.mainloop()