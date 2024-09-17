import random
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image
from diffusers import StableDiffusionPipeline
import torch


class Character:
    def __init__(self, name, race, char_class, strength, dexterity, constitution, intelligence, wisdom, charisma,
                 starting_items, skills_with_damage, background, alignment, additional_equipment):
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


def roll_stat():
    rolls = [random.randint(1, 6) for _ in range(4)]
    return sum(sorted(rolls)[1:])


def generate_image(character):
    description = generate_character_description(character)
    image = generate_character_image(description)
    if image:
        image.save(f"{character.name}_character_image.png")


def generate_character_description(character):
    return f"Name: {character.name}, Race: {character.race}, Class: {character.char_class}, Background: {character.background}, Alignment: {character.alignment}, Additional Equipment: {character.additional_equipment}"


def generate_character_image(description):
    # Load the pre-trained Stable Diffusion model
    print(f"Generating image for description: {description}")  # Debug print

    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate the image using the description
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

    if not name or not race or not char_class or not background or not alignment:
        messagebox.showerror("Input Error", "Please fill in all fields")
        return

    stats = roll_stats_for_class(char_class)
    starting_items, skills_with_damage = get_starting_items_and_skills(char_class)
    character = Character(name, race, char_class, stats["strength"], stats["dexterity"], stats["constitution"],
                          stats["intelligence"], stats["wisdom"], stats["charisma"], starting_items, skills_with_damage,
                          background, alignment, additional_equipment)
    generate_image(character)
    messagebox.showinfo("Success", f"Character {name} created and saved to image")


def generate_random_name():
    random_names = [
        "Aragorn", "Legolas", "Gimli", "Frodo", "Gandalf", "Boromir", "Samwise", "Pippin", "Merry", "Eowyn",
        "Arwen", "Elrond", "Galadriel", "Thranduil", "Celeborn", "Glorfindel", "Eomer", "Theoden", "Faramir", "Denethor"
    ]
    name_entry.delete(0, tk.END)
    name_entry.insert(0, random.choice(random_names))


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

# Add new fields for background, alignment, and additional equipment
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

create_button = tk.Button(app, text="Create Character", command=create_character)
create_button.grid(row=6, columnspan=2)

app.mainloop()