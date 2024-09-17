import random
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image
import torch
from torchvision import transforms as tfms
from safetensors.torch import load_file

import torch
import torch.nn as nn

class StableDiffusion(nn.Module):
    def __init__(self, state_dict):
        super(StableDiffusion, self).__init__()
        # Define your actual model architecture here
        self.time_embed_0 = nn.Linear(256, 512)
        self.time_embed_0.weight = nn.Parameter(state_dict['time_embed_0.weight'])
        self.time_embed_0.bias = nn.Parameter(state_dict['time_embed_0.bias'])
        self.time_embed_2 = nn.Linear(512, 256)
        self.time_enbed_2.weight = nn.Parameter(state_dict['time_embed_2.weight'])
        self.time_embed_2.bias = nn.Parameter(state_dict['time_embed_2.bias'])
        self.input_blocks_0_0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.input_blocks_0_0.weight = nn.Parameter(state_dict['model.diffusion_model.input_blocks.0.0.weight'])
        self.input_blocks_0_0.bias = nn.Parameter(state_dict['model.input_blocks.0.0.bias'])
        self.input_blocks_1_0_in_layers_0 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.input_blocks_1_0_in_layers_0.weight = nn.Parameter(state_dict['model.diffusion_model.input_blocks.1.0.in_layers.0.weight'])
        self.input_blocks_1_0_in_layers_0.bias = nn.Parameter(state_dict['model.diffusion_model.input_blocks.1.0.in_layers.0.bias'])
        self.input_blocks_1_0_in_layers_2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.input_blocks_1_0_in_layers_2.weight = nn.Parameter(state_dict['model.diffusion_model.input_blocks.1.0.in_layers.2.weight'])
        self.input_blocks_1_0_in_layers_2.bias = nn.Parameter(state_dict['model.diffusion_model.input_blocks.1.0.in_layers.2.bias'])
        self.input_blocks_1_0_emb_layers_1 = nn.Linear(256, 512)
        self.input_blocks_1_0_emb_layers_1.weight = nn.Parameter(state_dict['model.diffusion_model.input_blocks.1.0.emb_layers.1.weight'])
        self.input_blocks_1_0_emb_layers_1.bias = nn.Parameter(state_dict['model.diffusion_model.input_blocks.1.0.emb_layers.1.bias'])
        self.input_blocks_1_0_out_layers_0 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.input_blocks_1_0_out_layers_0.weight = nn.Parameter(state_dict['model.diffusion_model.input_blocks.1.0.out_layers.0.weight'])
        self.input_blocks_1_0_out_layers_0.bias = nn.Parameter(state_dict['model.diffusion_model.input_blocks.1.0.out_layers.0.bias'])
        self.cond_stage_model_transformer_vision_model_encoder_layers_5_self_attn_v_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_5_self_attn_q_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_5_self_attn_out_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_5_layer_norm1 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_5_mlp_fc1 = nn.Linear(512, 2048)
        self.cond_stage_model_transformer_vision_model_encoder_layers_5_mlp_fc2 = nn.Linear(2048, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_5_layer_norm2 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_6_self_attn_k_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_6_self_attn_v_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_6_self_attn_q_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_6_self_attn_out_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_6_layer_norm1 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_6_mlp_fc1 = nn.Linear(512, 2048)
        self.cond_stage_model_transformer_vision_model_encoder_layers_6_mlp_fc2 = nn.Linear(2048, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_6_layer_norm2 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_7_self_attn_k_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_7_self_attn_v_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_7_self_attn_q_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_7_self_attn_out_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_7_layer_norm1 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_7_mlp_fc1 = nn.Linear(512, 2048)
        self.cond_stage_model_transformer_vision_model_encoder_layers_7_mlp_fc2 = nn.Linear(2048, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_7_layer_norm2 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_8_self_attn_k_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_8_self_attn_v_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_8_self_attn_q_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_8_self_attn_out_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_8_layer_norm1 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_8_mlp_fc1 = nn.Linear(512, 2048)
        self.cond_stage_model_transformer_vision_model_encoder_layers_8_mlp_fc2 = nn.Linear(2048, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_8_layer_norm2 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_9_self_attn_k_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_9_self_attn_v_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_9_self_attn_q_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_9_self_attn_out_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_9_layer_norm1 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_9_mlp_fc1 = nn.Linear(512, 2048)
        self.cond_stage_model_transformer_vision_model_encoder_layers_9_mlp_fc2 = nn.Linear(2048, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_9_layer_norm2 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_10_self_attn_k_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_10_self_attn_v_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_10_self_attn_q_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_10_self_attn_out_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_10_layer_norm1 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_10_mlp_fc1 = nn.Linear(512, 2048)
        self.cond_stage_model_transformer_vision_model_encoder_layers_10_mlp_fc2 = nn.Linear(2048, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_10_layer_norm2 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_11_self_attn_k_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_11_self_attn_v_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_11_self_attn_q_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_11_self_attn_out_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_11_layer_norm1 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_11_mlp_fc1 = nn.Linear(512, 2048)
        self.cond_stage_model_transformer_vision_model_encoder_layers_11_mlp_fc2 = nn.Linear(2048, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_11_layer_norm2 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_12_self_attn_k_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_12_self_attn_v_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_12_self_attn_q_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_12_self_attn_out_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_12_layer_norm1 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_12_mlp_fc1 = nn.Linear(512, 2048)
        self.cond_stage_model_transformer_vision_model_encoder_layers_12_mlp_fc2 = nn.Linear(2048, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_12_layer_norm2 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_13_self_attn_k_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_13_self_attn_v_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_13_self_attn_q_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_13_self_attn_out_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_13_layer_norm1 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_13_mlp_fc1 = nn.Linear(512, 2048)
        self.cond_stage_model_transformer_vision_model_encoder_layers_13_mlp_fc2 = nn.Linear(2048, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_13_layer_norm2 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_14_self_attn_k_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_14_self_attn_v_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_14_self_attn_q_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_14_self_attn_out_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_14_layer_norm1 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_14_mlp_fc1 = nn.Linear(512, 2048)
        self.cond_stage_model_transformer_vision_model_encoder_layers_14_mlp_fc2 = nn.Linear(2048, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_14_layer_norm2 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_15_self_attn_k_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_15_self_attn_v_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_15_self_attn_q_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_15_self_attn_out_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_15_layer_norm1 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_15_mlp_fc1 = nn.Linear(512, 2048)
        self.cond_stage_model_transformer_vision_model_encoder_layers_15_mlp_fc2 = nn.Linear(2048, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_15_layer_norm2 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_16_self_attn_k_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_16_self_attn_v_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_16_self_attn_q_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_16_self_attn_out_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_16_layer_norm1 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_16_mlp_fc1 = nn.Linear(512, 2048)
        self.cond_stage_model_transformer_vision_model_encoder_layers_16_mlp_fc2 = nn.Linear(2048, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_16_layer_norm2 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_17_self_attn_k_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_17_self_attn_v_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_17_self_attn_q_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_17_self_attn_out_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_17_layer_norm1 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_17_mlp_fc1 = nn.Linear(512, 2048)
        self.cond_stage_model_transformer_vision_model_encoder_layers_17_mlp_fc2 = nn.Linear(2048, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_17_layer_norm2 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_18_self_attn_k_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_18_self_attn_v_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_18_self_attn_q_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_18_self_attn_out_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_18_layer_norm1 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_18_mlp_fc1 = nn.Linear(512, 2048)
        self.cond_stage_model_transformer_vision_model_encoder_layers_18_mlp_fc2 = nn.Linear(2048, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_18_layer_norm2 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_19_self_attn_k_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_19_self_attn_v_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_19_self_attn_q_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_19_self_attn_out_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_19_layer_norm1 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_19_mlp_fc1 = nn.Linear(512, 2048)
        self.cond_stage_model_transformer_vision_model_encoder_layers_19_mlp_fc2 = nn.Linear(2048, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_19_layer_norm2 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_20_self_attn_k_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_20_self_attn_v_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_20_self_attn_q_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_20_self_attn_out_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_20_layer_norm1 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_20_mlp_fc1 = nn.Linear(512, 2048)
        self.cond_stage_model_transformer_vision_model_encoder_layers_20_mlp_fc2 = nn.Linear(2048, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_20_layer_norm2 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_21_self_attn_k_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_21_self_attn_v_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_21_self_attn_q_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_21_self_attn_out_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_21_layer_norm1 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_21_mlp_fc1 = nn.Linear(512, 2048)
        self.cond_stage_model_transformer_vision_model_encoder_layers_21_mlp_fc2 = nn.Linear(2048, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_21_layer_norm2 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_22_self_attn_k_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_22_self_attn_v_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_22_self_attn_q_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_22_self_attn_out_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_22_layer_norm1 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_22_mlp_fc1 = nn.Linear(512, 2048)
        self.cond_stage_model_transformer_vision_model_encoder_layers_22_mlp_fc2 = nn.Linear(2048, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_22_layer_norm2 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_23_self_attn_k_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_23_self_attn_v_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_23_self_attn_q_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_23_self_attn_out_proj = nn.Linear(512, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_23_layer_norm1 = nn.LayerNorm(512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_23_mlp_fc1 = nn.Linear(512, 2048)
        self.cond_stage_model_transformer_vision_model_encoder_layers_23_mlp_fc2 = nn.Linear(2048, 512)
        self.cond_stage_model_transformer_vision_model_encoder_layers_23_layer_norm2 = nn.LayerNorm(512)



    @staticmethod
    def load_from_checkpoint(checkpoint_path):
        try:
            # Load the checkpoint and extract the state dictionary
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            state_dict = checkpoint['state_dict']
            model = StableDiffusion()
            model.load_state_dict(state_dict)
            return model
        except Exception as e:
            print(f"Error loading model checkpoint: {e}")
            return None

    def generate(self, description):
        # Generate an image based on the description
        # This is a placeholder implementation
        image = torch.zeros((3, 256, 256))  # Example: a blank image
        return image

class Character:
    def __init__(self, name, race, char_class, strength, dexterity, constitution, intelligence, wisdom, charisma, starting_items, skills_with_damage, background, alignment, additional_equipment):
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
    return f"Name: {character.name}\nRace: {character.race}\nClass: {character.char_class}\nBackground: {character.background}\nAlignment: {character.alignment}\nAdditional Equipment: {character.additional_equipment}"

def generate_character_image(description):
    # Load the model
    model = StableDiffusion.load_from_checkpoint('StableDiffusion/stable-diffusion-webui/models/StableDiffusion/model.ckpt')

    if model is None:
        print("Failed to load the model.")
        return None

    # Generate the image
    with torch.no_grad():
        image = model.generate(description)

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
    print("create_character function called")  # Debug statement
    name = name_entry.get()
    race = race_var.get()
    char_class = class_var.get()
    background = background_entry.get()
    alignment = alignment_var.get()
    additional_equipment = equipment_entry.get()

    print(f"Name: {name}, Race: {race}, Class: {char_class}, Background: {background}, Alignment: {alignment}, Additional Equipment: {additional_equipment}")  # Debug statement

    if not name or not race or not char_class or not background or not alignment:
        messagebox.showerror("Input Error", "Please fill in all fields")
        return

    stats = roll_stats_for_class(char_class)
    starting_items, skills_with_damage = get_starting_items_and_skills(char_class)
    character = Character(name, race, char_class, stats["strength"], stats["dexterity"], stats["constitution"], stats["intelligence"], stats["wisdom"], stats["charisma"], starting_items, skills_with_damage, background, alignment, additional_equipment)
    generate_image(character)
    messagebox.showinfo("Success", f"Character {name} created and saved to image")

def generate_random_name():
    random_names = [
        "Aragorn", "Legolas", "Gimli", "Frodo", "Gandalf", "Boromir", "Samwise", "Pippin", "Merry", "Eowyn",
        "Arwen", "Elrond", "Galadriel", "Thranduil", "Celeborn", "Glorfindel", "Eomer", "Theoden", "Faramir", "Denethor",
        "Isildur", "Anarion", "Gil-galad", "Cirdan", "Radagast", "Saruman", "Sauron", "Shelob", "Gollum", "Smaug",
        "Balin", "Dwalin", "Kili", "Fili", "Dori", "Nori", "Ori", "Oin", "Gloin", "Bifur",
        "Bofur", "Bombur", "Thorin", "Bard", "Beorn", "Thranduil", "Tauriel", "Azog", "Bolg", "Dain",
        "Thror", "Thrain", "Durin", "Dis", "Frerin", "Fundin", "Narvi", "Telchar", "Gundabad", "Khazad-dum",
        "Moria", "Erebor", "Dale", "Laketown", "Rivendell", "Lothlorien", "Mirkwood", "Fangorn", "Isengard", "Minas Tirith",
        "Minas Morgul", "Osgiliath", "Helm's Deep", "Edoras", "Dunharrow", "Pelennor Fields", "Mordor", "Barad-dur", "Mount Doom", "Cirith Ungol",
        "Gondor", "Rohan", "Shire", "Bree", "Weathertop", "Amon Hen", "Amon Sul", "Emyn Muil", "Dead Marshes", "Black Gate",
        "Grey Havens", "Valinor", "Numenor", "Anduin", "Misty Mountains", "Blue Mountains", "White Mountains", "Redhorn Pass", "Caradhras", "Mount Gundabad"
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
class_options = ["Barbarian", "Bard", "Cleric", "Druid", "Fighter", "Monk", "Paladin", "Ranger", "Rogue", "Sorcerer", "Warlock", "Wizard"]
class_menu = ttk.Combobox(app, textvariable=class_var, values=class_options)
class_menu.grid(row=2, column=1)

# Add new fields for background, alignment, and additional equipment
tk.Label(app, text="Background:").grid(row=3, column=0)
background_entry = tk.Entry(app)
background_entry.grid(row=3, column=1)

tk.Label(app, text="Alignment:").grid(row=4, column=0)
alignment_var = tk.StringVar()
alignment_options = ["Lawful Good", "Neutral Good", "Chaotic Good", "Lawful Neutral", "True Neutral", "Chaotic Neutral", "Lawful Evil", "Neutral Evil", "Chaotic Evil"]
alignment_menu = ttk.Combobox(app, textvariable=alignment_var, values=alignment_options)
alignment_menu.grid(row=4, column=1)

tk.Label(app, text="Additional Equipment:").grid(row=5, column=0)
equipment_entry = tk.Entry(app)
equipment_entry.grid(row=5, column=1)

create_button = tk.Button(app, text="Create Character", command=create_character)
create_button.grid(row=6, columnspan=2)

print(f"Button command: {create_button.cget('command')}")  # Debug statement

app.mainloop()