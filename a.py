import questionary
from questionary import Style

custom_style = Style([
    ("qmark",       "fg:#00b4d8 bold"),      # the leading "?" symbol
    ("question",    "bold"),                 # question text
    ("answer",      "fg:#90e0ef bold"),      # the answer after you confirm
    ("pointer",     "fg:#ff6b6b bold"),      # the ❯ pointer in lists
    ("highlighted", "fg:#ff6b6b bold"),      # the option under the cursor
    # ("selected",    "fg:#90e0ef"),           # checked items in checkbox
    # ("selected",    "fg:#ff6b6b noreverse"),           # checked items in checkbox
    ("separator",   "fg:#6c6c6c"),           # separator lines
    ("instruction", "fg:#6c6c6c italic"),    # "(Use arrow keys)" hint
    ("text",        ""),                     # normal option text
    ("disabled",    "fg:#858585 italic"),    # disabled choices
])

model = questionary.select(
    "Pick a model:",
    choices=["resnet50", "vit_b_16", "efficientnet_b0"],
    style=custom_style,
    qmark="🚀",              # replace the "?" symbol
    pointer="➜",             # replace the "❯" pointer
    instruction="(↑/↓ to move, Enter to confirm)",
).ask()

augs = questionary.checkbox(
    "Select augmentations:",
    choices=["RandomFlip", "ColorJitter", "MixUp"],
    style=custom_style,
    # use_indicator=True
    # use_jedi=False,          # ignore if not needed
).ask()

# Other prompt types
name = questionary.text("Run name:").ask()
ok = questionary.confirm("Start training?").ask()
path = questionary.path("Config file:").ask()

print(model, augs, name, ok, path)