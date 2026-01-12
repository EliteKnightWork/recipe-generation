"""
Input preprocessing module for recipe generation.
Handles ingredient normalization, validation, and cleaning.
"""

import re
from typing import List, Set, Tuple, Optional
from dataclasses import dataclass, field


# Common ingredient abbreviations and their expansions
ABBREVIATIONS = {
    # Volume measurements
    "tbsp": "tablespoon",
    "tbsps": "tablespoons",
    "tbs": "tablespoon",
    "tbl": "tablespoon",
    "tsp": "teaspoon",
    "tsps": "teaspoons",
    "c": "cup",
    "c.": "cup",
    "fl oz": "fluid ounce",
    "fl. oz.": "fluid ounce",
    "pt": "pint",
    "pts": "pints",
    "qt": "quart",
    "qts": "quarts",
    "gal": "gallon",
    "gals": "gallons",
    "ml": "milliliter",
    "mls": "milliliters",
    "l": "liter",
    "dl": "deciliter",

    # Weight measurements
    "oz": "ounce",
    "ozs": "ounces",
    "lb": "pound",
    "lbs": "pounds",
    "g": "gram",
    "gr": "gram",
    "gm": "gram",
    "gms": "grams",
    "kg": "kilogram",
    "kgs": "kilograms",

    # Size descriptors
    "sm": "small",
    "sml": "small",
    "med": "medium",
    "md": "medium",
    "lg": "large",
    "lrg": "large",
    "xl": "extra large",

    # Package/Container
    "pkg": "package",
    "pkgs": "packages",
    "pkt": "packet",
    "cn": "can",
    "cns": "cans",
    "btl": "bottle",
    "jar": "jar",
    "box": "box",
    "env": "envelope",
    "ct": "count",

    # Preparation states
    "chpd": "chopped",
    "dcd": "diced",
    "slcd": "sliced",
    "mcd": "minced",
    "crshd": "crushed",
    "grnd": "ground",
    "shrd": "shredded",
    "grtd": "grated",
    "drnd": "drained",
    "rns": "rinsed",
    "frsh": "fresh",
    "frz": "frozen",
    "thwd": "thawed",
    "cnd": "canned",
    "dryd": "dried",
    "rstd": "roasted",
    "tstd": "toasted",
    "sknd": "skinned",
    "bnls": "boneless",
    "sknls": "skinless",

    # Common food abbreviations
    "evoo": "extra virgin olive oil",
    "oo": "olive oil",
    "veg": "vegetable",
    "vegs": "vegetables",
    "chx": "chicken",
    "chkn": "chicken",
    "bf": "beef",
    "prk": "pork",
    "pot": "potato",
    "pots": "potatoes",
    "tom": "tomato",
    "toms": "tomatoes",
    "mush": "mushroom",
    "mushs": "mushrooms",
    "mus": "mushroom",
    "parm": "parmesan",
    "parmsan": "parmesan",
    "mozz": "mozzarella",
    "mayo": "mayonnaise",
    "worcs": "worcestershire",
    "worc": "worcestershire",

    # Misc
    "approx": "approximately",
    "abt": "about",
    "ea": "each",
    "w/": "with",
    "w/o": "without",
    "opt": "optional",
    "temp": "temperature",
    "min": "minute",
    "mins": "minutes",
    "hr": "hour",
    "hrs": "hours",
    "sec": "second",
    "secs": "seconds",
    "deg": "degree",
    "f": "fahrenheit",
    "fahr": "fahrenheit",
}

# Common ingredient synonyms (map to canonical form)
SYNONYMS = {
    # British to American English
    "capsicum": "bell pepper",
    "aubergine": "eggplant",
    "courgette": "zucchini",
    "coriander leaves": "cilantro",
    "fresh coriander": "cilantro",
    "rocket": "arugula",
    "roquette": "arugula",
    "prawns": "shrimp",
    "king prawns": "large shrimp",
    "gammon": "ham",
    "rashers": "bacon slices",
    "streaky bacon": "bacon",
    "back bacon": "canadian bacon",
    "mince": "ground meat",
    "minced meat": "ground meat",
    "minced beef": "ground beef",
    "beef mince": "ground beef",
    "minced pork": "ground pork",
    "pork mince": "ground pork",
    "minced lamb": "ground lamb",
    "lamb mince": "ground lamb",
    "minced turkey": "ground turkey",
    "turkey mince": "ground turkey",
    "minced chicken": "ground chicken",
    "spring onion": "green onion",
    "spring onions": "green onions",
    "salad onion": "green onion",
    "scallion": "green onion",
    "scallions": "green onions",
    "shallots": "shallot",
    "caster sugar": "superfine sugar",
    "castor sugar": "superfine sugar",
    "icing sugar": "powdered sugar",
    "confectioner's sugar": "powdered sugar",
    "demerara sugar": "turbinado sugar",
    "muscovado sugar": "dark brown sugar",
    "golden syrup": "light corn syrup",
    "treacle": "molasses",
    "black treacle": "blackstrap molasses",
    "bicarbonate of soda": "baking soda",
    "bicarb": "baking soda",
    "bread soda": "baking soda",
    "plain flour": "all-purpose flour",
    "strong flour": "bread flour",
    "strong bread flour": "bread flour",
    "self-raising flour": "self-rising flour",
    "wholemeal flour": "whole wheat flour",
    "cornflour": "cornstarch",
    "maize flour": "corn flour",
    "polenta": "cornmeal",
    "double cream": "heavy cream",
    "single cream": "light cream",
    "whipping cream": "heavy whipping cream",
    "clotted cream": "heavy cream",
    "soured cream": "sour cream",
    "natural yoghurt": "plain yogurt",
    "greek style yogurt": "greek yogurt",
    "creme fraiche": "sour cream",
    "fromage frais": "cream cheese",
    "full fat milk": "whole milk",
    "semi-skimmed milk": "2% milk",
    "skimmed milk": "skim milk",

    # Herbs and aromatics
    "coriander": "cilantro",
    "chinese parsley": "cilantro",
    "flat-leaf parsley": "italian parsley",
    "cos lettuce": "romaine lettuce",
    "pak choi": "bok choy",
    "chinese cabbage": "napa cabbage",
    "chinese leaves": "napa cabbage",
    "mangetout": "snow peas",
    "mange tout": "snow peas",
    "sugar snap peas": "snap peas",
    "french beans": "green beans",
    "haricot beans": "navy beans",
    "haricots verts": "french green beans",
    "runner beans": "green beans",
    "broad beans": "fava beans",
    "butter beans": "lima beans",
    "chickpea": "garbanzo bean",
    "chick pea": "garbanzo bean",
    "gram": "chickpea flour",
    "besan": "chickpea flour",

    # Vegetables
    "swede": "rutabaga",
    "beetroot": "beet",
    "celeriac": "celery root",
    "salsify": "oyster plant",
    "marrow": "large zucchini",
    "gem squash": "acorn squash",
    "butternut pumpkin": "butternut squash",
    "sweet corn": "corn",
    "baby sweetcorn": "baby corn",
    "tinned tomatoes": "canned tomatoes",
    "passata": "tomato puree",
    "sun-dried tomatoes": "sun dried tomatoes",
    "sundried tomatoes": "sun dried tomatoes",
    "cherry toms": "cherry tomatoes",
    "plum tomatoes": "roma tomatoes",

    # Seafood
    "king prawn": "jumbo shrimp",
    "tiger prawn": "large shrimp",
    "langoustine": "lobster tail",
    "crayfish": "crawfish",
    "calamari": "squid",
    "white fish": "cod",
    "oily fish": "salmon",
    "smoked haddock": "smoked fish",
    "finnan haddie": "smoked haddock",

    # Meat
    "gammon steak": "ham steak",
    "pork belly slices": "pork belly",
    "silverside": "bottom round",
    "topside": "top round",
    "braising steak": "chuck roast",
    "stewing beef": "beef stew meat",
    "frying steak": "sirloin steak",
    "escalope": "cutlet",
    "escalopes": "cutlets",

    # Cheese
    "mature cheddar": "sharp cheddar",
    "mild cheddar": "mild cheddar cheese",
    "red leicester": "colby cheese",
    "lancashire": "white cheddar",
    "stilton": "blue cheese",
    "dolcelatte": "gorgonzola",
    "gruyère": "gruyere",
    "emmenthal": "swiss cheese",
    "emmental": "swiss cheese",
    "edam": "gouda",
    "halloumi": "grilling cheese",
    "paneer": "indian cottage cheese",
    "quark": "fromage blanc",

    # Oils and fats
    "groundnut oil": "peanut oil",
    "rapeseed oil": "canola oil",
    "vegetable suet": "shortening",
    "lard": "pork fat",
    "dripping": "beef fat",

    # Condiments and sauces
    "brown sauce": "steak sauce",
    "hp sauce": "steak sauce",
    "salad cream": "mayonnaise",
    "tomato ketchup": "ketchup",
    "tomato sauce": "ketchup",
    "english mustard": "hot mustard",
    "french mustard": "dijon mustard",
    "wholegrain mustard": "whole grain mustard",
    "piccalilli": "mustard pickle",
    "branston pickle": "sweet pickle relish",
    "mango chutney": "indian mango chutney",

    # Asian ingredients
    "soya sauce": "soy sauce",
    "light soy": "light soy sauce",
    "dark soy": "dark soy sauce",
    "rice wine": "shaoxing wine",
    "chinese rice wine": "shaoxing wine",
    "nam pla": "fish sauce",
    "nuoc mam": "fish sauce",
    "galangal": "thai ginger",
    "kaffir lime": "makrut lime",
    "thai basil": "asian basil",
    "holy basil": "tulsi",
    "bird's eye chili": "thai chili",
    "birds eye chilli": "thai chili",
    "red chilli": "red chili",
    "green chilli": "green chili",
    "chilli flakes": "red pepper flakes",
    "chilli powder": "chili powder",
    "garam masala": "indian spice blend",
    "chinese five-spice": "five spice powder",
    "five spice": "five spice powder",

    # Baking
    "digestive biscuits": "graham crackers",
    "biscuits": "cookies",
    "savoury biscuits": "crackers",
    "sponge fingers": "ladyfingers",
    "hundreds and thousands": "sprinkles",
    "desiccated coconut": "shredded coconut",
    "glacé cherries": "candied cherries",
    "candied peel": "mixed peel",
    "mixed spice": "pumpkin pie spice",
    "vanilla essence": "vanilla extract",
    "almond essence": "almond extract",
    "lemon essence": "lemon extract",
    "rose water": "rosewater",
    "orange blossom water": "orange flower water",

    # Nuts
    "ground almonds": "almond flour",
    "flaked almonds": "sliced almonds",
    "nibbed almonds": "chopped almonds",
    "monkey nuts": "peanuts in shell",
    "pine kernels": "pine nuts",

    # Grains and pasta
    "macaroni cheese": "mac and cheese",
    "vermicelli": "angel hair pasta",
    "tagliatelle": "fettuccine",
    "pappardelle": "wide egg noodles",
    "conchiglie": "shell pasta",
    "farfalle": "bow tie pasta",
    "rigatoni": "tube pasta",
    "penne rigate": "penne",
    "wholemeal pasta": "whole wheat pasta",
    "egg noodle": "egg noodles",
    "rice stick": "rice noodles",
    "cellophane noodle": "glass noodles",
    "bean thread noodles": "glass noodles",

    # Drinks
    "fizzy water": "sparkling water",
    "soda water": "club soda",
    "lemonade": "lemon soda",
    "squash": "fruit concentrate",
    "cordial": "fruit syrup",
}

# Known valid food categories for validation
FOOD_CATEGORIES = {
    "proteins": {
        # Poultry
        "chicken", "chicken breast", "chicken thigh", "chicken wing", "chicken leg",
        "turkey", "turkey breast", "ground turkey", "duck", "goose", "quail",
        # Beef
        "beef", "ground beef", "steak", "sirloin", "ribeye", "filet mignon", "chuck",
        "brisket", "flank steak", "beef tenderloin", "roast beef", "corned beef",
        # Pork
        "pork", "pork chop", "pork loin", "pork tenderloin", "ground pork", "pork belly",
        "bacon", "ham", "prosciutto", "pancetta", "sausage", "chorizo", "pepperoni",
        "salami", "hot dog", "bratwurst", "kielbasa",
        # Lamb & Game
        "lamb", "lamb chop", "ground lamb", "lamb shank", "mutton", "venison", "bison",
        "rabbit", "goat",
        # Seafood - Fish
        "fish", "salmon", "tuna", "cod", "tilapia", "halibut", "trout", "bass",
        "catfish", "snapper", "mahi mahi", "swordfish", "sardine", "anchovy",
        "mackerel", "herring", "flounder", "sole", "perch", "pike", "carp",
        # Seafood - Shellfish
        "shrimp", "prawns", "lobster", "crab", "crab meat", "scallop", "clam",
        "mussel", "oyster", "squid", "calamari", "octopus", "crawfish", "crayfish",
        # Eggs
        "egg", "eggs", "egg white", "egg yolk", "quail egg",
        # Plant-based proteins
        "tofu", "tempeh", "seitan", "edamame", "textured vegetable protein", "tvp",
        "beyond meat", "impossible meat", "plant protein",
        # Legume proteins
        "lentils", "chickpeas", "black beans", "kidney beans", "navy beans",
        "pinto beans", "cannellini beans", "lima beans", "split peas",
    },
    "vegetables": {
        # Alliums
        "onion", "red onion", "white onion", "yellow onion", "sweet onion",
        "green onion", "scallion", "shallot", "leek", "chive", "garlic",
        # Nightshades
        "tomato", "cherry tomato", "roma tomato", "sun dried tomato", "tomato paste",
        "pepper", "bell pepper", "red pepper", "green pepper", "yellow pepper",
        "jalapeno", "serrano", "habanero", "poblano", "anaheim", "banana pepper",
        "eggplant", "potato", "red potato", "russet potato", "yukon gold", "sweet potato",
        # Cruciferous
        "broccoli", "cauliflower", "cabbage", "red cabbage", "napa cabbage",
        "brussels sprout", "kale", "collard greens", "bok choy", "kohlrabi",
        "arugula", "watercress", "radish", "daikon", "turnip", "rutabaga",
        # Leafy greens
        "spinach", "lettuce", "romaine", "iceberg", "butter lettuce", "mixed greens",
        "swiss chard", "mustard greens", "beet greens", "endive", "radicchio",
        "escarole", "frisee",
        # Root vegetables
        "carrot", "celery", "beet", "parsnip", "celeriac", "jicama", "ginger root",
        "turmeric root", "horseradish", "salsify",
        # Squash
        "zucchini", "yellow squash", "butternut squash", "acorn squash", "spaghetti squash",
        "delicata squash", "kabocha", "pumpkin", "cucumber", "pickle",
        # Legumes (as vegetables)
        "green bean", "string bean", "snap pea", "snow pea", "pea", "english pea",
        "edamame", "fava bean", "wax bean",
        # Corn
        "corn", "sweet corn", "corn on the cob", "baby corn",
        # Mushrooms
        "mushroom", "white mushroom", "cremini", "portobello", "shiitake",
        "oyster mushroom", "chanterelle", "porcini", "morel", "enoki", "maitake",
        "king trumpet", "button mushroom",
        # Other vegetables
        "asparagus", "artichoke", "artichoke heart", "hearts of palm", "bamboo shoot",
        "water chestnut", "bean sprout", "alfalfa sprout", "fennel", "celery root",
        "okra", "rhubarb", "tomatillo", "avocado",
    },
    "fruits": {
        # Citrus
        "lemon", "lime", "orange", "blood orange", "mandarin", "tangerine",
        "clementine", "grapefruit", "pomelo", "kumquat", "yuzu", "citron",
        # Berries
        "strawberry", "blueberry", "raspberry", "blackberry", "cranberry",
        "boysenberry", "gooseberry", "lingonberry", "mulberry", "acai",
        "goji berry", "elderberry", "currant", "red currant", "black currant",
        # Stone fruits
        "peach", "nectarine", "plum", "apricot", "cherry", "sour cherry",
        "sweet cherry", "date", "prune",
        # Pome fruits
        "apple", "green apple", "red apple", "granny smith", "honeycrisp", "fuji",
        "gala", "pear", "asian pear", "quince",
        # Tropical fruits
        "banana", "plantain", "mango", "papaya", "pineapple", "coconut",
        "passion fruit", "guava", "lychee", "longan", "rambutan", "dragon fruit",
        "star fruit", "jackfruit", "durian", "breadfruit", "tamarind", "kiwi",
        "persimmon", "pomegranate", "fig",
        # Melons
        "watermelon", "cantaloupe", "honeydew", "casaba", "crenshaw",
        # Grapes
        "grape", "red grape", "green grape", "concord grape", "raisin",
        "golden raisin", "sultana", "currant",
    },
    "dairy": {
        # Milk
        "milk", "whole milk", "skim milk", "2% milk", "1% milk", "buttermilk",
        "evaporated milk", "condensed milk", "sweetened condensed milk",
        "half and half", "heavy cream", "light cream", "whipping cream",
        "clotted cream", "creme fraiche",
        # Cheese - Fresh
        "cheese", "mozzarella", "fresh mozzarella", "burrata", "ricotta",
        "cottage cheese", "cream cheese", "mascarpone", "queso fresco",
        "paneer", "feta", "goat cheese", "chevre",
        # Cheese - Aged/Hard
        "cheddar", "sharp cheddar", "mild cheddar", "parmesan", "parmigiano reggiano",
        "pecorino", "pecorino romano", "asiago", "manchego", "gruyere",
        "emmental", "swiss cheese", "provolone", "fontina", "gouda",
        "edam", "havarti", "muenster", "monterey jack", "colby", "colby jack",
        "pepper jack", "american cheese",
        # Cheese - Blue
        "blue cheese", "gorgonzola", "roquefort", "stilton", "danish blue",
        # Cheese - Other
        "brie", "camembert", "limburger", "raclette", "halloumi",
        "string cheese", "queso", "velveeta", "cheese whiz",
        # Butter
        "butter", "unsalted butter", "salted butter", "clarified butter", "ghee",
        "margarine",
        # Yogurt & Sour cream
        "yogurt", "greek yogurt", "plain yogurt", "vanilla yogurt", "kefir",
        "sour cream", "creme fraiche", "labneh",
        # Ice cream
        "ice cream", "gelato", "frozen yogurt", "sherbet", "sorbet",
    },
    "grains": {
        # Rice
        "rice", "white rice", "brown rice", "jasmine rice", "basmati rice",
        "arborio rice", "sushi rice", "wild rice", "black rice", "red rice",
        "sticky rice", "glutinous rice", "rice flour",
        # Pasta
        "pasta", "spaghetti", "penne", "rigatoni", "fusilli", "farfalle",
        "fettuccine", "linguine", "angel hair", "capellini", "bucatini",
        "orecchiette", "orzo", "macaroni", "elbow macaroni", "rotini",
        "lasagna", "lasagna noodles", "ravioli", "tortellini", "gnocchi",
        "egg noodles", "rice noodles", "udon", "soba", "ramen", "vermicelli",
        "cellophane noodles", "glass noodles",
        # Bread
        "bread", "white bread", "wheat bread", "whole wheat bread", "rye bread",
        "sourdough", "baguette", "ciabatta", "focaccia", "pita", "naan",
        "flatbread", "tortilla", "flour tortilla", "corn tortilla", "wrap",
        "brioche", "challah", "pumpernickel", "bagel", "english muffin",
        "croissant", "roll", "dinner roll", "hamburger bun", "hot dog bun",
        "breadcrumbs", "panko", "crouton",
        # Flour
        "flour", "all-purpose flour", "bread flour", "cake flour", "pastry flour",
        "whole wheat flour", "self-rising flour", "almond flour", "coconut flour",
        "rice flour", "cornmeal", "corn flour", "masa harina", "semolina",
        "buckwheat flour", "rye flour", "chickpea flour", "tapioca flour",
        # Other grains
        "oats", "rolled oats", "steel cut oats", "instant oats", "oat flour",
        "quinoa", "bulgur", "couscous", "barley", "pearl barley", "farro",
        "millet", "amaranth", "teff", "spelt", "kamut", "freekeh", "wheat berry",
        "polenta", "grits", "cornmeal",
        # Cereals
        "cereal", "corn flakes", "granola", "muesli", "bran",
    },
    "seasonings": {
        # Salt
        "salt", "sea salt", "kosher salt", "table salt", "himalayan salt",
        "fleur de sel", "finishing salt", "smoked salt", "celery salt",
        "garlic salt", "onion salt", "seasoned salt",
        # Pepper
        "pepper", "black pepper", "white pepper", "pink peppercorn", "green peppercorn",
        "szechuan peppercorn", "cayenne", "red pepper flakes", "crushed red pepper",
        # Sugar & Sweeteners
        "sugar", "white sugar", "brown sugar", "light brown sugar", "dark brown sugar",
        "powdered sugar", "confectioners sugar", "raw sugar", "turbinado sugar",
        "demerara sugar", "coconut sugar", "maple sugar",
        "honey", "maple syrup", "agave", "molasses", "corn syrup", "golden syrup",
        "stevia", "monk fruit", "erythritol",
        # Oils
        "oil", "vegetable oil", "canola oil", "olive oil", "extra virgin olive oil",
        "coconut oil", "sesame oil", "toasted sesame oil", "peanut oil", "avocado oil",
        "grapeseed oil", "sunflower oil", "safflower oil", "corn oil", "walnut oil",
        "truffle oil", "chili oil",
        # Vinegars
        "vinegar", "white vinegar", "red wine vinegar", "white wine vinegar",
        "balsamic vinegar", "apple cider vinegar", "rice vinegar", "sherry vinegar",
        "champagne vinegar", "malt vinegar", "distilled vinegar",
        # Sauces - Asian
        "soy sauce", "tamari", "fish sauce", "oyster sauce", "hoisin sauce",
        "teriyaki sauce", "ponzu", "mirin", "sake", "rice wine", "shaoxing wine",
        "sriracha", "sambal", "gochujang", "doenjang", "miso", "white miso",
        "red miso", "tahini",
        # Sauces - Western
        "ketchup", "mustard", "yellow mustard", "dijon mustard", "whole grain mustard",
        "mayonnaise", "hot sauce", "tabasco", "worcestershire sauce", "bbq sauce",
        "steak sauce", "horseradish", "tartar sauce", "cocktail sauce", "ranch",
        "blue cheese dressing", "italian dressing", "caesar dressing", "vinaigrette",
        # Pastes & Concentrates
        "tomato paste", "tomato sauce", "tomato puree", "harissa", "curry paste",
        "red curry paste", "green curry paste", "yellow curry paste", "thai chili paste",
        "anchovy paste", "wasabi", "chipotle in adobo",
        # Stocks & Broths
        "chicken broth", "chicken stock", "beef broth", "beef stock",
        "vegetable broth", "vegetable stock", "fish stock", "bone broth",
        "bouillon", "bouillon cube", "dashi",
    },
    "herbs_spices": {
        # Fresh herbs
        "basil", "thai basil", "holy basil", "oregano", "thyme", "rosemary",
        "parsley", "flat leaf parsley", "curly parsley", "cilantro", "coriander",
        "dill", "mint", "spearmint", "peppermint", "sage", "tarragon", "chervil",
        "marjoram", "bay leaf", "lemongrass", "chive", "lavender", "epazote",
        "shiso", "curry leaf", "kaffir lime leaf", "makrut lime leaf",
        # Dried herbs
        "dried basil", "dried oregano", "dried thyme", "dried rosemary",
        "dried parsley", "dried dill", "dried sage", "dried tarragon",
        "herbes de provence", "italian seasoning", "bouquet garni", "fines herbes",
        # Ground spices
        "cumin", "ground cumin", "paprika", "smoked paprika", "sweet paprika",
        "hot paprika", "chili powder", "ancho chili powder", "chipotle powder",
        "cinnamon", "ground cinnamon", "nutmeg", "allspice", "clove", "ground cloves",
        "ginger", "ground ginger", "turmeric", "cardamom", "green cardamom",
        "black cardamom", "coriander seed", "ground coriander", "fennel seed",
        "mustard seed", "yellow mustard seed", "black mustard seed", "celery seed",
        "caraway seed", "dill seed", "anise", "star anise", "fenugreek",
        "sumac", "za'atar", "ras el hanout", "garam masala", "curry powder",
        "madras curry", "chinese five spice", "old bay", "cajun seasoning",
        "creole seasoning", "jerk seasoning", "taco seasoning", "chili flakes",
        # Whole spices
        "cinnamon stick", "whole cloves", "whole allspice", "whole nutmeg",
        "black peppercorn", "white peppercorn", "juniper berry", "vanilla bean",
        "vanilla extract", "almond extract", "peppermint extract",
        # Peppers & Chilies (dried)
        "dried chili", "guajillo", "ancho", "pasilla", "arbol", "cascabel",
        "chipotle", "morita", "mulato", "new mexico chili", "california chili",
        # Aromatics
        "saffron", "annatto", "achiote",
    },
    "nuts_seeds": {
        # Tree nuts
        "almond", "sliced almonds", "slivered almonds", "almond butter",
        "walnut", "pecan", "cashew", "cashew butter", "pistachio",
        "macadamia", "hazelnut", "filbert", "brazil nut", "pine nut", "pignoli",
        "chestnut", "marcona almond",
        # Peanuts (legumes but commonly grouped with nuts)
        "peanut", "peanut butter", "roasted peanut",
        # Seeds
        "sunflower seed", "pumpkin seed", "pepita", "sesame seed", "black sesame",
        "white sesame", "flax seed", "flaxseed", "chia seed", "hemp seed",
        "poppy seed", "caraway seed", "nigella seed", "black seed",
        # Coconut
        "coconut", "shredded coconut", "coconut flakes", "coconut milk",
        "coconut cream", "coconut water", "coconut oil",
    },
    "baking": {
        # Leaveners
        "baking powder", "baking soda", "yeast", "active dry yeast", "instant yeast",
        "fresh yeast", "cream of tartar",
        # Chocolate
        "chocolate", "dark chocolate", "milk chocolate", "white chocolate",
        "chocolate chips", "cocoa powder", "dutch process cocoa", "cacao",
        "unsweetened chocolate", "bittersweet chocolate", "semisweet chocolate",
        "chocolate bar", "cocoa butter",
        # Baking additions
        "vanilla", "vanilla extract", "vanilla bean", "vanilla paste",
        "almond extract", "lemon extract", "orange extract", "peppermint extract",
        "rum extract", "coconut extract",
        # Thickeners
        "cornstarch", "arrowroot", "tapioca starch", "potato starch", "gelatin",
        "pectin", "agar agar", "xanthan gum",
        # Dried fruits for baking
        "raisin", "golden raisin", "currant", "dried cranberry", "craisin",
        "dried cherry", "dried apricot", "dried fig", "dried date", "prune",
        "dried mango", "dried pineapple", "dried apple", "dried blueberry",
        "candied fruit", "candied ginger", "crystallized ginger",
        # Other baking items
        "graham cracker", "graham cracker crumbs", "oreo", "cookie crumbs",
        "pie crust", "puff pastry", "phyllo", "filo dough",
        "food coloring", "sprinkles", "meringue powder",
    },
    "condiments": {
        # Pickled & Preserved
        "pickle", "dill pickle", "bread and butter pickle", "gherkin",
        "pickled onion", "pickled jalapeno", "pickled ginger", "kimchi",
        "sauerkraut", "olive", "green olive", "black olive", "kalamata olive",
        "caper", "sun dried tomato", "roasted red pepper",
        # Spreads
        "jam", "jelly", "preserves", "marmalade", "apple butter", "nutella",
        "peanut butter", "almond butter", "tahini", "hummus",
        # Relishes
        "relish", "sweet relish", "dill relish", "chutney", "mango chutney",
        "cranberry sauce", "apple sauce", "salsa", "pico de gallo", "guacamole",
    },
    "beverages": {
        # Coffee & Tea
        "coffee", "espresso", "instant coffee", "coffee grounds",
        "tea", "black tea", "green tea", "oolong tea", "white tea",
        "herbal tea", "chamomile tea", "earl grey", "matcha",
        # Alcohol (for cooking)
        "wine", "red wine", "white wine", "dry white wine", "cooking wine",
        "sherry", "port", "marsala", "vermouth", "beer", "ale", "stout",
        "whiskey", "bourbon", "rum", "vodka", "brandy", "cognac",
        "grand marnier", "kahlua", "amaretto", "sake", "mirin",
        # Juices
        "orange juice", "lemon juice", "lime juice", "apple juice", "grape juice",
        "cranberry juice", "pineapple juice", "tomato juice", "pomegranate juice",
    },
    "canned_preserved": {
        # Canned vegetables
        "canned tomato", "diced tomatoes", "crushed tomatoes", "whole tomatoes",
        "canned corn", "canned peas", "canned green beans", "canned carrots",
        "canned beets", "canned artichoke", "canned hearts of palm",
        "canned pumpkin", "canned sweet potato",
        # Canned beans
        "canned beans", "canned black beans", "canned kidney beans",
        "canned chickpeas", "canned white beans", "canned pinto beans",
        "canned refried beans", "canned lentils",
        # Canned fish
        "canned tuna", "canned salmon", "canned sardines", "canned anchovies",
        "canned crab", "canned clams",
        # Canned fruits
        "canned peaches", "canned pears", "canned pineapple", "canned mandarin",
        "canned cherries", "canned fruit cocktail", "canned coconut milk",
    },
}


@dataclass
class ValidationResult:
    """Result of ingredient validation."""
    is_valid: bool
    original: str
    normalized: str
    warnings: List[str] = field(default_factory=list)
    category: Optional[str] = None


class InputPreprocessor:
    """
    Preprocessor for recipe generation inputs.
    Handles normalization, validation, and cleaning of ingredients.
    """

    def __init__(
        self,
        lowercase: bool = True,
        strip_whitespace: bool = True,
        remove_duplicates: bool = True,
        expand_abbreviations: bool = True,
        normalize_synonyms: bool = True,
        validate_ingredients: bool = True,
        min_length: int = 2,
        max_length: int = 50,
        max_ingredients: int = 20
    ):
        self.lowercase = lowercase
        self.strip_whitespace = strip_whitespace
        self.remove_duplicates = remove_duplicates
        self.expand_abbreviations = expand_abbreviations
        self.normalize_synonyms = normalize_synonyms
        self.validate_ingredients = validate_ingredients
        self.min_length = min_length
        self.max_length = max_length
        self.max_ingredients = max_ingredients

        # Build flat set of known ingredients for validation
        self._known_ingredients: Set[str] = set()
        for category_items in FOOD_CATEGORIES.values():
            self._known_ingredients.update(category_items)

    def preprocess(self, ingredients: List[str]) -> Tuple[List[str], List[ValidationResult]]:
        """
        Preprocess a list of ingredients.

        Returns:
            Tuple of (processed_ingredients, validation_results)
        """
        results = []
        processed = []
        seen = set()

        for ingredient in ingredients:
            result = self._process_single(ingredient)
            results.append(result)

            if result.is_valid:
                # Check for duplicates
                normalized_lower = result.normalized.lower()
                if self.remove_duplicates and normalized_lower in seen:
                    result.warnings.append("Duplicate ingredient removed")
                    continue

                seen.add(normalized_lower)
                processed.append(result.normalized)

        # Limit number of ingredients
        if len(processed) > self.max_ingredients:
            processed = processed[:self.max_ingredients]
            results.append(ValidationResult(
                is_valid=True,
                original="",
                normalized="",
                warnings=[f"Truncated to {self.max_ingredients} ingredients"]
            ))

        return processed, results

    def _process_single(self, ingredient: str) -> ValidationResult:
        """Process a single ingredient."""
        original = ingredient
        warnings = []

        # Basic cleaning
        if self.strip_whitespace:
            ingredient = ingredient.strip()

        if self.lowercase:
            ingredient = ingredient.lower()

        # Remove extra whitespace
        ingredient = re.sub(r'\s+', ' ', ingredient)

        # Length validation
        if len(ingredient) < self.min_length:
            return ValidationResult(
                is_valid=False,
                original=original,
                normalized=ingredient,
                warnings=["Ingredient too short"]
            )

        if len(ingredient) > self.max_length:
            ingredient = ingredient[:self.max_length]
            warnings.append("Ingredient truncated")

        # Expand abbreviations
        if self.expand_abbreviations:
            ingredient = self._expand_abbreviations(ingredient)

        # Normalize synonyms
        if self.normalize_synonyms:
            old_ingredient = ingredient
            ingredient = self._normalize_synonyms(ingredient)
            if ingredient != old_ingredient:
                warnings.append(f"Normalized from '{old_ingredient}'")

        # Validate against known ingredients
        category = None
        if self.validate_ingredients:
            category = self._get_category(ingredient)
            if category is None:
                warnings.append("Unknown ingredient - may affect quality")

        return ValidationResult(
            is_valid=True,
            original=original,
            normalized=ingredient,
            warnings=warnings,
            category=category
        )

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
        words = text.split()
        expanded = []
        for word in words:
            expanded.append(ABBREVIATIONS.get(word.lower(), word))
        return ' '.join(expanded)

    def _normalize_synonyms(self, ingredient: str) -> str:
        """Normalize ingredient synonyms to canonical form."""
        # Check full match first
        if ingredient in SYNONYMS:
            return SYNONYMS[ingredient]

        # Check if any synonym is contained in the ingredient
        for synonym, canonical in SYNONYMS.items():
            if synonym in ingredient:
                return ingredient.replace(synonym, canonical)

        return ingredient

    def _get_category(self, ingredient: str) -> Optional[str]:
        """Get the category of an ingredient if known."""
        ingredient_lower = ingredient.lower()

        for category, items in FOOD_CATEGORIES.items():
            for item in items:
                if item in ingredient_lower or ingredient_lower in item:
                    return category

        return None

    def format_for_model(self, ingredients: List[str], prefix: str = "items: ") -> str:
        """Format ingredients for model input."""
        return prefix + ", ".join(ingredients)
