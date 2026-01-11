"""
Food ingredients database for validation.
Contains common cooking ingredients organized by category.
"""

# Common proteins
PROTEINS = {
    "chicken", "beef", "pork", "lamb", "turkey", "duck", "fish", "salmon", "tuna",
    "shrimp", "prawns", "crab", "lobster", "scallops", "mussels", "clams", "oysters",
    "bacon", "ham", "sausage", "ground beef", "ground turkey", "ground pork",
    "steak", "ribs", "chicken breast", "chicken thigh", "chicken wings",
    "tofu", "tempeh", "seitan", "eggs", "egg whites", "egg yolks",
    "anchovies", "sardines", "cod", "tilapia", "halibut", "trout", "mackerel",
    "venison", "bison", "rabbit", "quail", "goose", "chorizo", "pepperoni",
    "prosciutto", "pancetta", "salami", "hot dog", "meatballs", "ground chicken"
}

# Vegetables
VEGETABLES = {
    "onion", "garlic", "tomato", "potato", "carrot", "celery", "bell pepper",
    "broccoli", "cauliflower", "spinach", "kale", "lettuce", "cabbage",
    "zucchini", "squash", "eggplant", "cucumber", "mushroom", "asparagus",
    "green beans", "peas", "corn", "artichoke", "leek", "shallot", "scallion",
    "green onion", "radish", "turnip", "beet", "parsnip", "sweet potato",
    "yam", "butternut squash", "acorn squash", "spaghetti squash", "pumpkin",
    "brussels sprouts", "bok choy", "swiss chard", "arugula", "watercress",
    "endive", "radicchio", "fennel", "okra", "jalapeno", "serrano", "habanero",
    "poblano", "anaheim", "chili", "red onion", "white onion", "yellow onion",
    "cherry tomato", "roma tomato", "sun-dried tomato", "tomatoes", "onions",
    "potatoes", "carrots", "peppers", "mushrooms", "garlic cloves"
}

# Fruits
FRUITS = {
    "apple", "banana", "orange", "lemon", "lime", "grapefruit", "strawberry",
    "blueberry", "raspberry", "blackberry", "cranberry", "cherry", "grape",
    "peach", "plum", "apricot", "nectarine", "mango", "papaya", "pineapple",
    "coconut", "kiwi", "pomegranate", "fig", "date", "raisin", "prune",
    "watermelon", "cantaloupe", "honeydew", "avocado", "olive", "tomato",
    "pear", "persimmon", "guava", "passion fruit", "lychee", "dragon fruit",
    "starfruit", "tangerine", "clementine", "mandarin", "blood orange",
    "apples", "bananas", "oranges", "lemons", "limes", "berries", "grapes",
    "cherries", "peaches", "plums", "mangoes", "olives", "avocados"
}

# Dairy and eggs
DAIRY = {
    "milk", "cream", "butter", "cheese", "yogurt", "sour cream", "cottage cheese",
    "cream cheese", "ricotta", "mozzarella", "cheddar", "parmesan", "feta",
    "gouda", "brie", "camembert", "blue cheese", "goat cheese", "swiss cheese",
    "provolone", "monterey jack", "pepper jack", "american cheese", "gruyere",
    "mascarpone", "half and half", "heavy cream", "whipping cream", "buttermilk",
    "evaporated milk", "condensed milk", "coconut milk", "almond milk", "soy milk",
    "oat milk", "ice cream", "gelato", "whipped cream", "ghee", "clarified butter"
}

# Grains and starches
GRAINS = {
    "rice", "pasta", "bread", "flour", "oats", "quinoa", "barley", "bulgur",
    "couscous", "farro", "millet", "buckwheat", "cornmeal", "polenta", "grits",
    "noodles", "spaghetti", "penne", "fettuccine", "linguine", "macaroni",
    "lasagna", "tortilla", "pita", "naan", "baguette", "sourdough", "rye",
    "whole wheat", "white rice", "brown rice", "jasmine rice", "basmati rice",
    "arborio rice", "wild rice", "risotto", "orzo", "gnocchi", "ravioli",
    "tortellini", "breadcrumbs", "panko", "croutons", "crackers", "chips",
    "cereal", "granola", "oatmeal", "cornstarch", "tapioca", "arrowroot"
}

# Herbs (fresh and dried)
HERBS = {
    "basil", "oregano", "thyme", "rosemary", "sage", "parsley", "cilantro",
    "dill", "mint", "chives", "tarragon", "marjoram", "bay leaf", "bay leaves",
    "lavender", "lemongrass", "curry leaves", "kaffir lime leaves", "epazote",
    "chervil", "savory", "sorrel", "lovage", "borage", "fresh basil",
    "fresh oregano", "fresh thyme", "fresh rosemary", "fresh parsley",
    "fresh cilantro", "fresh dill", "fresh mint", "dried basil", "dried oregano",
    "dried thyme", "dried rosemary", "dried parsley", "italian seasoning"
}

# Spices
SPICES = {
    "salt", "pepper", "black pepper", "white pepper", "paprika", "cayenne",
    "chili powder", "cumin", "coriander", "turmeric", "ginger", "cinnamon",
    "nutmeg", "cloves", "allspice", "cardamom", "star anise", "fennel seed",
    "mustard seed", "celery seed", "caraway", "anise", "saffron", "sumac",
    "za'atar", "ras el hanout", "garam masala", "curry powder", "five spice",
    "old bay", "cajun seasoning", "taco seasoning", "ranch seasoning",
    "garlic powder", "onion powder", "smoked paprika", "crushed red pepper",
    "red pepper flakes", "sesame seeds", "poppy seeds", "vanilla", "vanilla extract",
    "almond extract", "peppercorns", "sea salt", "kosher salt", "msg"
}

# Oils and fats
OILS = {
    "olive oil", "vegetable oil", "canola oil", "coconut oil", "sesame oil",
    "peanut oil", "sunflower oil", "corn oil", "grapeseed oil", "avocado oil",
    "walnut oil", "truffle oil", "chili oil", "garlic oil", "infused oil",
    "cooking spray", "shortening", "lard", "duck fat", "bacon fat", "schmaltz",
    "oil", "extra virgin olive oil", "light olive oil"
}

# Condiments and sauces
CONDIMENTS = {
    "ketchup", "mustard", "mayonnaise", "hot sauce", "soy sauce", "worcestershire",
    "fish sauce", "oyster sauce", "hoisin sauce", "teriyaki sauce", "bbq sauce",
    "sriracha", "tabasco", "salsa", "guacamole", "hummus", "tahini", "pesto",
    "marinara", "alfredo", "ranch dressing", "italian dressing", "vinaigrette",
    "balsamic vinegar", "red wine vinegar", "white wine vinegar", "apple cider vinegar",
    "rice vinegar", "sherry vinegar", "mirin", "sake", "cooking wine", "white wine",
    "red wine", "beer", "brandy", "rum", "whiskey", "vodka", "tequila",
    "dijon mustard", "honey mustard", "yellow mustard", "relish", "pickles",
    "capers", "anchovies", "miso", "gochujang", "sambal", "harissa", "chimichurri"
}

# Sweeteners
SWEETENERS = {
    "sugar", "brown sugar", "powdered sugar", "honey", "maple syrup", "molasses",
    "agave", "stevia", "corn syrup", "golden syrup", "treacle", "coconut sugar",
    "palm sugar", "muscovado", "turbinado", "demerara", "confectioners sugar",
    "cane sugar", "raw sugar", "simple syrup", "caramel", "chocolate syrup"
}

# Nuts and seeds
NUTS_SEEDS = {
    "almond", "almonds", "walnut", "walnuts", "pecan", "pecans", "cashew", "cashews",
    "peanut", "peanuts", "pistachio", "pistachios", "hazelnut", "hazelnuts",
    "macadamia", "brazil nut", "pine nut", "pine nuts", "chestnut", "chestnuts",
    "sunflower seed", "sunflower seeds", "pumpkin seed", "pumpkin seeds",
    "sesame seed", "chia seed", "chia seeds", "flax seed", "flaxseed", "hemp seed",
    "peanut butter", "almond butter", "cashew butter", "tahini", "nutella"
}

# Legumes
LEGUMES = {
    "black beans", "kidney beans", "pinto beans", "navy beans", "cannellini beans",
    "great northern beans", "lima beans", "chickpeas", "garbanzo beans", "lentils",
    "split peas", "black-eyed peas", "edamame", "soybeans", "fava beans",
    "mung beans", "adzuki beans", "red beans", "white beans", "baked beans",
    "refried beans", "bean sprouts", "hummus"
}

# Baking ingredients
BAKING = {
    "flour", "all-purpose flour", "bread flour", "cake flour", "pastry flour",
    "whole wheat flour", "almond flour", "coconut flour", "self-rising flour",
    "baking powder", "baking soda", "yeast", "active dry yeast", "instant yeast",
    "cream of tartar", "gelatin", "pectin", "xanthan gum", "cocoa powder",
    "chocolate chips", "dark chocolate", "milk chocolate", "white chocolate",
    "unsweetened chocolate", "bittersweet chocolate", "semisweet chocolate",
    "food coloring", "sprinkles", "frosting", "fondant", "marzipan"
}

# Canned and preserved goods
CANNED = {
    "canned tomatoes", "tomato paste", "tomato sauce", "diced tomatoes",
    "crushed tomatoes", "tomato puree", "canned beans", "canned corn",
    "canned peas", "canned tuna", "canned salmon", "canned chicken",
    "coconut cream", "evaporated milk", "condensed milk", "stock", "broth",
    "chicken broth", "beef broth", "vegetable broth", "chicken stock",
    "beef stock", "vegetable stock", "bouillon", "consomme", "bone broth"
}

# Combine all categories
VALID_INGREDIENTS = (
    PROTEINS | VEGETABLES | FRUITS | DAIRY | GRAINS | HERBS |
    SPICES | OILS | CONDIMENTS | SWEETENERS | NUTS_SEEDS |
    LEGUMES | BAKING | CANNED
)

# Common cooking units for parsing
COOKING_UNITS = {
    "cup", "cups", "tablespoon", "tablespoons", "tbsp", "teaspoon", "teaspoons",
    "tsp", "ounce", "ounces", "oz", "pound", "pounds", "lb", "lbs",
    "gram", "grams", "g", "kilogram", "kilograms", "kg", "ml", "milliliter",
    "milliliters", "liter", "liters", "l", "quart", "quarts", "qt", "pint",
    "pints", "pt", "gallon", "gallons", "gal", "pinch", "dash", "handful",
    "bunch", "sprig", "sprigs", "clove", "cloves", "slice", "slices",
    "piece", "pieces", "can", "cans", "jar", "jars", "package", "packages",
    "box", "boxes", "bag", "bags", "stick", "sticks", "head", "heads",
    "stalk", "stalks", "leaf", "leaves", "drop", "drops"
}

# Common cooking verbs for coherence checking
COOKING_VERBS = {
    "preheat", "heat", "warm", "boil", "simmer", "sauté", "saute", "fry",
    "deep fry", "pan fry", "stir fry", "bake", "roast", "broil", "grill",
    "barbecue", "smoke", "steam", "poach", "braise", "stew", "blanch",
    "reduce", "deglaze", "caramelize", "brown", "sear", "toast", "char",
    "mix", "stir", "whisk", "beat", "fold", "blend", "puree", "mash",
    "chop", "dice", "mince", "slice", "julienne", "cube", "shred", "grate",
    "peel", "core", "seed", "zest", "juice", "squeeze", "crush", "pound",
    "knead", "roll", "shape", "form", "spread", "layer", "stuff", "fill",
    "coat", "dredge", "bread", "batter", "marinate", "season", "rub",
    "drizzle", "pour", "add", "combine", "incorporate", "toss", "flip",
    "turn", "remove", "transfer", "drain", "strain", "cool", "chill",
    "refrigerate", "freeze", "thaw", "rest", "let", "allow", "set",
    "garnish", "serve", "plate", "top", "sprinkle", "finish", "enjoy",
    "taste", "adjust", "cover", "uncover", "wrap", "unwrap"
}

# Words that indicate end of recipe
ENDING_VERBS = {"serve", "garnish", "enjoy", "plate", "top", "sprinkle", "finish"}

# Words that indicate beginning of recipe
STARTING_VERBS = {"preheat", "heat", "prepare", "gather", "combine", "mix"}


def get_valid_ingredients():
    """Return the set of all valid ingredients."""
    return VALID_INGREDIENTS


def get_cooking_units():
    """Return the set of cooking units."""
    return COOKING_UNITS


def get_cooking_verbs():
    """Return the set of cooking verbs."""
    return COOKING_VERBS


def normalize_ingredient(ingredient: str) -> str:
    """
    Normalize an ingredient string by removing quantities and units.
    Returns the base ingredient name in lowercase.
    """
    import re

    text = ingredient.lower().strip()

    # Remove quantities (numbers, fractions)
    text = re.sub(r'\d+[\d/\.\s]*', '', text)

    # Remove units
    for unit in COOKING_UNITS:
        text = re.sub(rf'\b{unit}\b', '', text, flags=re.IGNORECASE)

    # Remove common modifiers
    modifiers = ['fresh', 'dried', 'chopped', 'diced', 'minced', 'sliced',
                 'grated', 'shredded', 'crushed', 'ground', 'whole', 'large',
                 'medium', 'small', 'thin', 'thick', 'cooked', 'raw', 'ripe',
                 'unripe', 'frozen', 'canned', 'packed', 'loosely', 'firmly',
                 'about', 'approximately', 'optional', 'to taste', 'divided']
    for mod in modifiers:
        text = re.sub(rf'\b{mod}\b', '', text, flags=re.IGNORECASE)

    # Remove extra punctuation and whitespace
    text = re.sub(r'[,\(\)\[\]]', '', text)
    text = ' '.join(text.split())

    return text.strip()
