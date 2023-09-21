import culinary_chatbot_database

# Constants 
'''
Percentage of calories to ingest diary recommend by the OMS 
See "https://metabolicas.sjdhospitalbarcelona.org/etiquetas/piramide-alimentaria"
See "https://blog.zespri.eu/es/calcular-macros/"
'''
# 50-55% of carbohydrates
carbohidrates_min = 0.50
carbohidrates_max = 0.55
# 30-35%  of fats
fats_min = 0.30
fats_max = 0.35
# 12-15% of protein 
protein_min = 0.12
protein_max = 0.15

# 1 gram of carbohydrates is the equivalent to 4 kcal
contribution_per_gram_carbohydrates = 4
# 1 gram of fats is the equivalent to 9 calories
contribution_per_gram_fats = 9
# 1 gram of protein is the equivalent to 4 kcal
contribution_per_gram_protein = 4 


'''
Calculate calories recoomended to be consumed diary. Harris Benedict equation.
See https://es.wikipedia.org/wiki/Ecuaci%C3%B3n_de_Harris-Benedict
'''
def calculate_recommended_calories_diary():
    user_info = culinary_chatbot_database.get_user_info()

    user_weight = user_info['weight_kg']
    user_height = user_info['height_cm']
    user_age = user_info['age']
    activity_factor = user_info['activity_factor']
    
    
    if(user_info['gender'] == 'F'):
        basal_metabolic_rate = ((10 * user_weight) + (6.25 * user_height ) + (5 * user_age)) - 161 
        calories = basal_metabolic_rate * activity_factor
    else:
        basal_metabolic_rate = ((10 * user_weight) + (6.25 * user_height ) + (5 * user_age)) + 5  
        calories = basal_metabolic_rate * activity_factor
        
    return calories


# Update the calories eaten by the user on the chatbot recommend a recipe to the user 
def update_calories_eaten(recipe_calories):
    user_info = culinary_chatbot_database.get_user_info()
    calories_eaten = user_info['calories_eaten_day'] + recipe_calories 
    culinary_chatbot_database.update_user_calories_eaten_day(calories_eaten)


# Calculate calories that the user have left to ingest in a day 
def get_remaining_calories_ingest():
    user_info = culinary_chatbot_database.get_user_info()
    calories_to_eat = user_info['calories_recommended_day'] - user_info['calories_eaten_day'] 

    return calories_to_eat


# Calculate the grams of carbohydrates recommneded for a day
def calculate_grams_carbohydrates(user_calories_recommended):
    kcal = user_calories_recommended * carbohidrates_min
    grams_carbohydrates = kcal / contribution_per_gram_carbohydrates 

    return grams_carbohydrates


# Calculate the grams of fats recommneded for a day
def calculate_grams_fats(user_calories_recommended):
    kcal = user_calories_recommended * fats_min
    grams_fats = kcal / contribution_per_gram_fats 

    return grams_fats


# Calculate the grams of protein recommneded for a day
def calculate_grams_protein(user_calories_recommended):
    kcal = user_calories_recommended * protein_min
    grams_protein = kcal / contribution_per_gram_protein

    return grams_protein


# The percent of each nutrition information(carbohydrates, fats, protein,...) of the recipes correspond to
# the '% daily value' recommended in an standart diet of 2000 calories. It is necesary get the cuantity in grams of 
# nutritions in the recipe portion and then ajust that cuantity to the recommended calories for the user.
def transform_daily_value_in_grams_adjusted_to_user_carbohydrates(nutrition_perc):
    user_info = culinary_chatbot_database.get_user_info()

    # Get the cuantity of the nutrient in the the recipe portion 
    portion_nutrition_in_recipe = user_info['carbohidrates_grams_recommended'] * (nutrition_perc / 100)
    
    # Adjust the cuantity to the calories of the user
    portion = (portion_nutrition_in_recipe * user_info['calories_recommended_day']) / 2000 

    return portion


def transform_daily_value_in_grams_adjusted_to_user_fats(nutrition_perc):
    user_info = culinary_chatbot_database.get_user_info()

    # Get the cuantity of the nutrient in the the recipe portion 
    portion_nutrition_in_recipe = user_info['fats_grams_recommended'] * (nutrition_perc / 100)
    
    # Adjust the cuantity to the calories of the user
    portion = (portion_nutrition_in_recipe * user_info['calories_recommended_day']) / 2000 

    return portion


def transform_daily_value_in_grams_adjusted_to_user_protein(nutrition_perc):
    user_info = culinary_chatbot_database.get_user_info()

    # Get the cuantity of the nutrient in the the recipe portion 
    portion_nutrition_in_recipe = user_info['protein_grams_recommended'] * (nutrition_perc / 100)
    
    # Adjust the cuantity to the calories of the user
    portion = (portion_nutrition_in_recipe * user_info['calories_recommended_day']) / 2000 

    return portion