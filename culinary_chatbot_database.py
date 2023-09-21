import sqlite3
import pandas as pd
import random 
import json
import re

from pattern.text.en import singularize


'''
Basic functions of a database
'''
# Connect the database
def connect_database():
    conn = sqlite3.connect('culinaryDB.db')
    return conn


# Disconnect the database
def disconnect_database(conn):
    conn.close()


# Create a cursor object
def create_cursor(conn):
    return conn.cursor()


# Add table to database
def create_table(schema):
    conn = connect_database()
    cursor = create_cursor(conn)
    cursor.execute(schema)
    conn.commit()


# Consult a table
def consult_table(table_name):
    conn = connect_database()
    query = f'SELECT * FROM {table_name}'
    return pd.read_sql_query(query, conn)


# Drop a table     
def drop_table(table_name):
    conn = connect_database()
    cursor = create_cursor(conn)
    query = f'DROP TABLE IF EXISTS {table_name}'
    cursor.execute(query)
    conn.commit()


# Insert a row in a table
def insert_row(table_name, new_row):
    conn = connect_database()
    cursor = create_cursor(conn)
    values = []
    for value in new_row:
        values.append(str(value))
    query = f'INSERT INTO {table_name} VALUES ({",".join("?" for v in values)})'
    cursor.execute(query, values)
    conn.commit()


# Delete a row from a table
def delete_row(table_name, condition):
    conn = connect_database()
    cursor = create_cursor(conn)
    query = f'DELETE FROM {table_name} WHERE {condition}'
    cursor.execute(query)
    conn.commit()


# Update a row from a table
def update_row(table_name, column_name, new_value):
    conn = connect_database()
    cursor = create_cursor(conn)
    query = f'UPDATE {table_name} SET {column_name} = ?'
    cursor.execute(query, new_value)
    conn.commit()



'''
Create the necesary tables
'''
# Create the necesary tables
def create_tables():
    # Add tables Recipes, Alergies, NutritionAliments, User and ConversationHistoryto the database 
    schema_recipes = '''CREATE TABLE IF NOT EXISTS recipes (
                    recipe_id INTEGER PRIMARY KEY,
                    name TEXT,
                    minutes INTEGER,
                    tags TEXT,
                    n_steps INTEGER,
                    steps TEXT,
                    ingredients TEXT,
                    n_ingredients INTEGER,
                    ingredients_cuantity TEXT,
                    serving_size TEXT,
                    servings INTEGER,
                    search_terms TEXT,
                    calories REAL, 
                    total_fat_perc REAL,
                    sugar_perc REAL, 
                    sodium_perc REAL,
                    protein_perc REAL, 
                    saturated_fat_perc REAL,
                    carbohydrates_perc REAL,
                    allergies TEXT,
                    diets TEXT,
                    meal TEXT)'''

    schema_ingredients = '''CREATE TABLE IF NOT EXISTS ingredients (
                            ingredient_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                            ingredient_name TEXT)'''

    schema_recipe_ingredient = '''CREATE TABLE IF NOT EXISTS recipe_ingredient (
                            recipe_id INTEGER,
                            ingredient_id INTEGER,
                            PRIMARY KEY (recipe_id, ingredient_id),
                            FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id),
                            FOREIGN KEY (ingredient_id) REFERENCES ingredients(ingredient_id))'''

    schema_nutrition_types_aliments = '''CREATE TABLE IF NOT EXISTS nutrition_types_aliments (
                                    aliment_type_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    aliment_type_name TEXT,
                                    energ_Kcal INTEGER,
                                    protein_g REAL,
                                    carbohydrates_g REAL,
                                    saturated_fat_g REAL)'''
    
    schema_nutrition_ingredients_by_types = '''CREATE TABLE IF NOT EXISTS nutrition_ingredients_by_types (
                                        ingredient_id INTEGER,
                                        aliment_type_id INTEGER,
                                        energ_Kcal INTEGER,
                                        protein_g REAL,
                                        carbohydrates_g REAL,
                                        saturated_fat_g REAL,
                                        PRIMARY KEY (aliment_type_id, ingredient_id),
                                        FOREIGN KEY (aliment_type_id) REFERENCES nutrition_types_aliments(aliment_type_id),
                                        FOREIGN KEY (ingredient_id) REFERENCES ingredients(ingredient_id))'''

    schema_allergies_grouped_by_aliments = '''CREATE TABLE IF NOT EXISTS allergies_grouped_by_aliments (
                                                allergy_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                allergy_name TEXT,
                                                aliments TEXT)'''

    schema_allergies_ingredients = '''CREATE TABLE IF NOT EXISTS allergies_ingredients (
                                    allergy_id INTEGER,
                                    ingredient_id INTEGER,
                                    PRIMARY KEY (allergy_id, ingredient_id),
                                    FOREIGN KEY (allergy_id) REFERENCES allergies_grouped_by_aliments(allergy_id),
                                    FOREIGN KEY (ingredient_id) REFERENCES ingredients(ingredient_id))'''
    
    schema_diets = '''CREATE TABLE IF NOT EXISTS diets (
                    diet_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    diet_name TEXT)'''
    
    schema_recipe_diets = '''CREATE TABLE IF NOT EXISTS recipe_diets (
                            recipe_id INTEGER,
                            diet_id INTEGER,
                            PRIMARY KEY (recipe_id, diet_id),
                            FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id),
                            FOREIGN KEY (diet_id) REFERENCES diets(diet_id))'''
    
    schema_user = '''CREATE TABLE IF NOT EXISTS user (
                    user_name TEXT,
                    weight_kg REAL, 
                    height_cm REAL,
                    age INTEGER,
                    gender TEXT, 
                    activity_factor REAL,
                    allergies TEXT,
                    diets TEXT,
                    calories_recommended_day REAL DEFAULT 0.0,
                    calories_eaten_day REAL DEFAULT 0.0,
                    carbohidrates_grams_recommended REAL DEFAULT 0.0, 
                    fats_grams_recommended REAL DEFAULT 0.0,
                    protein_grams_recommended REAL DEFAULT 0.0,
                    carbohidrates_grams_day REAL DEFAULT 0.0, 
                    fats_grams_day REAL DEFAULT 0.0,
                    protein_grams_day REAL DEFAULT 0.0)'''

    schema_conversation_history = '''CREATE TABLE IF NOT EXISTS conversation_history (
                                    conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    date DATE,
                                    question TEXT, 
                                    answer TEXT,
                                    tag TEXT)'''
    
      
    # Create a table to save the recipes
    create_table(schema_recipes)
    
    # Create a table to save the nutritional values of the aliments 
    create_table(schema_nutrition_types_aliments)

    # Create the associative table to save the ingredients and their respective types/states with the nutrition values 
    create_table(schema_nutrition_ingredients_by_types)
    
    # Create a table to save the ingredients
    create_table(schema_ingredients)
    
    # Create the associative table to save the recipes and their respective ingredients relation
    create_table(schema_recipe_ingredient)

    # Create the table to save the allergies and the respective aliments that provoke that allergy
    create_table(schema_allergies_grouped_by_aliments)
    
    # Create the associative table between the ingredients table and the allergies table
    create_table(schema_allergies_ingredients)

    # Create a table to save the diets
    create_table(schema_diets)

    # Create the associative table between the recipes table and the diets table
    create_table(schema_recipe_diets)

    # Create a table to save the information about the user
    create_table(schema_user)
    
    # Initialize the values of the table
    conn = connect_database()
    cursor = create_cursor(conn)

    update_query = '''UPDATE user
                  SET weight_kg = 0.0,
                      height_cm = 0.0,
                      activity_factor = 0.0,
                      calories_recommended_day = 0.0,
                      calories_eaten_day = 0.0,
                      carbohidrates_grams_recommended = 0.0,
                      fats_grams_recommended = 0.0,
                      protein_grams_recommended = 0.0,
                      carbohidrates_grams_day = 0.0,
                      fats_grams_day = 0.0,
                      protein_grams_day = 0.0'''

    cursor.execute(update_query)
    conn.commit()
   
    # Create a table to save the conversations history
    create_table(schema_conversation_history)
        

'''
Functions to add data to the tables 
'''    
# Add data to the table recipes
def add_data_to_table_recipes(dataframe):
    conn = connect_database()
    cursor = create_cursor(conn)

    # Insert dataframe into the table recipes
    for index, row in dataframe.iterrows():
        diets_list = row['diets']
        diets_str = ",".join(diets_list)

        allergies_list = row['allergies']
        allergies_str = ",".join(allergies_list)

        values = (row['recipe_id'], row['name'], row['minutes'], json.dumps(row['tags']), row['n_steps'], json.dumps(row['steps']),
                    json.dumps(row['ingredients']), row['n_ingredients'], json.dumps(row['ingredients_cuantity']), row['serving_size'], 
                    row['servings'], row['search_terms'], row['calories'], row['total_fat_perc'], row['sugar_perc'], row['sodium_perc'], 
                    row['protein_perc'], row['saturated_fat_perc'],row['carbohydrates_perc'], allergies_str, diets_str, row['meal'])
        
        query = ("INSERT INTO recipes (recipe_id, name, minutes, tags, n_steps, steps, ingredients, n_ingredients, ingredients_cuantity, "
                "serving_size, servings, search_terms, calories, total_fat_perc, sugar_perc, sodium_perc, protein_perc, saturated_fat_perc, "
                "carbohydrates_perc, allergies, diets, meal) " 
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
        cursor.execute(query, values)

    conn.commit()


# Add data to the table ingredients 
def add_data_to_table_ingredients(dataframe):
    conn = connect_database()
    cursor = create_cursor(conn)

    # Insert dataframe into the table ingredients 
    for index, row in dataframe.iterrows():
        ingredient_name = row['ingredient']
        
        query = "INSERT INTO ingredients (ingredient_name) VALUES (?)"
        cursor.execute(query, (ingredient_name,))

    conn.commit()


# Add data to the table ingredients 
def add_data_to_table_recipe_ingredient(dataframe):
    conn = connect_database()
    cursor = create_cursor(conn)

    # Insert dataframe into the table recipe_ingredient
    for index, row in dataframe.iterrows():
        recipe_ids = row['recipe_id']
        ingredient_name = row['ingredient']

        # Query to retrieve the ingredient_id with ingredient_name
        select_query = "SELECT ingredient_id FROM ingredients WHERE ingredient_name = ?"
        cursor.execute(select_query, (ingredient_name,))
        result = cursor.fetchone()

        if result is not None:
            ingredient_id = result[0]

            for recipe_id in recipe_ids:
                # Check if the combination of recipe_id and ingredient_id already exists
                check_query = "SELECT COUNT(*) FROM recipe_ingredient WHERE recipe_id = ? AND ingredient_id = ?"
                cursor.execute(check_query, (recipe_id, ingredient_id))
                count = cursor.fetchone()[0]

                if count == 0:
                    # Insert into recipe_ingredient table for each recipe_id
                    insert_query = "INSERT INTO recipe_ingredient (recipe_id, ingredient_id) VALUES (?, ?)"
                    cursor.execute(insert_query, (recipe_id, ingredient_id))
            
    conn.commit()


# Add data to the table nutrition_types_aliments
def add_data_to_table_nutrition_types_aliments(dataframe):
    conn = connect_database()
    cursor = create_cursor(conn)

    # Insert dataframe into the table recipes
    for index, row in dataframe.iterrows():
        values = (row['aliment_type_name'], row['energ_Kcal'], row['protein_g'], row['carbohydrates_g'], row['saturated_fat_g'])
        
        query = f"INSERT INTO nutrition_types_aliments (aliment_type_name, energ_Kcal, protein_g, carbohydrates_g, saturated_fat_g) VALUES (?, ?, ?, ?, ?)"
        cursor.execute(query, values)

    conn.commit()


# Add data to the table nutrition_ingredients_by_types
def add_data_to_table_nutrition_ingredients_by_types():
    conn = connect_database()
    cursor = create_cursor(conn)

     # Get the ingredients id and their names from the table ingredients
    select_ingredients_query = "SELECT ingredient_id, ingredient_name FROM ingredients"
    cursor.execute(select_ingredients_query)
    ingredients = cursor.fetchall()

    for ingredient in ingredients:
        ingredient_id = ingredient[0]
        ingredient_name = ingredient[1]

        # Find matching aliment types in nutrition_types_aliments
        select_aliment_types_query = "SELECT aliment_type_id, energ_Kcal, protein_g, carbohydrates_g, saturated_fat_g FROM nutrition_types_aliments WHERE LOWER(aliment_type_name) LIKE ?"
        cursor.execute(select_aliment_types_query, ('%' + ingredient_name.lower() + '%',))
        aliment_types = cursor.fetchall()

        for aliment_type in aliment_types:
            aliment_type_id = aliment_type[0]
            energ_Kcal = aliment_type[1]
            protein_g = aliment_type[2]
            carbohydrates_g = aliment_type[3]
            saturated_fat_g = aliment_type[4]

            # Insert into the nutrition_ingredients_by_types table the values
            insert_query = "INSERT INTO nutrition_ingredients_by_types (ingredient_id, aliment_type_id, energ_Kcal, protein_g, carbohydrates_g, saturated_fat_g) VALUES (?, ?, ?, ?, ?, ?)"
            cursor.execute(insert_query, (ingredient_id, aliment_type_id, energ_Kcal, protein_g, carbohydrates_g, saturated_fat_g))

    conn.commit()


# Add data to the table allergies_grouped_by_aliemnts
def add_data_to_table_allergies_grouped_by_aliments(allergies_grouped_by_aliments):
    conn = connect_database()
    cursor = create_cursor(conn)

    # Insert dataframe into the table ingredients 
    for index, row in allergies_grouped_by_aliments.iterrows():
        allergy_name = row['allergy_name']
        aliments = json.dumps(row['aliments']) 
        
        query = "INSERT INTO allergies_grouped_by_aliments (allergy_name, aliments) VALUES (?, ?)"
        cursor.execute(query, (allergy_name, aliments))

    conn.commit()


# Add data to the table allergies_ingredients
def add_data_to_table_allergies_ingredients():
    conn = connect_database()
    cursor = create_cursor(conn)

    # Get all ingredients from the ingredients table
    select_ingredients_query = "SELECT ingredient_id, ingredient_name FROM ingredients"
    cursor.execute(select_ingredients_query)
    ingredients = cursor.fetchall()
    
    # Get all allergies and their associated allergenic foods from the allergies_grouped_by_aliments table
    select_allergies_query = "SELECT allergy_id, allergy_name, aliments FROM allergies_grouped_by_aliments"
    cursor.execute(select_allergies_query)
    allergies = cursor.fetchall()
    
    for allergy in allergies:
        allergy_id = allergy[0]
        allergy_name = allergy[1]
        aliments = allergy[2]
        
        for ingredient in ingredients:
            ingredient_id = ingredient[0]
            ingredient_name = ingredient[1]
            
            # Convert ingredient and allergy names to singular form
            singular_ingredient_name = singularize(ingredient_name)
            singular_allergy_name = singularize(allergy_name)
            
            # Check if the singular ingredient name or the singular allergy name is present in the aliments column
            if singular_ingredient_name.lower() in aliments.lower() or singular_allergy_name.lower() in aliments.lower():
                # Check if the record already exists in the allergies_ingredients table
                select_query = "SELECT * FROM allergies_ingredients WHERE allergy_id = ? AND ingredient_id = ?"
                cursor.execute(select_query, (allergy_id, ingredient_id))
                result = cursor.fetchone()
                
                # Insert the record if it doesn't already exist
                if result is None:
                    insert_query = "INSERT INTO allergies_ingredients (allergy_id, ingredient_id) VALUES (?, ?)"
                    cursor.execute(insert_query, (allergy_id, ingredient_id))

    conn.commit()


# Add data to the table diets
def add_data_to_table_diets():
    conn = connect_database()
    cursor = create_cursor(conn)

    diets = ["vegetarian", "vegan"]
    for diet in diets:
        cursor.execute("INSERT INTO diets (diet_name) VALUES (?)", (diet,))

    conn.commit()


# Add data to the table recipe_diets
def add_data_to_table_recipe_diets():   
    conn = connect_database()
    cursor = create_cursor(conn)

    cursor.execute("SELECT recipe_id, diets FROM recipes")
    recipes = cursor.fetchall()

    for recipe in recipes:
        recipe_id = recipe[0]
        diets = recipe[1].split(',')

        for diet in diets:
            # Get the id of the diet with the diet_name
            cursor.execute("SELECT diet_id FROM diets WHERE diet_name = ?", (diet,))
            diet_id_data = cursor.fetchone()

            if diet_id_data:
                diet_id = diet_id_data[0]
                cursor.execute("INSERT INTO recipe_diets (recipe_id, diet_id) VALUES (?, ?)", (recipe_id, diet_id))
    
    conn.commit()


# Add the user information to the table user
def add_user_info(user_name, weight_kg, height_cm, age, gender, activity_factor, allergies, diets):
    conn = connect_database()
    cursor = create_cursor(conn)
    query = 'INSERT INTO user (user_name, weight_kg, height_cm, age, gender, activity_factor, allergies, diets) VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
    values = (user_name, weight_kg, height_cm, age, gender, activity_factor, allergies, diets)
    cursor.execute(query, values)
    conn.commit()
    conn.close()


# Update the user information when the user wants to update his information
def update_user_info(weight_kg, height_cm, age, gender, activity_factor, allergies, diets):
    conn = connect_database()
    cursor = create_cursor(conn)
    cursor.execute("UPDATE user SET weight_kg = ?, height_cm = ?, age = ?, gender = ?, activity_factor = ?, allergies = ?, diets = ?",
                    (weight_kg, height_cm, age, gender, activity_factor, allergies, diets))
    conn.commit()    


# Update the user calories recommended 
def update_user_calories_recommended(user_calories_recommended):
    conn = connect_database()
    cursor = create_cursor(conn)
    cursor.execute("UPDATE user SET calories_recommended_day = ?", (user_calories_recommended,))
    conn.commit()


# Update the recommended grams of carbohydrates, fats and protein for the user per day 
def update_user_grams_recommended(carbohydrates_grams, fats_grams, protein_grams):
    conn = connect_database()
    cursor = create_cursor(conn)
    cursor.execute("UPDATE user SET carbohidrates_grams_recommended = ?, fats_grams_recommended = ?, protein_grams_recommended = ?",
                   (carbohydrates_grams, fats_grams, protein_grams))
    conn.commit()


# Update the calories that the user ingests
def update_calories_eaten(calories_eaten):
    conn = connect_database()
    cursor = create_cursor(conn)
    cursor.execute("UPDATE user SET calories_eaten_day = ?",
                   (calories_eaten,))
    conn.commit()


# Update the grams of carbohydrates, fats and protein consumed 
def update_nutrition_grams_day(carbohydrates_grams, fats_grams, protein_grams):
    conn = connect_database()
    cursor = create_cursor(conn)
    cursor.execute("UPDATE user SET carbohidrates_grams_day = ?, fats_grams_day = ?, protein_grams_day = ?",
                   (carbohydrates_grams, fats_grams, protein_grams))
    conn.commit()


# Update the user calories eaten in a day 
def update_user_calories_eaten_day(calories_eaten):
    conn = connect_database()
    cursor = create_cursor(conn)
    cursor.execute("UPDATE user SET calories_eaten_day = ?", (calories_eaten,))
    conn.commit()


# Put all the user nutrition information to 0 for a new day
def restart_user_nutrition_values():
    conn = connect_database()
    cursor = create_cursor(conn)
    cursor.execute('''UPDATE user SET  calories_eaten_day = 0.0, carbohidrates_grams_day = 0.0, fats_grams_day = 0.0,
            protein_grams_day = 0.0''')
    conn.commit()


'''
Specific database functions for the culinary chatbot
 '''
# Get all the allergies from the table allergies_grouped_by_aliments
def get_all_allergies():
    conn = connect_database()
    cursor = create_cursor(conn)
    cursor.execute("SELECT allergy_name FROM allergies_grouped_by_aliments")
    allergies = cursor.fetchall()
    allergies = [row[0] for row in allergies]
    conn.close()

    return allergies


# Get all the diets from the table allergies_grouped_by_aliments
def get_all_diets():
    conn = connect_database()
    cursor = create_cursor(conn)
    cursor.execute("SELECT diet_name FROM diets")
    diets = cursor.fetchall()
    diets = [row[0] for row in diets]  
    conn.close()

    return diets


# Get the information from the table user
def get_user_info():
    conn = connect_database()
    cursor = create_cursor(conn)
    cursor.execute('SELECT * FROM user')
    user_info = cursor.fetchone()

    if user_info:
        (user_name, weight_kg, height_cm, age, gender, activity_factor, allergies, diets, calories_recommended_day,
        calories_eaten_day, carbohidrates_grams_recommended, fats_grams_recommended, protein_grams_recommended,
        carbohidrates_grams_day, fats_grams_day, protein_grams_day) = user_info
        result = {
            'user_name': user_name,
            'weight_kg': weight_kg,
            'height_cm': height_cm,
            'age': age,
            'gender': gender,
            'activity_factor': activity_factor,
            'allergies': allergies,
            'diets': diets,
            'calories_recommended_day': calories_recommended_day,
            'calories_eaten_day': calories_eaten_day,
            'carbohidrates_grams_recommended': carbohidrates_grams_recommended,
            'fats_grams_recommended': fats_grams_recommended,
            'protein_grams_recommended': protein_grams_recommended,
            'carbohidrates_grams_day': carbohidrates_grams_day,
            'fats_grams_day': fats_grams_day,
            'protein_grams_day': protein_grams_day
        }
    conn.close()

    return result 


# Save the conversations between the user and the chatbot with the current date and the tag predicted by the model
def save_conversation_in_db(date, question, answer, tag):
    conn = connect_database()
    cursor = create_cursor(conn)
    cursor.execute('INSERT INTO conversation_history (date, question, answer, tag) VALUES (?, ?, ?, ?)', (date, question, answer, tag))
    conn.commit()


# Get the recipes with the user preferences(allergies and diets)
def get_recipes():
    conn = connect_database()
    cursor = create_cursor(conn)
    cursor.execute('SELECT * FROM Recipes')
    recipes = cursor.fetchall()
    recipes_with_preferences = []

    user_info = get_user_info()
    allergies = user_info['allergies']
    diets = user_info['diets']

    for recipe in recipes:
        if check_preferences_in_recipe(recipe, allergies, diets):
            recipes_with_preferences.append(recipe)
    return recipes_with_preferences


# Get a random recipe 
def get_random_recipe():
    recipes_with_preferences = get_recipes()
    if recipes_with_preferences:
        selected_recipe = random.choice(recipes_with_preferences)
        recipe_name = selected_recipe[1]
        recipe_to_return = f'{recipe_name}\n\nIngredients: {selected_recipe[8]}\n\nSteps: {selected_recipe[5]}\n'
        return recipe_to_return, recipe_name
    else:
        return None, None


# Get recipes by name 
def get_recipes_by_name(recipe_name):
    conn = connect_database()
    cursor = create_cursor(conn)
    user_info = get_user_info()
    allergies = user_info['allergies']
    diets = user_info['diets']

    cursor.execute("SELECT * FROM recipes WHERE name LIKE ?", ('%' + recipe_name + '%',))
    results = cursor.fetchall()
    conn.close()

    recipes_with_preferences = []
    for recipe in results:
        if any(word.lower() in recipe['name'].lower() for word in recipe_name.split()):
            if check_preferences_in_recipe(recipe, allergies, diets):
                recipes_with_preferences.append(recipe)

    return recipes_with_preferences


# Check if the recipe respects the user preferences 
def check_preferences_in_recipe(recipe, allergies, diets):
    conn = connect_database()
    cursor = create_cursor(conn)
    is_recipe_ok = True

    # Check the allergies
    if allergies:
        allergies = [allergy.strip() for allergy in allergies.split(",")]
        for allergy in allergies:
            cursor.execute("SELECT allergy_id FROM allergies_grouped_by_aliments WHERE allergy_name = ?", (allergy,))
            allergy_id = cursor.fetchone()

            if allergy_id:
                cursor.execute("SELECT * FROM allergies_ingredients ai JOIN recipe_ingredient ri ON ai.ingredient_id = ri.ingredient_id WHERE ai.allergy_id = ? AND ri.recipe_id = ?", (allergy_id[0], recipe[0]))
                result = cursor.fetchone()

                if result is not None:
                    is_recipe_ok = False
                    break
            
    # Check the diets
    if diets:
        diets = [diet.strip() for diet in diets.split(",")]
        for diet in diets:
            cursor.execute("SELECT diet_id FROM diets WHERE diet_name = ?", (diet,))
            diet_id = cursor.fetchone()

            if diet_id:
                cursor.execute("SELECT * FROM recipe_diets WHERE recipe_id = ? AND diet_id = ?", (recipe[0], diet_id[0]))
                result = cursor.fetchone()

                # If no row is found in the recipe_diets table, the recipe does not comply with the diet.
                if result is None:
                    is_recipe_ok = False
                    break

    return is_recipe_ok


# Get recipe by id
def get_recipe_by_id(recipe_id):
    conn = connect_database()
    cursor = create_cursor(conn)

    query = "SELECT name, ingredients_cuantity, steps FROM recipes WHERE recipe_id = ?"
    cursor.execute(query, (recipe_id,))
    recipe = cursor.fetchone()

    recipe_to_return = None

    if recipe:
        recipe_name = recipe[0]
        ingredients_cuantity = recipe[1]
        steps = recipe[2]

        recipe_to_return = {
            'recipe_name': recipe_name,
            'ingredients_cuantity': ingredients_cuantity,
            'steps': steps
        }

    conn.close()

    return recipe_to_return


# Get recipe for a given ingredients 
def get_recipe_by_ingredients(filtered_words):
    conn = connect_database()
    cursor = create_cursor(conn)

    user_info = get_user_info()
    allergies = user_info['allergies']
    diets = user_info['diets']
    ingredient_ids = []
    finish = False

    # Get the ingredients id
    for word in filtered_words:
        cursor.execute('SELECT ingredient_id FROM ingredients WHERE LOWER(ingredient_name) LIKE ?', (word.lower() + '%',))
        result = cursor.fetchall()

        for ingredient in result:
            ingredient_id = ingredient[0]
            ingredient_ids.append(ingredient_id)

    # Get the recipe id based on ingredient_ids and preferences
    for ingredient_id in ingredient_ids:
        query = "SELECT recipe_id FROM recipe_ingredient WHERE ingredient_id = ?"
        cursor.execute(query, (ingredient_id,))
        result = cursor.fetchall()

        for recipe in result:
            if check_preferences_in_recipe(recipe, allergies, diets):
                recipe = get_recipe_by_id(recipe[0])
                finish = True
                break
        if finish:
            break    
    
    conn.close()

    return recipe

    
# Get the recipe name from an answer given from the chatbot to the user
def get_recipe_name_from_answer(answer):
    recipe_name = None
    answer_str = ' '.join(answer)
    recipe_info = answer_str.split("Ingredients", 1)
    if len(recipe_info) > 0:
        recipe_part = recipe_info[0]
        # The name of the recipe is going to be after "recipe:"
        recipe_match = re.search(r"recipe:\s+(.*)", recipe_part)
        if recipe_match:
            recipe_name = recipe_match.group(1).strip()
    return recipe_name


# Get the recipe nutritional values(calories, carbohydrates, fats and protein) from the recipe
def get_recipe_nutritional_values(recipe_name): 
    conn = connect_database()
    cursor = create_cursor(conn)

    # Get the nutrition values of the recipe with recipe_name
    cursor.execute("SELECT calories, total_fat_perc, protein_perc, saturated_fat_perc, carbohydrates_perc FROM recipes WHERE name = ?", (recipe_name,))
    result = cursor.fetchone()

    conn.close()

    if result:
        nutrition_values = {
            'calories': result[0],
            'fats_perc': result[1] + result[3],
            'protein_perc': result[2],
            'carbohydrates_perc': result[4]
        }

        return nutrition_values
    else:
        return None
    

# Get calories from an aliment/ingredient
def get_nutritional_values_by_aliment(filtered_words):
    conn = connect_database()
    cursor = create_cursor(conn)
    aliment_list = []
    response = ''

    # Check every filtered word, to see if matches with an aliment name (aliment_type_name)
    for word in filtered_words:
        cursor.execute('SELECT aliment_type_name, energ_Kcal, protein_g, carbohydrates_g, saturated_fat_g FROM nutrition_types_aliments WHERE LOWER(aliment_type_name) LIKE ?', (word.lower() + '%',))
        aliment_info = cursor.fetchall()

        if aliment_info:
            for row in aliment_info:
                aliment_list.append(row)

    if aliment_list:
        for aliment_info in aliment_list:
            # Prepare the response with the nutrition values of the aliment detected before
            response += f'\n{aliment_info[0]}: Calories: {aliment_info[1]} Protein: {aliment_info[2]} Carbohydrates: {aliment_info[3]} Saturated fat: {aliment_info[4]}\n'
        return aliment_list, response.rstrip('\n')
    else:
        return None, 'Not found'
    

# Get the previous answers where the tag was equal to "recipe" and date is the current date
def get_previous_recipe_tag_answers(date):
    conn = connect_database()
    cursor = create_cursor(conn)

    query = "SELECT answer FROM conversation_history WHERE tag = 'recipe' AND date = ? ORDER BY conversation_id DESC LIMIT 1"
    cursor.execute(query, (date,))
    previous_recipe_answers = cursor.fetchall()

    conn.close()

    previous_recipe_answers_list = [answer[0] for answer in previous_recipe_answers]

    return previous_recipe_answers_list


# Get recipe id by name 
def get_recipe_id_by_name(recipe_name):
    conn = connect_database()
    cursor = create_cursor(conn)

    # Get the recipe id by its name 
    query = "SELECT recipe_id FROM recipes WHERE name = ?"
    cursor.execute(query, (recipe_name,))
    recipe_id = cursor.fetchone()[0]

    conn.close()

    return recipe_id


# Get the ingredients that are used in the recipe
def get_used_ingredients_from_recipe(recipe_id):
    conn = connect_database()
    cursor = create_cursor(conn)

    # Exclude the ingredients salt, pepper, olive oil
    exceptional_ingredients = ['salt', 'pepper', 'olive oil']
    exceptional_ingredient_ids = []

    for ingredient in exceptional_ingredients:
        cursor.execute("SELECT ingredient_id FROM ingredients WHERE ingredient_name = ?", (ingredient,))
        ingredient_id = cursor.fetchone()
        if ingredient_id:
            exceptional_ingredient_ids.append(ingredient_id[0])

    # Get the ingredients id where the recipe_id matches and exclude the exceptional ingredients
    cursor.execute("SELECT ingredient_id FROM recipe_ingredient WHERE recipe_id = ?", (recipe_id,))
    recipe_ingredient_ids = [row[0] for row in cursor.fetchall() if row[0] not in exceptional_ingredient_ids]

    conn.close()

    return recipe_ingredient_ids


# Get recipe information 
def get_recipe_info(recipe_id):
    conn = connect_database()
    cursor = create_cursor(conn)

    query = "SELECT * FROM recipe WHERE recipe_id = ?"
    cursor.execute(query, (recipe_id,))
    recipe_columns = cursor.fetchone()

    conn.close()

    if recipe_columns:
        result = {
            'recipe_id': recipe_columns[0],
            'name': recipe_columns[1],
            'minutes': recipe_columns[2],
            'tags': recipe_columns[3],
            'n_steps': recipe_columns[4],
            'steps': recipe_columns[5],
            'ingredients': recipe_columns[6],
            'n_ingredients': recipe_columns[7],
            'ingredients_cuantity': recipe_columns[8],
            'serving_size': recipe_columns[9],
            'servings': recipe_columns[10],
            'search_terms': recipe_columns[11],
            'calories': recipe_columns[12],
            'total_fat_perc': recipe_columns[13],
            'sugar_perc': recipe_columns[14],
            'sodium_perc': recipe_columns[15],
            'protein_perc': recipe_columns[16],
            'saturated_fat_perc': recipe_columns[17],
            'carbohydrates_perc': recipe_columns[18],
            'allergies': recipe_columns[19],
            'diets': recipe_columns[20],
            'meal': recipe_columns[21]
        }
    else:
        result = None

    return result