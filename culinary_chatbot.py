import random
import json
import os
import pickle
import numpy as np
import nltk
import culinary_chatbot_database
import nutrition_functions
import time

from tkinter import *
from DataPreparation import DataPreparation
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from datetime import datetime

# Constants
SPLIT_SEQ = ', '
FILE_PATH = "persistent_variables.json"
DATE_FORMAT = "%Y-%m-%d"
MAX_WAITING_TIME_S = 420

# Global variables
lemmatizer = WordNetLemmatizer()
current_date = datetime.now().date()
used_ingredients = []


# Save the persistent variables when updated in the file "persistent_variables"
def save_persistent_data(data):
    with open(FILE_PATH, "w") as file:
        json.dump(data, file)


# Load the file "persistent_variables"
def load_persistent_data():
    if os.path.isfile(FILE_PATH):
        with open(FILE_PATH, "r") as file:
            data = json.load(file)
            return data
    else:
        return {}  
    

# Preprocessing the input 
def clean_up_sentences(sentence):
    # Tokenize the sentence of the user
    sentence_words = nltk.word_tokenize(sentence)

    # Get the lema of each token/word
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]

    return sentence_words


# Return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bag_of_words(sentence):
     # Preprocess the input
    sentence_words = clean_up_sentences(sentence)

    # Bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


# Predict the label/class
def predict_class(sentence):
    # Filter out predictions below a threshold
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort by stength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list
  

# Get the response to answer the user
def get_response(intents_list, intents_json, question, date):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    response = ""
    for i in list_of_intents:
        if i['tag'] == tag:
            if tag == 'greeting':
                response = random.choice(i['responses'])
            elif tag == 'goodbye':
                response = random.choice(i['responses'])
            elif tag == 'thanks':
                response = random.choice(i['responses'])
            elif tag == 'noanswer':
                response = random.choice(i['responses'])
            elif tag == 'options':
                response = random.choice(i['responses'])
            elif tag == 'recipe':
                found = False
                user_info = culinary_chatbot_database.get_user_info()
                # Get the previous answers where the recipe tag was "recipe" and date is the same as the one passed as parameter
                previous_recipes_answers = culinary_chatbot_database.get_previous_recipe_tag_answers(date)
                used_ingredients = []
                additional_info = ("To promote a balanced diet, it is important to consider the nutrients carbohydrates, fats, and protein. "
                    "However, it is recommended to consume 5 servings per day of vegetables, fruits, and vegetables to obtain vitamins, minerals, and fiber.")
                

                if previous_recipes_answers:
                    for previous_recipe_answer in previous_recipes_answers:
                        # Get the recipe name from the answer
                        recipe_name = culinary_chatbot_database.get_recipe_name_from_answer(previous_recipe_answer)
                        if recipe_name is not None:
                            recipe_id = culinary_chatbot_database.get_recipe_id_by_name(recipe_name)
                            # Get the ingredients ids from the recipe passed before to be compared with the new recipe that the chatbot wants to pass to the user, 
                            # to don't repit the same ingredients 
                            recipe_ingredients = culinary_chatbot_database.get_used_ingredients_from_recipe(recipe_id)
                            used_ingredients.extend(recipe_ingredients)

                    # Add a timer
                    time_start = time.time() 
                    while not found:
                        recipe, recipe_name = culinary_chatbot_database.get_random_recipe()
                        recipe_id = culinary_chatbot_database.get_recipe_id_by_name(recipe_name)
                        recipe_to_approve_ingredients = culinary_chatbot_database.get_used_ingredients_from_recipe(recipe_id)
                        nutrition_values_recipe = culinary_chatbot_database.get_recipe_nutritional_values(recipe_name)
                        # Transform the daily value of the nutrients on the recipe based on 2000 calories recommended and adapt to the user calories recommended and transform to grams 
                        carbohydrates_portion = nutrition_functions.transform_daily_value_in_grams_adjusted_to_user_carbohydrates(nutrition_values_recipe['carbohydrates_perc'])
                        fats_portion = nutrition_functions.transform_daily_value_in_grams_adjusted_to_user_fats(nutrition_values_recipe['fats_perc'])
                        protein_portion = nutrition_functions.transform_daily_value_in_grams_adjusted_to_user_protein(nutrition_values_recipe['protein_perc'])
                        # Get the calories from the recipe
                        calories = nutrition_values_recipe['calories']
                        # Check that the recipe does not exceed the recommended limits and does not contain any used ingredients used before in the recipes of previous answers
                        if (carbohydrates_portion + user_info['carbohidrates_grams_day'] <= user_info['carbohidrates_grams_recommended'] and
                            fats_portion + user_info['fats_grams_day'] <= user_info['fats_grams_recommended'] and
                            protein_portion + user_info['protein_grams_day'] <= user_info['protein_grams_recommended'] and 
                            calories + user_info['calories_eaten_day'] <= user_info['calories_recommended_day']):
                            recipe_ingredients_set = set(recipe_to_approve_ingredients)
                            used_ingredients_set = set(used_ingredients)
                            if recipe_ingredients_set.isdisjoint(used_ingredients_set):
                                found = True
                        curr_time = time.time()
                        if curr_time - time_start > MAX_WAITING_TIME_S:
                            break
                    if found == True:
                        # Update the nutrition values of the user, taking into account that the user is always going to prepare the recipe that the user recommends
                        carbohydrates_grams = user_info['carbohidrates_grams_day'] + carbohydrates_portion
                        fats_grams = user_info['fats_grams_day'] + fats_portion
                        protein_grams = user_info['protein_grams_day'] + protein_portion
                        calories_eaten = user_info['calories_eaten_day'] + calories
                        culinary_chatbot_database.update_nutrition_grams_day(carbohydrates_grams, fats_grams, protein_grams)
                        culinary_chatbot_database.update_calories_eaten(calories_eaten)
                        response = random.choice(i['responses']).format(recipe=recipe + "\n\n" + additional_info)
                    else:
                        response = "No recipe was found"
                else:
                    # Is the first time of the day that the user asks for a recipe
                    while not found:
                        recipe, recipe_name = culinary_chatbot_database.get_random_recipe()
                        nutrition_values_recipe = culinary_chatbot_database.get_recipe_nutritional_values(recipe_name)
                         # Transform the daily value of the nutrients on the recipe based on 2000 calories recommended and adapt to the user calories recommended and transform to grams 
                        carbohydrates_portion = nutrition_functions.transform_daily_value_in_grams_adjusted_to_user_carbohydrates(nutrition_values_recipe['carbohydrates_perc'])
                        fats_portion = nutrition_functions.transform_daily_value_in_grams_adjusted_to_user_fats(nutrition_values_recipe['fats_perc'])
                        protein_portion = nutrition_functions.transform_daily_value_in_grams_adjusted_to_user_protein(nutrition_values_recipe['protein_perc'])
                        calories = nutrition_values_recipe['calories']
                        # Check if the nutritional values of the recipe don't excede the recoomended for the user
                        if (carbohydrates_portion <= user_info['carbohidrates_grams_recommended'] and
                            fats_portion <= user_info['fats_grams_recommended'] and
                            protein_portion <= user_info['protein_grams_recommended'] and 
                            calories + user_info['calories_eaten_day'] <= user_info['calories_recommended_day']):
                            found = True
                    if found == True:
                        # Update the nutrition values that the user is going to get by eating this recipe
                        carbohydrates_grams = carbohydrates_portion
                        fats_grams =  fats_portion
                        protein_grams =  protein_portion
                        calories_eaten = calories
                        culinary_chatbot_database.update_nutrition_grams_day(carbohydrates_grams, fats_grams, protein_grams)
                        culinary_chatbot_database.update_calories_eaten(calories_eaten)
                        response = random.choice(i['responses']).format(recipe=recipe + "\n\n" + additional_info)
            elif tag == 'recipe_ingredient':
                # Get the tagged words from the question of the user
                tagged_words = pos_tag(clean_up_sentences(question))
                tagged_sentence = [(word, tag) for word, tag in tagged_words]
                # Get only the NN tagged words and exclude 'recipe' and 'aliment' 
                filtered_words = [word for word, tag in tagged_sentence if tag.startswith('NN') and word.lower() not in ['recipe', 'aliment']]

                if filtered_words:
                    # Get the recipe with the ingredient passed as parameter
                    recipe = culinary_chatbot_database.get_recipe_by_ingredients(filtered_words)
                    if recipe is not None:
                        recipe_return = f"recipe: {recipe['recipe_name']}\n\nIngredients: {recipe['ingredients_cuantity']}\n\nSteps: {recipe['steps']}"
                        response = random.choice(i['responses']).format(recipe=recipe_return)
                    else:
                        response = "My apologies. I couldn't find a recipe with those ingredients."
                else:
                    response = "I'm sorry, I couldn't understand the ingredients that you mentioned."
            elif tag == 'recipe_calories':
                previous_answer = culinary_chatbot_database.get_previous_recipe_tag_answers(date)
                recipe_name = culinary_chatbot_database.get_recipe_name_from_answer(previous_answer)
                if recipe_name:
                    # Get the nutrition values of the recipe passing her name
                    nutritional_values = culinary_chatbot_database.get_recipe_nutritional_values(recipe_name)
                    if nutritional_values:
                        response = random.choice(i['responses']).format(recipe=recipe_name, calories=nutritional_values['calories'])
                    else:
                        response = f'My apologies. I do not have any knowledge about the nutritional values of the recipe ({recipe_name}).'
            elif tag == 'recipe_nutrition_values':
                # it will be considered that it refers to the recipe present in the last answer of the chatbot (last answer saved in conversation_history table)
                # of the chatbot (last answer saved in conversation_history table). In addition, it will be considered that this last response that the chatbot has
                # generated in response to a user's question have the tag "recipe".
                previous_answer = culinary_chatbot_database.get_previous_recipe_tag_answers(date)
                recipe_name = culinary_chatbot_database.get_recipe_name_from_answer(previous_answer)
                if recipe_name:
                    nutritional_values = culinary_chatbot_database.get_recipe_nutritional_values(recipe_name)
                    if nutritional_values:
                        response = random.choice(i['responses']).format(recipe=recipe_name, nutritional_values=nutritional_values)
                    else:
                        response = f'My apologies. I do not have any knowledge about the nutritional values of the recipe ({recipe}).'.format(recipe_name)
            elif tag == 'aliment_nutrition_values':
                # Get the tagged words from the question of the user
                tagged_words = pos_tag(clean_up_sentences(question))
                tagged_sentence =  [(word, tag) for word, tag in tagged_words]
                # Get only the NN tagged words and exclude 'nutrition', 'values', 'value' and 'information'
                filtered_words =  [word for word, tag in tagged_sentence if tag.startswith('NN') and word.lower() not in ['nutrition', 'values', 'value', 'information']]
                if filtered_words:
                    # Get the nutritional values for the aliment in filtered_words. For example, if the aliment is 'egg', the chatbot is going to give the nutrition values of cooked egg, fried egg,...
                    aliment, nutritional_values = culinary_chatbot_database.get_nutritional_values_by_aliment(filtered_words)
                    if nutritional_values == 'Not found':
                        response = f'My apologies. I do not have any knowledge about the nutritional values of the aliment ({aliment}).'.format(aliment)
                    else:
                        response = random.choice(i['responses']).format(aliment = aliment, nutritional_values = nutritional_values)          
            elif tag == 'plan_meal_day':
                recipes = culinary_chatbot_database.get_recipes()
                user_info = culinary_chatbot_database.get_user_info()
                # Define the different types of meals for the day
                meal_types_per_day = ['breakfast', 'breakfast', 'lunch', 'dessert', 'breakfast', 'dinner']

                meal_plan = []
                used_ingredients = []
                total_carbohydrates = 0
                total_fats = 0
                total_protein = 0
                total_calories = 0

                for meal_type in meal_types_per_day:
                    valid_recipe = False

                    while not valid_recipe:
                        recipe = random.choice(recipes)

                        if recipe[21] == meal_type:
                            carbohydrates_perc = recipe[18]
                            fats_perc = recipe[13]
                            protein_perc = recipe[16]
                            calories = recipe[12]
                            # Transform the daily value of the nutrients on the recipe based on 2000 calories recommended and adapt to the user calories recommended and transform to grams 
                            carbohydrates_portion = nutrition_functions.transform_daily_value_in_grams_adjusted_to_user_carbohydrates(carbohydrates_perc)
                            fats_portion = nutrition_functions.transform_daily_value_in_grams_adjusted_to_user_fats(fats_perc)
                            protein_portion = nutrition_functions.transform_daily_value_in_grams_adjusted_to_user_protein(protein_perc)
                            
                            recipe_id = recipe[0]
                            # Get the ingredients of the recipe
                            recipe_ingredients = culinary_chatbot_database.get_used_ingredients_from_recipe(recipe_id)

                            # Check if the recipe exceeds the recommended limits and does not contain used ingredients
                            if (total_carbohydrates + carbohydrates_portion <= user_info['carbohidrates_grams_recommended'] and
                                    total_fats + fats_portion <= user_info['fats_grams_recommended'] and
                                    total_protein + protein_portion <= user_info['protein_grams_recommended'] and 
                                    calories <= user_info['calories_recommended_day']):
                                recipe_ingredients_set = set(recipe_ingredients)
                                used_ingredients_set = set(used_ingredients)
                                if recipe_ingredients_set.isdisjoint(used_ingredients_set):
                                    valid_recipe = True

                    meal_plan.append(f"{meal_type}: {recipe[1]}\n\nIngredients:\n{recipe[8]}\n\nSteps:\n{recipe[5]}")
                    used_ingredients.extend(recipe_ingredients)

                    # Update the total quantities of proteins, carbohydrates, and fats
                    total_protein += protein_portion
                    total_carbohydrates += carbohydrates_portion
                    total_fats += fats_portion
                    total_calories += calories


                additional_info = ("To promote a balanced diet, it is important to consider the nutrients carbohydrates, fats, and protein. "
                                "However, it is recommended to consume 5 servings per day of vegetables, fruits, and vegetables to obtain vitamins, minerals, and fiber.")

                response = random.choice(i['responses']).format(meal_plan="\n\n".join(meal_plan) + "\n\n" + additional_info)
            elif tag == 'user_calories_left':
                calories_to_ingest = nutrition_functions.get_remaining_calories_ingest()
                response = random.choice(i['responses']).format(calories_to_ingest=calories_to_ingest)
            elif tag == 'user_calories_consumed':
                user_info = culinary_chatbot_database.get_user_info()
                response = random.choice(i['responses']).format(calories_ingested=user_info['calories_eaten_day'])
            elif tag == 'user_calories':
                user_info = culinary_chatbot_database.get_user_info()
                response = random.choice(i['responses']).format(calories=user_info['calories_recommended_day'])
            else:
                response = "I'm sorry I couldn't understand you. Can you please, say me in another way."
                
    return response, tag

# when the 
def send_message():
    global current_date 
    global used_ingredients

    # When is the first time that the user interacts with the chatbot, 
    # is necesary that the user inserts is information
    if persistent_data['is_first_time_user_info']:
        open_update_info_window()
    else:
        # Update date
        current_date = datetime.now().date()
        question = entry.get()
        entry.delete(0, END)

        if current_date > datetime.strptime(persistent_data['current_date'], DATE_FORMAT).date():
            # New date, restart the nutritional information of the user
            culinary_chatbot_database.restart_user_nutrition_values()
            used_ingredients = []
            persistent_data['current_date'] = current_date.strftime(DATE_FORMAT)
            save_persistent_data(persistent_data)

        ints = predict_class(question)
        answer, tag = get_response(ints, intents, question, current_date)
        culinary_chatbot_database.save_conversation_in_db(current_date, question, answer, tag)
        chat_window.insert(END, "User: " + question + "\n")
        chat_window.insert(END, "CulinaryBot: " + answer + "\n")
        chat_window.insert(END, "\n")
        chat_window.see(END)


# Create the window for the user introduce his data
def open_update_info_window():
    update_window = Toplevel(window)
    update_window.title("Update User Info")

    # Create labels and entry fields for user info
    name_label = Label(update_window, text="Name:")
    name_entry = Entry(update_window)
    name_label.pack()
    name_entry.pack()

    weight_label = Label(update_window, text="Weight (kg):")
    weight_entry = Entry(update_window)
    weight_label.pack()
    weight_entry.pack()

    height_label = Label(update_window, text="Height (cm):")
    height_entry = Entry(update_window)
    height_label.pack()
    height_entry.pack()

    age_label = Label(update_window, text="Age:")
    age_entry = Entry(update_window)
    age_label.pack()
    age_entry.pack()

    gender_label = Label(update_window, text="Gender:")
    gender_var = StringVar(update_window)
    gender_var.set("F")
    gender_optionmenu = OptionMenu(update_window, gender_var, "F", "M")
    gender_label.pack()
    gender_optionmenu.pack()

    activity_label = Label(update_window, text="Activity Factor:")
    activity_var = StringVar(update_window)
    activity_var.set("1.2")  
    activity_optionmenu = OptionMenu(update_window, activity_var, "1.2", "1.375", "1.55", "1.725", "1.9")
    activity_label.pack()
    activity_optionmenu.pack()

    allergies_list = culinary_chatbot_database.get_all_allergies()
    selected_allergies = []

    allergies_label = Label(update_window, text="Allergies:")
    allergies_label.pack()

    for allergy in allergies_list:
        allergy_var = IntVar()
        allergy_checkbox = Checkbutton(update_window, text=allergy, variable=allergy_var)
        allergy_checkbox.pack()
        selected_allergies.append((allergy, allergy_var))

    diets_list = culinary_chatbot_database.get_all_diets()
    selected_diets = []

    diets_label = Label(update_window, text="Diets:")
    diets_label.pack()

    for diet in diets_list:
        diet_var = IntVar()
        diet_checkbox = Checkbutton(update_window, text=diet, variable=diet_var)
        diet_checkbox.pack()
        selected_diets.append((diet, diet_var))

    # Create submit button
    submit_button = Button(update_window, text="Submit", command=lambda: update_info(update_window, name_entry.get(), weight_entry.get(), height_entry.get(), age_entry.get(), gender_var.get(), activity_var.get(), selected_allergies, selected_diets))
    submit_button.pack()

    # Show the window to insert the information of the user in foreground
    update_window.attributes('-topmost', True)  


# Update the user information in the table user
def update_info(update_window, user_name, weight, height, age, gender, activity_factor, allergies, diets):
    # Perform necessary operations to update user info with the provided values
    weight_kg = float(weight)
    height_cm = float(height)
    age = int(age)
    gender = gender
    activity_factor = float(activity_factor)

    # Obtener los valores seleccionados de las casillas de verificación
    selected_allergies = [allergy[0] for allergy in allergies if allergy[1].get() == 1]
    allergies_str = SPLIT_SEQ.join(selected_allergies)

    selected_diets = [diet[0] for diet in diets if diet[1].get() == 1]
    diets_str = SPLIT_SEQ.join(selected_diets)
    
    if persistent_data['is_first_time_user_info']:
        # Save the information about the user in the table "user"
        culinary_chatbot_database.add_user_info(user_name, weight_kg, height_cm, age, gender, activity_factor, allergies_str, diets_str)

        # Calculate the recommended calories, carbohydrates, fats and proteins to the user with info passed before and update 
        user_calories_recommended = nutrition_functions.calculate_recommended_calories_diary()
        culinary_chatbot_database.update_user_calories_recommended(user_calories_recommended)
        carbohydrates_grams = nutrition_functions.calculate_grams_carbohydrates(user_calories_recommended)
        fats_grams = nutrition_functions.calculate_grams_fats(user_calories_recommended)
        protein_grams = nutrition_functions.calculate_grams_protein(user_calories_recommended)
        culinary_chatbot_database.update_user_grams_recommended(carbohydrates_grams, fats_grams, protein_grams)
        persistent_data['is_first_time_user_info'] = False
        save_persistent_data(persistent_data)
    else:   
        # Save the updated information to the database or any other necessary operations
        culinary_chatbot_database.update_user_info(weight_kg, height_cm, age, gender, activity_factor, allergies_str, diets_str)

        # Calculate the recommended calories, carbohydrates, fats and proteins to the user with info passed before and update 
        user_calories_recommended = nutrition_functions.calculate_recommended_calories_diary()
        culinary_chatbot_database.update_user_calories_recommended(user_calories_recommended)
        carbohydrates_grams = nutrition_functions.calculate_grams_carbohydrates(user_calories_recommended)
        fats_grams = nutrition_functions.calculate_grams_fats(user_calories_recommended)
        protein_grams = nutrition_functions.calculate_grams_protein(user_calories_recommended)
        culinary_chatbot_database.update_user_grams_recommended(carbohydrates_grams, fats_grams, protein_grams)

    # Close the update window after saving the information
    update_window.destroy()


'''
Setup and start Chatbot   
'''

persistent_data = load_persistent_data()

if persistent_data['is_first_time_running']:
    # Preparation of datasets 
    dp = DataPreparation()
    # Create the tables and add data to them 
    culinary_chatbot_database.create_tables()
    culinary_chatbot_database.add_data_to_table_recipes(dp.recipes)
    culinary_chatbot_database.add_data_to_table_ingredients(dp.recipe_ingredients)
    culinary_chatbot_database.add_data_to_table_recipe_ingredient(dp.recipe_ingredients)
    culinary_chatbot_database.add_data_to_table_nutrition_types_aliments(dp.nutrition_aliments)
    culinary_chatbot_database.add_data_to_table_nutrition_ingredients_by_types()
    culinary_chatbot_database.add_data_to_table_allergies_grouped_by_aliments(dp.allergies_grouped_by_aliments)
    culinary_chatbot_database.add_data_to_table_allergies_ingredients()
    culinary_chatbot_database.add_data_to_table_diets()
    culinary_chatbot_database.add_data_to_table_recipe_diets()
	
    persistent_data['is_first_time_running'] = False
    save_persistent_data(persistent_data)


# Load files 
intents = json.loads(open("intents.json").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')


# Setup inface chatbot
# Create main window
window = Tk()
window.title("Culinary Chatbot")

# Create chat window
chat_window = Text(window, height=40, width=120)
chat_window.pack()

# Create the user's input field
entry = Entry(window, width=100)
entry.pack()

# Create button to send message
send_button = Button(window, text="Send", command=send_message)
send_button.pack()

# Create button to update the user information
update_info_button = Button(window, text="Update user info", command=open_update_info_window)
update_info_button.pack()

# Start main window loop
window.mainloop()
 