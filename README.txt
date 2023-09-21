El archivo principal es culinary_chatbot.py. Para ejecutar el chatbot, simplemente se ingresa el siguiente comando en la terminal:
pyhton culinary_chatbot.py

La primera vez al ejecutar el chatbot (culinary_chatbot.py), se abrirá una ventana para que el usuario añada información sobre sí. 
Con esa información, se calculará las calorías y nutrientes recomendados.
Una vez que el usuario haya introducido los datos, podrá iniciar una conversación con el chatbot a través de la ventana principal 
de la interfaz “Culinary Chatbot”. Además, se ha incluido un botón "Update user info” que permite al usuario cambiar la información
proporcionada al chatbot. 

Para reiniciar el chatbot, es decir empezar de cero, se deben borrar las tablas de la base de datos utilizando el siguiente código:
culinary_chatbot_database.drop_table('recipes')
    culinary_chatbot_database.drop_table('allergies')
    culinary_chatbot_database.drop_table('ingredients')
    culinary_chatbot_database.drop_table('recipe_ingredient')
    culinary_chatbot_database.drop_table('nutrition_types_aliments')
    culinary_chatbot_database.drop_table('nutrition_ingredients_by_types')
    culinary_chatbot_database.drop_table('allergies_grouped_by_aliments')
    culinary_chatbot_database.drop_table('allergies_ingredients')
    culinary_chatbot_database.drop_table('diets')
    culinary_chatbot_database.drop_table('recipe_diets')
    culinary_chatbot_database.drop_table('user')
    culinary_chatbot_database.drop_table('conversation_history')
	
Además, se deben establecer las variables "is_first_time_user_info" y "is_first_time_running" en "true" en archivo “persistent_variables.json”.
En el archivo “persistent_variables.json”, también se incluye la variable "current_date", que se utiliza para determinar si es un nuevo día y 
así restablecer los valores nutricionales diarios (calories_eaten_day,carbohidrates_grams_day, fats_grams_day, protein_grams_day) del usuario.

Notas:

Solo se consideran los nutrientes de carbohidratos, proteínas y grasas, ya que solo conocemos los porcentajes recomendados por la OMS de estos 
nutrientes. Sin embargo, se informará al usuario de la necesidad de consumir 5 raciones diarias de verduras, hortalizas y frutas para conseguir 
obtener fibra, minerales y vitaminas necesarias. 

Actualmente, al recomendar una receta o planificar las comidas para un día completo, no se considera que el usuario rechace la receta. Es decir,
se considera que el usuario consumirá esa receta o seguirá ese plan de comidas.

Para las etiquetas “recipe_calories” y “recipe_nutrition_values”, se considerará que se refieren a la receta presente en la última respuesta del 
chatbot (ultima respuesta guardada en tabla conversation_history con la etiqueta “recipe”). 

Se pueden consultar y descargar los distintos datasets en los siguientes enlaces (Kaggle):
-recipes: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv

-recipesTerms: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv

-nutritionIngredients: https://www.kaggle.com/code/leogenzano/nutrientes-an-lisis-exploratorio-eda/input

-intolerances: https://www.kaggle.com/datasets/boltcutters/food-allergens-and-allergies
