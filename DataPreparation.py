import pandas as pd
import ast
import nltk
import matplotlib.pyplot as plt 
import seaborn as sns
import culinary_chatbot_database

from pattern.text.en import singularize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class DataPreparation:
    # Constants
    SPLIT_SEQ = ', '


    def __init__(self):
        # Attributes
        self.recipes = None
        self.recipe_ingredients = None
        self.intolerances = None
        self.nutrition_aliments = None 
        self.allergies_grouped_by_aliments = None
        self.tfidf_vectorizer_allergies = None
        self.clf_allergies = None
        self.tfidf_vectorizer_diets = None
        self.clf_diets = None
        self.multilabel_binarizer_allergies = None
        self.multilabel_binarizer_diets = None

        # Preprocessing of datasets 
        self._preprocessing_dataset_recipes()
        self._preprocessing_dataset_intolerances()
        self._preprocessing_dataset_nutrition_aliments()
        self._classificate_allergies_in_each_recipe()
        self._classificate_diets_in_each_recipe()
        self.classificate_type_of_meal_in_each_recipe()


    ''' 
    Public methods
    '''
    # Getter for recipes     
    def get_recipes(self):
        return self.recipes


    # Getter for intolerances   
    def get_intolerances(self):
        return self.intolerances 


    # Getter for nutrition_aliments
    def get_nutrition_aliments(self):
        return self.nutrition_aliments

    '''
    Preparation of datasets
    '''
    # Preparation of the data recipes
    def _preprocessing_dataset_recipes(self):
        # Load the dataset with the recipes 
        self.recipes = pd.read_csv('recipes.csv')
        recipes_terms = pd.read_csv('recipesTerms.csv')

        # Merge the datasets recipes and recipesTerms by matched ID number
        self.recipes = pd.merge(self.recipes, recipes_terms, on='id')
        # Rename columns
        self.recipes = self.recipes.rename(columns={'id':'recipe_id', 'name_x':'name', 'tags_x':'tags', 'steps_x':'steps', 'ingredients_x':'ingredients', 'ingredients_raw_str':'ingredients_cuantity'})
        
        # Split the nutritional values of the column nutrition
        self.recipes['nutrition'] = self.recipes['nutrition'].apply(ast.literal_eval)
        nutrition = self.recipes["nutrition"].apply(lambda n: pd.Series(n, index=['calories', 'total_fat_perc', 'sugar_perc', 'sodium_perc', 'protein_perc', 'saturated_fat_perc', 'carbohydrates_perc']))
        # Convert the type of the new columns to float
        nutrition = nutrition.astype(float)
        # Join the new columns to the dataset recipes 
        self.recipes = pd.concat([self.recipes, nutrition], axis=1)

        # Delete the unnecessary columns of the datasets
        self.recipes = self.recipes.drop(['contributor_id', 'submitted', 'description_x', 'name_y', 'description_y', 'ingredients_y', 'steps_y', 'tags_y', 'nutrition'],axis=1)

        # Remove duplicated recipes
        self.recipes.drop_duplicates(subset=['recipe_id'], inplace=True)

        # Convert to lowercase the values of the column 'ingredients'
        self.recipes['ingredients'] = self.recipes['ingredients'].str.lower()

        # Select the first 10000 rows from  the dataset recipes 
        self.recipes = self.recipes.head(10000)

        # Create a list to save the ingredients and the id of the recipes
        ingredients_and_recipe_id = []
        ingredients_dictionary = {}

        
        for index, row in self.recipes.iterrows():
            id_recipe = row['recipe_id']
            ingredients = eval(row['ingredients'])

            for ingredient in ingredients:
                ingredient = ingredient.strip()

                if ingredient in ingredients_dictionary:
                    ingredients_dictionary[ingredient].append(id_recipe)
                else:
                    ingredients_dictionary[ingredient] = [id_recipe]

        for ingredient, recipe_ids in ingredients_dictionary.items():
            ingredients_and_recipe_id.append({'ingredient': ingredient, 'recipe_id': recipe_ids})

        # Create a new dataframe to save all the ingredients of each recipe and the id of the recipe where it is used 
        self.recipe_ingredients = pd.DataFrame(ingredients_and_recipe_id) 


    # Preparation of the data intolerances
    def _preprocessing_dataset_intolerances(self):
        # Load the dataset with the recipes
        self.intolerances = pd.read_csv('intolerances.csv')
        self.intolerances = self.intolerances.drop(['Class', 'Type', 'Group'],axis=1)

        # Group the aliments by it's intolerance/allergy
        self.intolerances = self.intolerances.groupby('Allergy')['Food'].agg(self.SPLIT_SEQ.join).reset_index()

        # Black list with allergies and intolerances not relevant
        columns_to_remove_intolerances = ['Allium Allergy',  'Alpha-gal Syndrome', 'Aquagenic Urticaria', 'Banana Allergy', 'Beer Allergy', 
                    'Broccoli allergy', 'Citrus Allergy',  'Cruciferous Allergy', 'Histamine Allergy', 'Honey Allergy','Hypersensitivity', 
                    'Insulin Allergy', 'LTP Allergy', 'Legume Allergy','Mint Allergy', 'Mushroom Allergy', 'Nightshade Allergy', 
                    'Ochratoxin Allergy', 'Oral Allergy Syndrome', 'Pepper Allergy', 'Potato Allergy', 'Poultry Allergy', 'Poultry Allergy',
                    'Ragweed Allergy', 'Rice Allergy', 'Salicylate Allergy', 'Stone Fruit Allergy', 'Lactose Intolerance', 'Sugar Allergy / Intolerance',
                    'Tannin Allergy', 'Thyroid']
        # Drop the allergies not relevant of the column Allergy
        self.intolerances = self.intolerances[self.intolerances.Allergy.isin(columns_to_remove_intolerances) == False]

        # Remove duplicated aliments in the intolerances dataset  
        self.intolerances.drop_duplicates(subset=['Food'], inplace=True)

        # Convert to lowercase the values of the column 'Food' 
        self.intolerances['Food'] = self.intolerances['Food'].str.lower()

        # Add the egg allergy
        self.intolerances.loc[len(self.intolerances.index)] = ['Egg Allergy',  'egg']

        # Update the value from the row 'Milk allergy / Lactose intolerance' to 'Milk Allergy/Intolerance'
        self.intolerances.loc[18, 'Allergy'] = 'Milk Allergy/Intolerance'

        # Update the value of the row to the soy allergy 
        self.intolerances.loc[34, 'Food'] = 'soy'
        
        # Group the aliments by it's intolerance/allergy
        self.allergies_grouped_by_aliments = self.intolerances.groupby('Allergy')['Food'].agg(self.SPLIT_SEQ.join).reset_index()

        self.allergies_grouped_by_aliments = self.allergies_grouped_by_aliments.rename(columns={'Allergy':'allergy_name', 'Food':'aliments'})

    # Preparation of the data nutrition aliments 
    def _preprocessing_dataset_nutrition_aliments(self):
        # Load the dataset with the recipes
        self.nutrition_aliments = pd.read_csv('nutritionIngredients.csv')

        # Rename columns
        self.nutrition_aliments = self.nutrition_aliments.rename(columns={'Shrt_Desc':'aliment_type_name', 'Energ_Kcal':'energ_Kcal', 'Protein_(g)':'protein_g', 'Carbohydrt_(g)':'carbohydrates_g',
                                                                          'FA_Sat_(g)':'saturated_fat_g'})
                            
        # Black list with nutrition values not relevant
        columns_to_remove_nutrition_values =[0, 1, 3, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
                                             40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52,53]
        
        # Drop the nutrition values not relevant
        self.nutrition_aliments = self.nutrition_aliments.drop(self.nutrition_aliments.columns[columns_to_remove_nutrition_values], axis=1)
        

    '''
    Classification of the allergies of each recipe
    '''
    # Get the allergies present in each recipe by going through the ingredients of each recipe and comparing them with the ingredients associated with each allergy
    def _get_allergies(self, recipe):
        allergies = []
        # Get recipe ingredients
        ingredients = eval(recipe['ingredients'])
        for ingredient in ingredients:
            # Ingredients from intolerances are always singular
            ingredient = singularize(ingredient)
            for index, intolerance in self.intolerances.iterrows():
                # Convert the string to a list of ingredients
                into_ingredients = intolerance['Food'].split(self.SPLIT_SEQ)
                for into_ingre in into_ingredients:
                    if into_ingre in ingredient and intolerance['Allergy'] not in allergies:
                        allergies.append(intolerance['Allergy'])
        return allergies


    # Classificate the allergies in each recipe
    def _classificate_allergies_in_each_recipe(self):
        # Create an empty list to save the allergies of each recipe
        allergies_list = []

        for index, recipe in self.recipes.iterrows():
            # Get the allergies of a recipe
            allergies = self._get_allergies(recipe)
            # Add the allergies of a recipe at allergies_list
            allergies_list.append(allergies)

        # Add the allergies list as a new column to recipes 
        self.recipes['allergies'] = allergies_list


    '''
    Classification of the types of diets in each recipe
    '''
    # Get the diets from a recipe by looking at the tags of the column 'tags'. We will only consider the 'vegetarian' and 'vegan' diet
    def _get_diets(self, recipe):
        diets = []
        # Get recipe tags
        tags = eval(recipe['tags']) # eval to convert string (read from csv) to list
        for tag in tags:
            if tag == 'vegetarian':
                diets.append('vegetarian')
            
            if tag == 'vegan':
                diets.append('vegan')
        
        return diets


    # Classificate the diets in each recipe
    def _classificate_diets_in_each_recipe(self):
        # Create an empty list to save the diets of each recipe
        diets_list = []

        for index, recipe in self.recipes.iterrows():
            # Get the diets of a recipe by looking at the tags of the column 'tags'
            diets = self._get_diets(recipe)
            # Add the diets of a recipe at diets_list
            diets_list.append(diets)

        # Add the diets list as a new column to recipes 
        self.recipes['diets'] = diets_list


    '''
    Classification of the meals of each recipe
    '''
    # Get the allergies present in each recipe by going through the ingredients of each recipe and comparing them with the ingredients associated with each allergy
    def classificate_type_of_meal_in_each_recipe(self):
         # Create an empty list to save the meal types of each recipe
        meals_list = []

        for index, recipe in self.recipes.iterrows():
            # Get the search terms of the recipe
            search_terms = recipe['search_terms']
            
            # Check if the search terms contain meal types
            if 'breakfast' in search_terms:
                meals_list.append('breakfast')
            elif 'lunch' in search_terms:
                meals_list.append('lunch')
            elif 'dinner' in search_terms:
                meals_list.append('dinner')
            elif 'dessert' in search_terms:
                meals_list.append('dessert')
            else:
                meals_list.append('other')  # another type of meal
            
        # Add the meals list as a new column to recipes
        self.recipes['meal'] = meals_list


    '''
    Multi-label classification to classificate the allergies present in the recipes  
    '''
    # Model multi-label classification to predict the allergies 
    def _model_multi_label_classification_allergies(self):
        # Extract all types of allergies that are in the dataset "recipes", in the column "allergies". 
        # Get all allergies tags in a list
        all_allergies = []
        for allergies in self.recipes['allergies']:
            for allergy in allergies:
                all_allergies.append(allergy)

        all_allergies = nltk.FreqDist(all_allergies)

        # create dataframe
        all_allergies_dataframe = pd.DataFrame({'Allergy': list(all_allergies.keys()), 
                                    'Count': list(all_allergies.values())})

        g = all_allergies_dataframe.nlargest(columns="Count", n = 50) 
        plt.figure(figsize=(9,9)) 
        ax = sns.barplot(data=g, x= 'Count', y = 'Allergy') 
        ax.set(ylabel = 'Allergies') 
        plt.show()
        
        
        self.multilabel_binarizer_allergies = MultiLabelBinarizer()
        self.multilabel_binarizer_allergies.fit(self.recipes['allergies'])

        # Transform target variable
        y_allergies = self.multilabel_binarizer_allergies.transform(self.recipes['allergies'])      

        # Split data into trianing and validation test 
        self.tfidf_vectorizer_allergies = TfidfVectorizer(max_df=0.8, max_features=8000)
        xtrain_allergies, xval_allergies, ytrain_allergies, yval_allergies = train_test_split(self.recipes['ingredients'], y_allergies, test_size=0.2, random_state=9)

        # Create TF-IDF features
        xtrain_tfidf_allergies = self.tfidf_vectorizer_allergies.fit_transform(xtrain_allergies)
        xval_tfidf_allergies = self.tfidf_vectorizer_allergies.transform(xval_allergies)

        # Prediction model
        lr_allergies = LogisticRegression(solver='lbfgs', max_iter=1000)
        self.clf_allergies = OneVsRestClassifier(lr_allergies)

        # Fit model on train data
        self.clf_allergies.fit(xtrain_tfidf_allergies, ytrain_allergies)

        # Make the predictions for the validation set
        y_pred_allergies = self.clf_allergies.predict(xval_tfidf_allergies)

        # Evaluate performance
        f1_score(yval_allergies, y_pred_allergies, average="micro")
        # Calculate the accuracy 
        accuracy_allergies = accuracy_score(yval_allergies, y_pred_allergies)
        print('The accuracy of the classification model for the allergies is ', accuracy_allergies)


    # Create inference function to classificate the allergies in new recipes
    def _infer_tags_allergies(self, q):
        vec = self.tfidf_vectorizer_allergies.transform([q])
        pred = self.clf_allergies.predict(vec)
        return self.multilabel_binarizer_allergies.inverse_transform(pred)


    '''
    Multi-label classification to classificate the kind of diet of the recipes 
    '''
    # Model multi-label classification to predict the diets 
    def _model_multi_label_classification_diets(self):
        # Extract all types of diets that are in the dataset "recipes", in the column "diets". 
        # Get all diets tags in a list
        all_diets = []
        for diets in self.recipes['diets']:
            for diet in diets:
                all_diets.append(diet)

        all_diets = nltk.FreqDist(all_diets)

        # create dataframe
        all_diets_dataframe = pd.DataFrame({'Diet': list(all_diets.keys()), 
                                    'Count': list(all_diets.values())})

        g = all_diets_dataframe.nlargest(columns="Count", n = 50) 
        plt.figure(figsize=(12,15)) 
        ax = sns.barplot(data=g, x= "Count", y = "Diet") 
        ax.set(ylabel = 'Diets') 
        plt.show()

        # Converting text to features
        self.multilabel_binarizer_diets = MultiLabelBinarizer()
        self.multilabel_binarizer_diets.fit(self.recipes['diets'])

        # Transform target variable
        y_diets = self.multilabel_binarizer_diets.transform(self.recipes['diets'])      

        # Split data into trianing and validation test 
        self.tfidf_vectorizer_diets = TfidfVectorizer(max_df=0.8, max_features=10000)
        xtrain_diets, xval_diets, ytrain_diets, yval_diets = train_test_split(self.recipes['ingredients'], y_diets, test_size=0.2, random_state=9)

        # Create TF-IDF features
        xtrain_tfidf_diets = self.tfidf_vectorizer_diets.fit_transform(xtrain_diets)
        xval_tfidf_diets = self.tfidf_vectorizer_diets.transform(xval_diets)

        # Prediction model
        lr_diets = LogisticRegression(solver='lbfgs', max_iter=1000)
        self.clf_diets = OneVsRestClassifier(lr_diets)

        # Fit model on train data
        self.clf_diets.fit(xtrain_tfidf_diets, ytrain_diets)

        # Make the predictions for the validation set
        y_pred_diets = self.clf_diets.predict(xval_tfidf_diets)

        # Evaluate the performance of the model 
        f1_score(yval_diets, y_pred_diets, average="micro")
        # Calculate the accuracy 
        accuracy_diets = accuracy_score(yval_diets, y_pred_diets)
        print('The accuracy of the classification model for the diets is ', accuracy_diets)


    # Create inference function to classificate the diets in new recipes
    def _infer_tags_diets(self, q):
        vec = self.tfidf_vectorizer_diets.transform([q])
        pred = self.clf_diets.predict(vec)
        return self.multilabel_binarizer_diets.inverse_transform(pred)


    # Function to add a new recipe to the table recipes
    def _add_new_recipe(self, recipe):
        recipe_list = [recipe.name, recipe.minutes, recipe.tags, recipe.n_steps, recipe.steps, recipe.ingredients, recipe.n_ingredients, recipe.ingredients_cuantity, recipe.serving_size,
                    recipe.servings, recipe.search_terms, recipe.calories, recipe.total_fat, recipe.sugar, recipe.sodium, recipe.protein, recipe.saturated_fat, recipe.carbohydrates, recipe.allergies, recipe.diets]
        
        # Call infer_tags_allergies to predict the allergies based on ingredients
        predicted_allergies = self._infer_tags_allergies(recipe.ingredients)
        
        # Update the recipe.allergies with the predicted allergies
        recipe.allergies.extend(predicted_allergies)

        # Call infer_tags_diets to predict the diets based on ingredients
        predicted_diets = self._infer_tags_diets(recipe.ingredients)

        # Update the recipe.diets with the predicted diets
        recipe.diets.extend(predicted_diets)

        culinary_chatbot_database.insert_row('recipes', recipe_list)