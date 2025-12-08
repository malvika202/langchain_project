import json

# Load your JSON movie data 
with open("movies.json", "r") as f:
    MOVIES = json.load(f)

# Tool function: look up a movie by title
def lookup_movie(input: dict):
    title = input["title"].lower()

    for movie in MOVIES:
        if movie["title"].lower() == title:
            return movie  

    return {"error": "Movie not found"}
