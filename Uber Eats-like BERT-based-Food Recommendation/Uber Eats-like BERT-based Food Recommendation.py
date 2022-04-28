from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

## Enter a Uber Eats food category - e.g. healthy, comfort, alcohol, breakfast, etc.
food_category = "breakfast & brunch"

model = SentenceTransformer('bert-base-nli-mean-tokens')

doc_embedding = model.encode([food_category])
candidate_embeddings = model.encode(candidates)

## Food list from https://github.com/imsky/wordlists/blob/master/nouns/food.txt
candidates=[]
for x in open("/Users/enverdenizaskin/Downloads/foodlist.txt","r"):
    candidates.append(x.strip())
candidate_embeddings = model.encode(candidates)

## Use top_n to select the number of recommendations 
top_n = 5
distances = cosine_similarity(doc_embedding, candidate_embeddings)
recommendations = [candidates[index] for index in distances.argsort()[0][-top_n:]]
print(recommendations)
