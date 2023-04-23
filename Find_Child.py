from scipy.spatial.distance import euclidean



def child_finder(child_data):
    min_distance = float('inf')
    closest_child = None
    for child_id, child_features in child_data:
        distance = euclidean(predicted_child_features, child_features)
        if distance < min_distance :
            min_distance = distance
            closest_child = child_id

print(f"Closest child found in the database: {closest_child}")
