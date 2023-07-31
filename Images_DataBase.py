import sqlite3
import json
import numpy as np
########################     ########################################
# create a database for the images with sqlite3
########################     ########################################
def add_To_DataBase(features):

    # Connect to the SQLite database (this will create a new file called "image_data.db")
    conn = sqlite3.connect("C:\\kobbi\\endProject\\TSKinFace_Data\\image_data.db")
    cursor = conn.cursor()

    # Create a table for storing image data
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS image_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_full_name TEXT NOT NULL,
        info JSON NOT NULL
    )
    """)
    conn.commit()
    for feature in features:
        landmarks_array = np.array(feature.landmarks)
        # Convert NumPy array to nested list
        landmarks = landmarks_array.tolist()
        # Convert nested list to list of (x, y) tuples
        landmarks_list = [(x, y) for [x, y] in landmarks[0]]
        # Sample data (replace this with your actual data)
        face_embeddings_array = np.array(feature.face_embeddings)
        resnet_feature = np.array(feature.feature_resnet)

        image_data = {
        "image_full_name": feature.image_name,
        "info": {
            "landmarks": landmarks_list,
            "face_embeddings": numpy_to_list(face_embeddings_array),
            "feature_resnet": numpy_to_list(resnet_feature),
            "ratio_features": numpy_to_list(feature.ratio_features),
            "angle_features": numpy_to_list(feature.angle_features),
            "color_features": numpy_to_list(feature.color_features)
            }
        }
        print("Image data to be stored:", image_data)
        

        # Check if the entry with the same image_full_name already exists
        cursor.execute("""
        SELECT * FROM image_data WHERE image_full_name = ?
        """, (image_data["image_full_name"],))

        data = cursor.fetchone()  # Fetch one record

        if data is not None:
            # The entry already exists, so we delete it first
            cursor.execute("""
            DELETE FROM image_data
            WHERE image_full_name = ?
            """, (image_data["image_full_name"],))

            # Then, we insert the new data
            cursor.execute("""
            INSERT INTO image_data (image_full_name, info)
            VALUES (?, ?)
            """, (image_data["image_full_name"], json.dumps(image_data["info"])))
        else:
            # The entry doesn't exist, so we insert
            cursor.execute("""
            INSERT INTO image_data (image_full_name, info)
            VALUES (?, ?)
            """, (image_data["image_full_name"], json.dumps(image_data["info"])))


        conn.commit()

def delete_from_database(image_full_name):
    # Connect to the SQLite database
    conn = sqlite3.connect("C:\\kobbi\\endProject\\TSKinFace_Data\\image_data.db")
    cursor = conn.cursor()

    try:
        # Check if the entry with the same image_full_name already exists
        cursor.execute("""
        SELECT * FROM image_data WHERE image_full_name = ?
        """, (image_full_name,))

        data = cursor.fetchone()  # Fetch one record

        if data is not None:
            # The entry exists, so we delete it
            cursor.execute("""
            DELETE FROM image_data
            WHERE image_full_name = ?
            """, (image_full_name,))
            
            conn.commit()
            return True  # Return True indicating that deletion was successful

        else:
            print(f"No entry with image_full_name {image_full_name} exists.")
            return False  # Return False indicating that deletion was unsuccessful

    except Exception as e:
        print(f"An error occurred: {e}")
        return False  # Return False indicating that deletion was unsuccessful

    finally:
        # Close the connection to the database
        conn.close()


def numpy_to_list(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data