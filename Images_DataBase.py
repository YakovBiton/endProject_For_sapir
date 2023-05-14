import sqlite3
import json
import numpy as np

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
        # Sample data (replace this with your actual data)
        image_data = {
        "image_full_name": feature.image_name,
        "info": {
            #"landmarks": numpy_to_list(feature.landmarks),
            #"face_embeddings": numpy_to_list(feature.face_embeddings),
            "ratio_features": numpy_to_list(feature.ratio_features),
            "angle_features": numpy_to_list(feature.angle_features),
            "color_features": numpy_to_list(feature.color_features)
            }
        }
        print("Image data to be stored:", image_data)


        # Insert the data into the table
        cursor.execute("""
        INSERT INTO image_data (image_full_name, info)
        VALUES (?, ?)
        """, (image_data["image_full_name"], json.dumps(image_data["info"])))

        conn.commit()

def numpy_to_list(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data