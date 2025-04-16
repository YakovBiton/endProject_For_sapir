import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

import "../Styles/FindChild.css";

function FindChild() {
  const [serverResponse, setServerResponse] = useState(null);

  const [motherImage, setMotherImage] = useState(null);
  const [fatherImage, setFatherImage] = useState(null);

  const [motherName, setMotherName] = useState("");
  const [fatherName, setFatherName] = useState("");

  const hairColorOptions = [
    "Black",
    "Brown",
    "Blonde",
    "Redhead",
    "Light Brown",
    "Other",
  ];
  const eyeColorOptions = ["Dark Brown", "Blue", "Green", "Brown", "Other"];
  const [selectedMotherHairColor, setSelectedMotherHairColor] = useState("");
  const [selectedMotherEyeColor, setSelectedMotherEyeColor] = useState("");
  const [selectedFatherHairColor, setSelectedFatherHairColor] = useState("");
  const [selectedFatherEyeColor, setSelectedFatherEyeColor] = useState("");

  const navigate = useNavigate();

  const submitImages = async () => {
    const formData = new FormData();
    formData.append("files", motherImage);
    formData.append("files", fatherImage);

    formData.append("motherHairColor", selectedMotherHairColor);
    formData.append("motherEyeColor", selectedMotherEyeColor);
    formData.append("fatherHairColor", selectedFatherHairColor);
    formData.append("fatherEyeColor", selectedFatherEyeColor);

    const response = await fetch("http://localhost:8000/uploadfiles/", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      alert("Failed to send images to server");
      return;
    }

    const result = await response.json();
    setServerResponse(result);
    // Now result contains the data returned from the server
    console.log(result);

    // Redirect to another page with the uploaded data
    navigate("/UserPage", { state: { data: result } });
  };

  return (
    <body className="findchild">
      <h2 className="heading">Welcome!</h2>
      <h1>Please add your photos</h1>
      <div className="image-upload-section">
        <div className="image-upload-box">
          <div className="input-container">
            <h1>Add Mother Image</h1>
            <label htmlFor="motherImageInput">
              {motherImage ? "Update File:" : "Choose File:"}
            </label>
            <input
              type="file"
              id="motherImageInput"
              onChange={(e) => setMotherImage(e.target.files[0])}
            />
            {motherImage && (
              <img
                src={URL.createObjectURL(motherImage)}
                alt="Mother"
                className="uploaded-image"
              />
            )}
            <label>Full Name:</label>
            <input
              type="text"
              value={motherName}
              onChange={(e) => setMotherName(e.target.value)}
            />
            <label>Hair Color:</label>
            <select
              value={selectedMotherHairColor}
              onChange={(e) => setSelectedMotherHairColor(e.target.value)}>
              <option value="">Select Hair Color</option>
              {hairColorOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
            <label>Eye Color:</label>
            <select
              value={selectedMotherEyeColor}
              onChange={(e) => setSelectedMotherEyeColor(e.target.value)}>
              <option value="">Select Eye Color</option>
              {eyeColorOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="image-upload-box">
          <div className="input-container">
            <h1>Add Father Image</h1>
            <label htmlFor="fatherImageInput">
              {fatherImage ? "Update File:" : "Choose File:"}
            </label>
            <input
              type="file"
              id="fatherImageInput"
              onChange={(e) => setFatherImage(e.target.files[0])}
            />
            {fatherImage && (
              <img
                src={URL.createObjectURL(fatherImage)}
                alt="Father"
                className="uploaded-image"
              />
            )}
            <label>Full Name:</label>
            <input
              type="text"
              value={fatherName}
              onChange={(e) => setFatherName(e.target.value)}
            />
            <label>Hair Color:</label>
            <select
              value={selectedFatherHairColor}
              onChange={(e) => setSelectedFatherHairColor(e.target.value)}>
              <option value="">Select Hair Color</option>
              {hairColorOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
            <label>Eye Color:</label>
            <select
              value={selectedFatherEyeColor}
              onChange={(e) => setSelectedFatherEyeColor(e.target.value)}>
              <option value="">Select Eye Color</option>
              {eyeColorOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>
      <div className="button-container">
        <button onClick={submitImages}>Submit</button>
      </div>
    </body>
  );
}

export default FindChild;
