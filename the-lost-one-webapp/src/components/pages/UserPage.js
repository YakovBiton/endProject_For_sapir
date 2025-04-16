// UserPage.js

import React from "react";
import { useLocation } from "react-router-dom";
import "../Styles/UserPage.css";

function UserPage() {
  const location = useLocation();
  const serverResponse = location.state.data;

  return (
    <body className="user">
      <div>
        {serverResponse && (
          <div className="results-section">
            {Object.keys(serverResponse).map((filename) => (
              <div key={filename} className="image-card">
                <img
                  className="result-image"
                  src={`data:image/jpeg;base64,${serverResponse[filename].image}`}
                  alt={filename}
                />
                <div className="image-info">
                  <p>Match Score: {serverResponse[filename].points}</p>
                  <p>
                    {serverResponse[filename].strong_match
                      ? "Strong Match!"
                      : "Match"}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </body>
  );
}

export default UserPage;
