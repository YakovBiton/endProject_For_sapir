import React, { useState, useContext } from "react";
import { createUserWithEmailAndPassword } from "firebase/auth";
import { useNavigate } from "react-router-dom";
import { auth } from "../../firebase";
import { AuthContext } from "../../context/AuthContext"; // Path may vary
import "../Styles/SignUp.css";

function SignUp() {
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const { setCurrentUser } = useContext(AuthContext);

  const navigate = useNavigate();

  const signUp = async () => {
    try {
      const userCredential = await createUserWithEmailAndPassword(
        auth,
        email,
        password
      );
      const user = userCredential.user;
      // Mocking user information here. You should replace it with actual user data.
      setCurrentUser({
        email,
        firstName,
        lastName,
        name: `${firstName} ${lastName}`,
        profileImage: user.photoURL || "url-to-default-image",
      });
      console.log("User created successfully");
      navigate("/FindChild");
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <body className="signUpBody">
      <div className="boxStyle">
        <h1> Registration</h1>

        <input
          type="text"
          placeholder="First Name"
          value={firstName}
          onChange={(e) => setFirstName(e.target.value)}
        />

        <input
          type="text"
          placeholder="Last Name"
          value={lastName}
          onChange={(e) => setLastName(e.target.value)}
        />

        <input
          type="email"
          placeholder="Email"
          onChange={(e) => setEmail(e.target.value)}
        />

        <input
          type="password"
          placeholder="Password"
          onChange={(e) => setPassword(e.target.value)}
        />

        <div className="button-sec">
          <button className="btn" onClick={signUp}>
            Sign Up
          </button>
        </div>
      </div>
    </body>
  );
}

export default SignUp;
